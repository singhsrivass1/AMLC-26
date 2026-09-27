"""scripts/resolve_conflicts.py: greedy target-conflict resolution + tuned cut-off.

* The vectorized resolution equals a literal global greedy walk (sort all
  predictions by probability, give each target to its first claimant) on random
  data with ties, and commutes with the threshold.
* ``predict.py --scores-out`` -> ``resolve_conflicts.py`` reproduces predict's own
  keep-best submission byte for byte, applies a stricter cut-off correctly, and
  refuses a cut-off below the scores file's floor.
* ``--tune`` sweeps the cut-off on out-of-fold scores, with and without resolution,
  against a hand-computed macro F0.5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from src.data_loader import load_config  # noqa: E402
from src.submission import best_claim_mask  # noqa: E402

from test_dense_blocker import _make_tree  # noqa: E402
from test_v2_features import _write_predict_fixture  # noqa: E402


def _literal_greedy(targets, scores, tiebreak) -> np.ndarray:
    """The algorithm as specified: global sort by probability desc, first claimant wins."""
    order = sorted(range(len(targets)), key=lambda i: (-scores[i], tiebreak[i]))
    taken, keep = set(), np.zeros(len(targets), dtype=bool)
    for i in order:
        if targets[i] not in taken:
            taken.add(targets[i])
            keep[i] = True
    return keep


@pytest.mark.parametrize("seed", range(5))
def test_vectorized_resolution_equals_the_literal_global_greedy_walk(seed):
    rng = np.random.default_rng(seed)
    n = 400
    targets = rng.integers(0, 60, n)  # many conflicts
    scores = np.round(rng.random(n), 2)  # coarse -> many exact ties
    tiebreak = rng.permutation(n)
    assert np.array_equal(best_claim_mask(targets, scores, tiebreak), _literal_greedy(targets, scores, tiebreak))

    # threshold and resolution commute
    for threshold in (0.3, 0.7, 0.95):
        passed = scores >= threshold
        after = best_claim_mask(targets, scores, tiebreak) & passed
        before = np.zeros(n, dtype=bool)
        before[np.flatnonzero(passed)] = best_claim_mask(targets[passed], scores[passed], tiebreak[passed])
        assert np.array_equal(after, before), threshold


def _run(module: str, argv: list[str]) -> int:
    return __import__(f"scripts.{module}", fromlist=["main"]).main([*argv, "--log-level", "CRITICAL"])


ROWS = [
    ("S1-1", "S2-10", 0.95),
    ("S1-2", "S2-10", 0.90),  # conflicting, lower-probability claim on S2-10
    ("S1-3", "S2-14", 0.90),
    ("S1-4", "S2-12", 0.80),
    ("S1-4", "S2-13", 0.20),
]


def test_apply_reproduces_predict_and_honours_the_cutoff_and_the_floor(tmp_path):
    config_path = _write_predict_fixture(tmp_path, ROWS)
    config = ["--config", str(config_path), "--split", "train"]
    assert _run("predict", [*config, "--scores-out", "auto", "--scores-floor", "0.1"]) == 0
    submission = tmp_path / "work" / "submission"
    scores = (submission / "train_scores.tsv").read_text(encoding="utf-8").splitlines()
    assert scores[0] == "source1_entity_id\tmatched_entity_id\tprobability"
    assert len(scores) - 1 == len(ROWS), "raw scores are written before threshold and resolution"

    # at the bundle's own threshold, resolve_conflicts == predict's keep-best output
    assert _run("resolve_conflicts", [*config, "--threshold", "bundle"]) == 0
    resolved = submission / "resolved" / "train_matching_results.tsv"
    assert resolved.read_bytes() == (submission / "train_matching_results.tsv").read_bytes()
    report = json.loads((submission / "resolved" / "train_matching_results_report.json").read_text())
    assert report["conflicts"]["pairs_dropped"] == 1
    assert report["score"]["resolved"]["all"] > report["score"]["unresolved"]["all"]

    # a stricter F0.5 cut-off: S1-4's 0.80 claim no longer survives
    assert _run("resolve_conflicts", [*config, "--threshold", "0.85"]) == 0
    lines = dict(l.split("\t", 1) for l in resolved.read_text(encoding="utf-8").splitlines()[1:])
    assert lines == {"S1-1": "S2-10", "S1-2": "", "S1-3": "S2-14", "S1-4": ""}

    # below the floor the file is incomplete -> refused, not silently wrong
    assert _run("resolve_conflicts", [*config, "--threshold", "0.05"]) == 2


def test_tune_sweeps_out_of_fold_scores_with_and_without_resolution(tmp_path):
    config_path = _make_tree(tmp_path, dense=False)
    work = Path(load_config(config_path)["resolved"]["work_dir"])
    model_dir = work / "experiments" / "v1"
    (model_dir / "model").mkdir(parents=True)
    # Ground truth: S1-1 -> S2-10,S2-11,S3-20 ; S1-2 -> "" ; S1-3 -> S2-14 ; S1-4 -> S2-12.
    oof = [("S1-1", "S2-10", 0.9, 1), ("S1-2", "S2-10", 0.8, 0), ("S1-3", "S2-14", 0.6, 1), ("S1-4", "S2-12", 0.7, 1)]
    features = work / "features.tsv"
    features.write_text("source1_entity_id\tmatched_entity_id\n" + "".join(f"{a}\t{b}\n" for a, b, _, _ in oof),
                        encoding="utf-8")
    np.save(model_dir / "oof_probabilities.npy", np.array([r[2] for r in oof], dtype=np.float32))
    np.save(model_dir / "val_owner_index.npy", np.array([0, 1, 2, 3], dtype=np.int32))  # GT row order
    np.save(model_dir / "val_labels.npy", np.array([r[3] for r in oof], dtype=np.int8))
    (model_dir / "v1_labels_report.json").write_text(json.dumps({"features_path": str(features), "rows_read": 4}))
    (model_dir / "model" / "model_meta.json").write_text(json.dumps({"threshold": 0.85}))

    code = _run("resolve_conflicts", ["--config", str(config_path), "--split", "train", "--tune",
                                      "--tune-population", "all", "--threshold", "0.85"])
    assert code == 0
    summary = json.loads((work / "submission" / "resolved" / "threshold_tuning.json").read_text())
    s1_1 = 1.25 * (1 / 3) / (0.25 + 1 / 3)  # P=1, R=1/3
    # With resolution S1-2's false merge is gone at every cut-off: all four entities score.
    assert summary["best_resolved"]["macro_f05_resolved"] == pytest.approx((s1_1 + 1 + 1 + 1) / 4)
    assert summary["best_resolved"]["threshold"] <= 0.6
    # Without it, any cut-off low enough to keep S1-3/S1-4 also keeps the false merge.
    assert summary["best_unresolved"]["macro_f05_unresolved"] == pytest.approx((s1_1 + 0 + 1 + 1) / 4)
    # The requested 0.85 keeps only S1-1 (and S1-2 stays empty): F0.5 = (s1_1 + 1 + 0 + 0) / 4.
    assert summary["at_requested_threshold"]["macro_f05_resolved"] == pytest.approx((s1_1 + 1) / 4)

    # misaligned feature file -> refused
    features.write_text("source1_entity_id\tmatched_entity_id\nS1-4\tS2-12\n", encoding="utf-8")
    assert _run("resolve_conflicts", ["--config", str(config_path), "--split", "train", "--tune"]) == 2
