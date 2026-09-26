"""V2: per-S1 context features and target-conflict resolution.

* Context features (``scripts/extract_pair_features.py``): hand-computed rank / gap /
  ratio values including ties and missing values; entity-aligned batching; identical
  output at any batch size; and the guard that fails a run whose S1 groups were split.
* Conflict resolution (``src/submission.py`` + ``scripts/predict.py``): hand-computed
  keep-best semantics with a deterministic tie-break, the ground-truth multiplicity
  check, and the predict CLI end to end - the written file never gives one target to
  two S1 entities, and the train dry run scores both modes.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from scripts import extract_pair_features as epf  # noqa: E402
from src.data_loader import GroundTruth, load_config  # noqa: E402
from src.matching_model import FEATURE_COLUMNS, ID_COLUMNS, NON_FEATURE_COLUMNS, ModelBundle, save_bundle  # noqa: E402
from src.submission import (  # noqa: E402
    ground_truth_target_multiplicity,
    resolve_target_conflicts,
)

import test_pair_features as tpf  # noqa: E402  (the extractor's fixture)
from test_dense_blocker import _make_tree  # noqa: E402

LOG = logging.getLogger("test_v2_features")
LOG.setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# per-S1 context features
# ---------------------------------------------------------------------------
def test_context_features_hand_computed():
    features = pd.DataFrame(
        {
            "name_token_set_ratio": [0.9, 0.6, 0.9, 0.3, 0.5],
            "name_char3_jaccard": [0.5, 0.25, 0.0, 0.0, 0.8],
            "dense_cosine": [np.nan, 0.8, 0.6, np.nan, np.nan],
        }
    )
    s1 = np.array(["S1-1", "S1-1", "S1-1", "S1-1", "S1-2"], dtype=object)
    epf.add_s1_context_features(features, s1)
    get = lambda name: features[name].tolist()  # noqa: E731

    # ties share the best rank ("min"), and the next value skips (rank 3, not 2)
    assert get("s1ctx_name_token_set_ratio_rank") == [1, 3, 1, 4, 1]
    assert get("s1ctx_name_token_set_ratio_gap_to_best") == pytest.approx([0.0, 0.3, 0.0, 0.6, 0.0])
    assert get("s1ctx_name_token_set_ratio_ratio_to_max") == pytest.approx([1.0, 0.6 / 0.9, 1.0, 0.3 / 0.9, 1.0])
    mean = (0.9 + 0.6 + 0.9 + 0.3) / 4
    assert get("s1ctx_name_token_set_ratio_ratio_to_mean") == pytest.approx(
        [0.9 / mean, 0.6 / mean, 0.9 / mean, 0.3 / mean, 1.0]
    )
    # a group whose max/mean is 0 has no meaningful ratio -> NaN, not inf
    char_ratio = get("s1ctx_name_char3_jaccard_ratio_to_max")
    assert char_ratio[:4] == pytest.approx([1.0, 0.5, 0.0, 0.0]) and char_ratio[4] == 1.0

    # missing base values: excluded from the group, NaN context, never a fake rank
    dense_rank = features["s1ctx_dense_cosine_rank"]
    assert dense_rank.isna().tolist() == [True, False, False, True, True]
    assert dense_rank[1] == 1 and dense_rank[2] == 2
    assert features["s1ctx_dense_cosine_gap_to_best"][2] == pytest.approx(0.2)
    assert features["s1ctx_dense_cosine_ratio_to_mean"][1] == pytest.approx(0.8 / 0.7)
    assert np.isnan(features["s1ctx_dense_cosine_gap_to_best"][4]), "an all-missing group stays missing"


def test_context_features_are_declared_in_both_schemas():
    assert set(epf.S1_CONTEXT_FEATURES) <= set(epf.FEATURE_DTYPES)
    assert set(epf.S1_CONTEXT_FEATURES) <= set(FEATURE_COLUMNS)
    assert not set(epf.S1_CONTEXT_FEATURES) & set(epf.TEXT_DERIVED_FEATURES)
    assert not set(epf.S1_CONTEXT_FEATURES) & set(epf.UNIT_INTERVAL_FEATURES), "rank is not in [0, 1]"


def test_entity_batches_never_split_a_group_and_keep_row_order(tmp_path):
    groups = [("S1-1", 3), ("S1-2", 1), ("S1-3", 5), ("S1-4", 2)]
    rows = [(s1, f"S2-{i}") for s1, size in groups for i in range(size)]
    path = tmp_path / "c.tsv"
    pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_id"]).to_csv(path, sep="\t", index=False)
    for batch_size in (1, 2, 3, 4, 100):
        batches = list(epf.iter_entity_batches(path, batch_size))
        flat = [tuple(r) for b in batches for r in b.itertuples(index=False)]
        assert flat == rows, batch_size
        seen: set[str] = set()
        for batch in batches:
            ids = set(batch["source1_entity_id"])
            assert not ids & seen, f"a group was split at batch_size={batch_size}"
            seen |= ids


def test_context_output_is_identical_at_any_batch_size_and_split_groups_fail_the_run():
    fixture = tpf._fixture()
    try:
        small, large = fixture.root / "b1", fixture.root / "b1000"
        code_small, report_small = fixture.run_into(small, sample_fraction=1.0, feature_batch_size=1)
        code_large, report_large = fixture.run_into(large, sample_fraction=1.0, feature_batch_size=1000)
        assert code_small == code_large == 0
        assert report_small["integrity"]["s1_context_partial_groups"] == 0
        assert (small / "features.tsv").read_bytes() == (large / "features.tsv").read_bytes()
        header = (small / "features.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")
        assert [c for c in header if c.startswith("s1ctx_")] == list(epf.S1_CONTEXT_FEATURES)

        # The guard: plain fixed-size batches split groups -> the run must fail.
        original = epf.iter_entity_batches
        epf.iter_entity_batches = lambda path, size, columns=None: epf.iter_tsv(path, columns=columns, chunksize=1)
        try:
            code, report = fixture.run_into(fixture.root / "split", sample_fraction=1.0, feature_batch_size=1)
        finally:
            epf.iter_entity_batches = original
        assert report["integrity"]["s1_context_partial_groups"] > 0
        assert code == 1
    finally:
        fixture.close()


# ---------------------------------------------------------------------------
# target-conflict resolution
# ---------------------------------------------------------------------------
def test_keep_best_resolution_hand_computed():
    order = {"S1-A": 0, "S1-B": 1, "S1-C": 2}
    s1 = ["S1-A", "S1-B", "S1-A", "S1-B", "S1-C", "S1-C", "S1-B"]
    targets = ["S2-1", "S2-1", "S2-2", "S2-2", "S2-3", "S2-2", "S3-9"]
    scores = [0.90, 0.95, 0.70, 0.70, 0.60, 0.10, 0.50]
    keep, stats = resolve_target_conflicts(s1, targets, scores, order)
    # S2-1: B outscores A -> B. S2-2: A/B tie at 0.70 -> A (first in S1 order), C loses.
    # S2-3 and S3-9 are uncontested.
    assert keep.tolist() == [False, True, True, False, True, False, True]
    assert stats == {
        "pairs_in": 7,
        "targets": 4,
        "targets_with_conflicts": 2,
        "pairs_dropped": 3,
        "tied_best_scores": 1,
        "s1_left_without_matches": 0,
    }
    # an S1 whose only claim is lost ends up with no match (it will be written as "")
    keep, stats = resolve_target_conflicts(["S1-A", "S1-B"], ["S2-1", "S2-1"], [0.9, 0.4], order)
    assert keep.tolist() == [True, False] and stats["s1_left_without_matches"] == 1
    empty, stats = resolve_target_conflicts([], [], [], order)
    assert empty.size == 0 and stats["pairs_in"] == 0


def test_ground_truth_multiplicity(tmp_path):
    path = tmp_path / "gt.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-2\nS1-2\tS2-2\nS1-3\t\nS1-4\tS3-5\n",
        encoding="utf-8",
    )
    stats = ground_truth_target_multiplicity(GroundTruth.from_tsv(path))
    assert stats == {
        "distinct_targets": 3,
        "targets_with_multiple_s1": 1,
        "true_pairs_on_shared_targets": 2,
        "max_s1_per_target": 2,
    }


def _write_predict_fixture(root: Path, rows: list[tuple[str, str, float]]) -> Path:
    """Raw train data + a threshold-arm bundle + a feature file the bundle can score."""
    config_path = _make_tree(root, dense=False)
    config = load_config(config_path)
    work = Path(config["resolved"]["work_dir"])
    save_bundle(
        ModelBundle(model="threshold", threshold=0.5, score_feature="name_token_set_ratio"),
        work / "experiments" / "v1",
    )
    feature_dir = work / "experiments" / "step3_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    header = [*ID_COLUMNS, *FEATURE_COLUMNS, *NON_FEATURE_COLUMNS]
    with open(feature_dir / "features.tsv", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        for s1, target, score in rows:
            values = {name: "0" for name in FEATURE_COLUMNS}
            values["name_token_set_ratio"] = str(score)
            cells = [s1, target, target[:2], *[values[n] for n in FEATURE_COLUMNS], "1"]
            handle.write("\t".join(cells) + "\n")
    (feature_dir / "step3_features_report.json").write_text(
        json.dumps({"inputs": {"split": "train", "sample_fraction": 1.0, "population": "all", "limit_rows": None}}),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize("mode", ["keep-best", "off"])
def test_predict_resolves_conflicts_and_dry_run_scores_both_modes(tmp_path, mode):
    from scripts import predict

    # Ground truth (tests/test_dense_blocker.py): S1-1 -> S2-10,S2-11,S3-20; S1-2 -> "";
    # S1-3 -> S2-14; S1-4 -> S2-12. S2-10 is also (wrongly) claimed by S1-2, the
    # empty-GT entity - a false merge that zeroes S1-2 under score_zero.
    rows = [
        ("S1-1", "S2-10", 0.95),
        ("S1-2", "S2-10", 0.90),
        ("S1-3", "S2-14", 0.90),
        ("S1-4", "S2-12", 0.80),
        ("S1-4", "S2-13", 0.20),  # below the threshold: not a claim at all
    ]
    config_path = _write_predict_fixture(tmp_path, rows)
    code = predict.main(["--config", str(config_path), "--split", "train",
                         "--target-conflicts", mode, "--log-level", "CRITICAL"])
    assert code == 0
    work = tmp_path / "work" / "submission"
    lines = (work / "train_matching_results.tsv").read_text(encoding="utf-8").splitlines()[1:]
    written = dict(line.split("\t", 1) for line in lines)
    report = json.loads((work / "train_matching_results_report.json").read_text(encoding="utf-8"))
    conflicts = report["target_conflicts"]

    assert conflicts["targets_with_conflicts"] == 1 and conflicts["pairs_dropped"] == 1
    if mode == "keep-best":
        assert written == {"S1-1": "S2-10", "S1-2": "", "S1-3": "S2-14", "S1-4": "S2-12"}
        owners = [t for v in written.values() for t in v.split(",") if t]
        assert len(owners) == len(set(owners)), "a target was written for two S1 entities"
    else:
        assert written["S1-2"] == "S2-10", "off must emit every thresholded pair"
    # The dry run scored the other mode too, and the ground truth's own multiplicity.
    comparison = conflicts["comparison"]
    assert set(comparison) == {"keep-best", "off"}
    assert comparison["keep-best"]["all"] > comparison["off"]["all"]
    assert conflicts["ground_truth"]["targets_with_multiple_s1"] == 0
