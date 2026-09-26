"""Regression tests for the dense blocker, the submission layer and artifact integrity.

Covers the fixes for audit findings C1-C4 and H2-H4:

* C4  the dense blocker: settings, build/save/load/query, union provenance and the
      end-to-end ``build_indexes -> generate_candidates`` path. These run on the
      ``hashing`` backend (deterministic char-trigram hashing), so they need no model
      download; ``tests/fixtures/synthetic_smoke_test.py`` runs the real bge-m3.
* C1  ``matching_results.tsv``: every S1 once, singletons an exact ``""``, strict
      validation, scoring against a hand-computed F0.5.
* C3  split-aware candidate paths (and the legacy read fallback).
* H2  a smoke-test (``--limit``) prepared table / index is never reused by a full run.
* H3  ``build_indexes.py`` defaults to the blockers enabled in config.
* H4  strict TSV: a literal ``"`` survives, and a malformed raw row fails the run.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.blocking import (  # noqa: E402
    BLOCKER_DENSE,
    BLOCKER_EXACT_NAME,
    BLOCKER_TOKEN,
    DenseIndex,
    build_index,
    enabled_blockers,
    index_dir_for,
    load_index,
    pack_pairs,
    resolve_blocker_settings,
    union_blockers,
    unpack_pairs,
)
from src.data_loader import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    GroundTruth,
    candidates_path,
    load_config,
    read_tsv,
)
from src.submission import (  # noqa: E402
    SubmissionError,
    score_submission,
    validate_submission,
    write_submission,
)
from src.utils import decode_entity_ids, encode_entity_id  # noqa: E402

LOG = logging.getLogger("test_dense_blocker")
LOG.setLevel(logging.CRITICAL)

RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
S1_ROWS = [
    ["S1-1", "Acme Holdings", "1 main st", "US"],
    ["S1-2", "Zyxwvut Quantum Aerospace", "9 far rd", "US"],
    ["S1-3", 'The "Best" Bakery', "5 oven ln", "US"],
    ["S1-4", "Quick Mart", "2 side st", "US"],
]
S2_ROWS = [
    ["S2-10", "Acme Holdings", "1 main st", "US"],
    ["S2-11", "Acme Holding", "1 main street", "US"],
    ["S2-12", "Quik Mart", "2 side st", "US"],
    ["S2-13", "Totally Unrelated Plumbing", "", "US"],
    ["S2-14", 'The "Best" Bakery', "5 oven ln", "US"],
]
S3_ROWS = [
    ["S3-20", "acme holdings inc", "1 main st", "US"],
    ["S3-21", "Harbor Freight Lines", "7 dock rd", "US"],
]


def _write_raw(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _make_tree(root: Path, dense: bool = True, token: bool = False) -> Path:
    """Raw train data plus a config whose every path points into ``root``."""
    for split in ("train",):
        _write_raw(root / split / f"{split}_source1.tsv", RAW_COLUMNS, S1_ROWS)
        _write_raw(root / split / f"{split}_source2.tsv", RAW_COLUMNS, S2_ROWS)
        _write_raw(root / split / f"{split}_source3.tsv", RAW_COLUMNS, S3_ROWS)
    _write_raw(
        root / "train" / "train_ground_truth.tsv",
        ["source1_entity_id", "matched_entity_ids"],
        [["S1-1", "S2-10,S2-11,S3-20"], ["S1-2", ""], ["S1-3", "S2-14"], ["S1-4", "S2-12"]],
    )
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["paths"].update(
        {
            "data_root": str(root / "train"),
            "test_data_root": str(root / "test"),
            "work_dir": str(root / "work"),
            "prepared_dir": str(root / "work" / "prepared"),
            "index_dir": str(root / "work" / "indexes"),
            "candidates_dir": str(root / "work" / "candidates"),
            "log_dir": str(root / "logs"),
        }
    )
    config["io"]["chunksize"] = 2  # exercise chunk boundaries
    config["compute"].update({"num_workers": 1, "device": "cpu"})
    config["blocking"]["token"]["enabled"] = token
    config["blocking"]["char_ngram"]["enabled"] = False
    config["blocking"]["dense"].update(
        {"enabled": dense, "backend": "hashing", "top_k": 2, "min_score": 0.3}
    )
    path = root / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _run(module_name: str, argv: list[str]) -> int:
    module = __import__(f"scripts.{module_name}", fromlist=["main"])
    return module.main([*argv, "--log-level", "CRITICAL"])


def _dense_settings(**overrides) -> dict:
    config = {"compute": {"device": "cpu"}, "blocking": {"dense": {"backend": "hashing", **overrides}}}
    return resolve_blocker_settings(config, BLOCKER_DENSE)


# ---------------------------------------------------------------------------
# C4: settings
# ---------------------------------------------------------------------------
def test_dense_settings_resolve_and_validate():
    settings = _dense_settings()
    assert settings["model_name_or_path"] == "BAAI/bge-m3"
    assert settings["local_files_only"] is True, "no network at runtime is the default"
    assert settings["device"] == "cpu", "device=auto must resolve through compute.device"
    for bad, message in [
        ({"top_k": 0}, "positive integer"),
        ({"min_score": 1.5}, "cosine"),
        ({"backend": "openai"}, "backend"),
        ({"tok_k": 5}, "unknown setting"),
    ]:
        with pytest.raises(ValueError, match=message):
            _dense_settings(**bad)


# ---------------------------------------------------------------------------
# C4: the index itself
# ---------------------------------------------------------------------------
def _frames(rows: list[list[str]], chunk: int = 2):
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS).rename(columns={"business_name": "name_norm"})
    frame["name_norm"] = frame["name_norm"].str.lower()
    return lambda: iter([frame.iloc[i : i + chunk] for i in range(0, len(frame), chunk)])


def test_dense_index_build_query_save_load_roundtrip(tmp_path):
    rows = S2_ROWS + [["S2-99", "", "", "US"]]  # an empty key is skipped, not embedded
    settings = _dense_settings(top_k=2, min_score=0.3)
    index = DenseIndex.build(_frames(rows), "source2", "S2", key_field="name_norm", log=LOG, **settings)
    assert index.n_keys == len(S2_ROWS)
    assert index.n_rows_without_key == 1

    queries = pd.Series(["acme holdings", "", "zzzz qqqq"], dtype=object)
    packed, evidence = index.query(queries)
    s1_positions, codes = unpack_pairs(packed)
    pairs = set(zip(s1_positions.tolist(), decode_entity_ids(codes).tolist()))
    assert (0, "S2-10") in pairs, "the identical name must be retrieved"
    assert all(position != 1 for position in s1_positions), "an empty query retrieves nothing"
    assert np.all(np.diff(packed) > 0), "output is sorted and unique, like every blocker's"
    cosine = evidence["dense_cosine"]
    assert cosine.shape == packed.shape
    assert np.all(cosine >= 0.3 - 1e-9) and np.all(cosine <= 1.0)
    assert np.bincount(s1_positions).max() <= 2, "top_k caps candidates per S1"
    exact = int(np.flatnonzero(packed == pack_pairs(np.array([0]), np.array([encode_entity_id("S2-10")])))[0])
    assert cosine[exact] == pytest.approx(1.0, abs=1e-3)

    index.save(tmp_path / "dense")
    reloaded = DenseIndex.load(tmp_path / "dense", settings=settings)
    again, again_evidence = reloaded.query(queries)
    assert np.array_equal(again, packed)
    np.testing.assert_allclose(again_evidence["dense_cosine"], cosine, atol=1e-6)

    # Query-time settings come from the config at load; no rebuild needed.
    strict = DenseIndex.load(tmp_path / "dense", settings={**settings, "min_score": 0.99})
    strict_packed, _ = strict.query(queries)
    assert set(strict_packed.tolist()) <= set(packed.tolist())
    assert len(strict_packed) < len(packed)


def test_dense_evidence_survives_the_union_next_to_a_lexical_blocker():
    s2_10, s2_11 = encode_entity_id("S2-10"), encode_entity_id("S2-11")
    exact = pack_pairs(np.array([0]), np.array([s2_10]))
    dense = pack_pairs(np.array([0, 0]), np.array([s2_10, s2_11]))
    positions, codes, provenance, evidence = union_blockers(
        {"source2:exact_name": exact, "source2:dense": dense},
        blocker_evidence={"source2:dense": {"dense_cosine": np.array([0.99, 0.81])}},
    )
    assert provenance.tolist() == ["source2:exact_name,source2:dense", "source2:dense"]
    np.testing.assert_allclose(evidence["dense_cosine"], [0.99, 0.81])


# ---------------------------------------------------------------------------
# C4 + H3: end to end through the real CLIs
# ---------------------------------------------------------------------------
def test_build_indexes_defaults_to_config_blockers_and_dense_reaches_the_candidates(tmp_path):
    config_path = _make_tree(tmp_path, dense=True, token=True)
    assert _run("prepare_data", ["--config", str(config_path), "--splits", "train"]) == 0
    # No --blockers: must build exactly what generate_candidates will load.
    assert _run("build_indexes", ["--config", str(config_path)]) == 0
    config = load_config(config_path)
    assert enabled_blockers(config) == [BLOCKER_EXACT_NAME, BLOCKER_TOKEN, BLOCKER_DENSE]
    for source in ("source2", "source3"):
        for blocker in enabled_blockers(config):
            assert (index_dir_for(config, "train", source, blocker) / "meta.json").is_file(), (source, blocker)

    assert _run("generate_candidates", ["--config", str(config_path), "--workers", "1"]) == 0
    table = read_tsv(candidates_path(config, "candidate_pairs", split="train"))
    assert "dense_cosine" in table.columns
    assert table["blockers"].str.contains(":dense").any()
    dense_only = table[table["blockers"].str.fullmatch(r"source\d:dense(,source\d:dense)*")]
    assert len(dense_only), "dense must propose pairs no lexical blocker proposed"
    assert (dense_only["dense_cosine"] != "").all()
    lexical_only = table[~table["blockers"].str.contains(":dense")]
    assert (lexical_only["dense_cosine"] == "").all(), "blank = dense did not propose the pair"


def test_exact_name_lookup_survives_a_chunk_with_an_empty_name():
    """Regression: a name that normalizes to "" crashed lookup_many (found by the smoke test)."""
    from src.blocking import ExactNameIndex

    frame = pd.DataFrame({"entity_id": ["S2-1", "S2-2"], "name_norm": ["acme", "quick mart"]})
    index = ExactNameIndex.build(iter([frame]), "source2", "S2", key_field="name_norm", log=LOG)
    positions, counts = index.lookup_many(pd.Series(["acme", "", "quick mart", "nope"], dtype=object))
    assert counts.tolist() == [1, 0, 1, 0]
    assert positions[1] == -1 and positions[3] == -1


# ---------------------------------------------------------------------------
# C3: split-aware candidate names
# ---------------------------------------------------------------------------
def test_candidate_paths_are_split_aware_with_a_legacy_read_fallback(tmp_path):
    config = load_config(_make_tree(tmp_path))
    train = candidates_path(config, "candidate_pairs", split="train", legacy_fallback=False)
    test = candidates_path(config, "candidate_pairs", split="test")
    assert train.name == "train_candidate_pairs.tsv"
    assert test.name == "test_candidate_pairs.tsv"
    assert train != test

    legacy = train.with_name("candidate_pairs.tsv")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("source1_entity_id\tmatched_entity_id\n", encoding="utf-8")
    assert candidates_path(config, "candidate_pairs", split="train") == legacy, "old HPC files stay readable"
    assert candidates_path(config, "candidate_pairs", split="test") == test, "never for test"
    assert candidates_path(config, "candidate_pairs", split="train", legacy_fallback=False) == train


# ---------------------------------------------------------------------------
# H2: smoke artifacts are never reused by a full run
# ---------------------------------------------------------------------------
def test_a_limited_prepare_and_index_are_rebuilt_by_the_full_run(tmp_path):
    config_path = _make_tree(tmp_path, dense=False)
    argv = ["--config", str(config_path), "--splits", "train"]
    assert _run("prepare_data", [*argv, "--limit", "2"]) == 0
    config = load_config(config_path)
    prepared = Path(config["resolved"]["prepared_dir"]) / "train_source2_norm.tsv"
    assert len(read_tsv(prepared)) == 2

    build_index(config, "train", "source2", BLOCKER_EXACT_NAME, limit=2, log=LOG)
    # The README order: smoke test first, then the full run WITHOUT --overwrite.
    assert _run("prepare_data", argv) == 0
    assert len(read_tsv(prepared)) == len(S2_ROWS), "the 2-row smoke table must not be reused"
    index = build_index(config, "train", "source2", BLOCKER_EXACT_NAME, log=LOG)
    assert index.n_entities_indexed == len(S2_ROWS), "the 2-row smoke index must not be reused"
    meta = json.loads((index_dir_for(config, "train", "source2", BLOCKER_EXACT_NAME) / "meta.json").read_text())
    assert meta["build"]["limit"] is None

    # A second full run is a genuine no-op...
    before = (index_dir_for(config, "train", "source2", BLOCKER_EXACT_NAME) / "meta.json").stat().st_mtime_ns
    build_index(config, "train", "source2", BLOCKER_EXACT_NAME, log=LOG)
    after = (index_dir_for(config, "train", "source2", BLOCKER_EXACT_NAME) / "meta.json").stat().st_mtime_ns
    assert before == after

    # ...and an index at another cell is refused at load, not silently used.
    build_index(config, "train", "source2", BLOCKER_TOKEN, log=LOG)
    config["blocking"]["token"]["df_cap"] = 5
    with pytest.raises(ValueError, match="df_cap"):
        load_index(config, "train", "source2", BLOCKER_TOKEN, verify=True)


# ---------------------------------------------------------------------------
# H4: strict TSV
# ---------------------------------------------------------------------------
def test_a_literal_quote_survives_and_a_malformed_raw_row_fails_the_run(tmp_path):
    config_path = _make_tree(tmp_path, dense=False)
    assert _run("prepare_data", ["--config", str(config_path), "--splits", "train"]) == 0
    config = load_config(config_path)
    s1 = read_tsv(Path(config["resolved"]["prepared_dir"]) / "train_source1_norm.tsv")
    assert len(s1) == len(S1_ROWS)
    assert s1.loc[s1["entity_id"] == "S1-3", "business_name"].item() == 'The "Best" Bakery'

    # A stray tab makes one line a 5-field row, which the parser skips with a warning.
    bad = S2_ROWS + [["S2-15", "Stray", "tab", "in", "US"]]
    _write_raw(tmp_path / "train" / "train_source2.tsv", RAW_COLUMNS, bad)
    code = _run("prepare_data", ["--config", str(config_path), "--splits", "train", "--sources", "source2"])
    assert code == 1, "a raw row the parser dropped must fail the run"
    assert (
        _run("prepare_data", ["--config", str(config_path), "--splits", "train", "--sources", "source2",
                              "--allow-row-loss"])
        == 0
    )


# ---------------------------------------------------------------------------
# C1: the submission
# ---------------------------------------------------------------------------
def test_submission_writes_every_entity_and_singletons_as_exact_empty_strings(tmp_path):
    path = tmp_path / "matching_results.tsv"
    universe = ["S1-1", "S1-2", "S1-3", "S1-4"]
    # S1-4 is absent from the predictions entirely (no candidates at all).
    stats = write_submission(path, universe, {"S1-1": ["S3-20", "S2-10", "S2-10"], "S1-3": []})
    assert stats == {"rows": 4, "singletons": 3, "matched_entities": 1, "pairs": 2}
    assert path.read_bytes() == (
        b"source1_entity_id\tmatched_entity_ids\n"
        b"S1-1\tS2-10,S3-20\n"
        b"S1-2\t\n"
        b"S1-3\t\n"
        b"S1-4\t\n"
    )
    assert validate_submission(path, universe) == {"rows": 4, "singletons": 3, "pairs": 2}
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert frame["matched_entity_ids"].tolist() == ["S2-10,S3-20", "", "", ""]

    with pytest.raises(SubmissionError, match="not in the S1 universe"):
        write_submission(tmp_path / "x.tsv", universe, {"S1-99": ["S2-1"]})


@pytest.mark.parametrize(
    "body, message",
    [
        (b"S1-1\tnan\nS1-2\t\n", "NA spelled as text"),
        (b"S1-1\t\n", "missing"),
        (b"S1-1\t\nS1-1\t\nS1-2\t\n", "more than once"),
        (b"S1-2\t\nS1-1\t\n", "file order"),
        (b"S1-1\tS2-1,S2-1\nS1-2\t\n", "duplicated matched ids"),
        (b"S1-1\t\r\nS1-2\t\r\n", "CR characters"),
    ],
)
def test_validation_rejects_every_way_a_submission_goes_wrong(tmp_path, body, message):
    path = tmp_path / "bad.tsv"
    path.write_bytes(b"source1_entity_id\tmatched_entity_ids\n" + body)
    with pytest.raises(SubmissionError, match=message):
        validate_submission(path, ["S1-1", "S1-2"])


def test_score_submission_matches_a_hand_computed_macro_f05(tmp_path):
    gt_path = tmp_path / "gt.tsv"
    _write_raw(gt_path, ["source1_entity_id", "matched_entity_ids"],
               [["S1-1", "S2-1,S2-2"], ["S1-2", ""], ["S1-3", "S3-1"], ["S1-4", ""]])
    ground_truth = GroundTruth.from_tsv(gt_path)
    path = tmp_path / "sub.tsv"
    # S1-1: 1 TP, 1 FP, 1 FN -> P=R=0.5 -> F0.5=0.5
    # S1-2: empty GT, predicted "" -> 1.0 ; S1-3: missed -> 0.0 ; S1-4: false merge -> 0.0
    write_submission(path, ["S1-1", "S1-2", "S1-3", "S1-4"],
                     {"S1-1": ["S2-1", "S2-9"], "S1-4": ["S2-5"]})
    score = score_submission(path, ground_truth)
    assert score["macro_f05_score_zero"] == pytest.approx((0.5 + 1.0 + 0.0 + 0.0) / 4)
    assert score["macro_f05_exclude"] == pytest.approx((0.5 + 0.0) / 2)
    assert score["true_positives"] == 1 and score["predicted_pairs"] == 3
