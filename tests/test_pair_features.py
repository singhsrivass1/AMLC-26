"""Tests for ``scripts/extract_pair_features.py`` (the Step 3 de-risk experiment).

What this file is guarding
--------------------------
Every feature the first matcher will see is computed here, so the two mistakes
that would silently poison the model are the ones under test:

1. **A blank evidence cell read as a zero.** The union writes ``token_df`` blank
   for every pair the token blocker did not propose. That blank means "not
   measured", not "scored 0", and it must become NaN.
2. **A failed text join read as a measurement.** If an id is missing from the
   prepared file, the pair still gets a row (never dropped - one row per
   candidate pair is the contract), but every text-derived feature must be
   blanked, while provenance, evidence and the S1 candidate count - which come
   from the candidate file, not the join - must survive.

It also guards the two invariants the sampling design rests on: whole S1 entities
are kept or dropped together (even when a small chunk size splits an entity across
chunks), and the per-S1 row count in the sample agrees with the count phase 1
recorded. The agreement check is the whole-entity invariant, and it is exercised
directly by corrupting the sample file.

There is no ground-truth file anywhere in this fixture, deliberately: the split
comes from ``assign_splits``, the same pure function ``src/evaluation.py`` uses,
so nothing here can leak a label into a feature.

The last group of tests covers ``--workers``: the parallel layer may change *where*
a row is featurized, never *which* rows are selected or *what* is computed. The
sample is chosen once in the parent, before any worker exists, so a worker count
cannot move it; the partition is complete, disjoint and deterministic; and the
merge runs in worker order, so the output does not depend on which worker finished
first. ``--workers 1`` is held to the original single-process path, unchanged.

The fixture is synthetic and self-contained. Nothing reads the dataset, and every
file written goes into a temp directory - ``outputs/`` is never touched.

Runs standalone (``python tests/test_pair_features.py``) and under pytest.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import assign_splits  # noqa: E402
from src.utils import stable_hash64  # noqa: E402

import scripts.extract_pair_features as epf  # noqa: E402

RAPIDFUZZ_INSTALLED = importlib.util.find_spec("rapidfuzz") is not None
RATIO_FEATURES = ("name_token_set_ratio", "name_token_sort_ratio", "name_partial_ratio")

CANDIDATE_COLUMNS = [
    epf.CANDIDATE_S1_COLUMN,
    epf.CANDIDATE_TARGET_COLUMN,
    epf.CANDIDATE_SOURCE_COLUMN,
    epf.CANDIDATE_BLOCKERS_COLUMN,
    "token_df",
    "char_jaccard",
]

# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
N_S1 = 60
S2_NAMES = {"S2-1": "acme alpha limited", "S2-2": "acme beta limited",
            "S2-3": "acme gamma limited"}
S3_NAMES = {"S3-1": "acme delta limited"}


def _text_row(entity_id: str, name: str, address: str, country: str) -> dict:
    return {
        "entity_id": entity_id,
        epf.PREPARED_NAME_NORM: name,
        epf.PREPARED_NAME_KEY: name.replace(" ", ""),
        epf.PREPARED_ADDRESS_NORM: address,
        epf.PREPARED_COUNTRY_NORM: country,
    }


def _s1_row(index: int) -> dict:
    """An S1 record with every missing/differing case the fixture needs represented.

    Address: missing every fifth entity. Country: missing every seventh, and a
    different value every eleventh, so ``country_equal`` and ``country_missing``
    both see a 0 and a 1.
    """
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    address = "" if index % 5 == 0 else "1 main st bengaluru"
    if index % 7 == 0:
        country = ""
    elif index % 11 == 0:
        country = "us"
    else:
        country = "in"
    return _text_row(f"S1-{index}", f"acme {words[index % len(words)]} limited",
                     address, country)


def _prepared_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": [row["entity_id"] for row in rows],
            "business_name": [row[epf.PREPARED_NAME_NORM] for row in rows],
            "business_address": [row[epf.PREPARED_ADDRESS_NORM] for row in rows],
            "country": [row[epf.PREPARED_COUNTRY_NORM] for row in rows],
            epf.PREPARED_NAME_NORM: [row[epf.PREPARED_NAME_NORM] for row in rows],
            epf.PREPARED_NAME_KEY: [row[epf.PREPARED_NAME_KEY] for row in rows],
            epf.PREPARED_ADDRESS_NORM: [row[epf.PREPARED_ADDRESS_NORM] for row in rows],
            epf.PREPARED_COUNTRY_NORM: [row[epf.PREPARED_COUNTRY_NORM] for row in rows],
        }
    )


def _fixture_candidate_rows() -> list[tuple]:
    """Three pairs per S1 - both evidence columns get measured and blank rows.

    The blank evidence is the point: ``token_df`` is blank on every pair the token
    blocker did not propose and ``char_jaccard`` on every pair the char blocker did
    not propose, so a fixture where only one of them is ever blank could not catch
    a blank being read as a zero.
    """
    rows = []
    for index in range(N_S1):
        entity_id = f"S1-{index}"
        rows.append((entity_id, "S2-1", "S2", "source2:exact_name", "3", ""))
        rows.append((entity_id, "S2-2", "S2", "source2:token", "5", ""))
        rows.append((entity_id, "S3-1", "S3", "source3:char_ngram", "", "0.4444"))
        if index == 0:
            # A duplicated pair: both copies must survive and be counted.
            rows.append((entity_id, "S2-1", "S2", "source2:exact_name", "3", ""))
            # A target id that is absent from the prepared files.
            rows.append((entity_id, "S9-7", "S2", "source2:token", "5", ""))
            # Two blockers on one pair, comma-joined as union_blockers writes them.
            rows.append((entity_id, "S2-3", "S2", "source2:exact_name,source2:token", "7", "0.6"))
    return rows


def _write_fixture(root: Path) -> Path:
    prepared = root / "prepared"
    candidates = root / "candidates"
    for path in (prepared, candidates):
        path.mkdir(parents=True, exist_ok=True)

    s1_rows = [_s1_row(index) for index in range(N_S1)]
    _prepared_frame(s1_rows).to_csv(prepared / "train_source1_norm.tsv", sep="\t", index=False)
    _prepared_frame([_text_row("S2-1", S2_NAMES["S2-1"], "1 main st bengaluru", "in"),
                     _text_row("S2-2", S2_NAMES["S2-2"], "", ""),
                     _text_row("S2-3", S2_NAMES["S2-3"], "1 main st bengaluru", "in")]).to_csv(
        prepared / "train_source2_norm.tsv", sep="\t", index=False
    )
    _prepared_frame([_text_row("S3-1", S3_NAMES["S3-1"], "9 other rd pune", "in")]).to_csv(
        prepared / "train_source3_norm.tsv", sep="\t", index=False
    )

    pd.DataFrame(_fixture_candidate_rows(), columns=CANDIDATE_COLUMNS).to_csv(
        candidates / "candidate_pairs.tsv", sep="\t", index=False
    )

    config_path = root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project: {name: er, seed: 42}",
                f"paths: {{data_root: '{(root / 'raw').as_posix()}', "
                f"prepared_dir: '{prepared.as_posix()}', "
                f"index_dir: '{(root / 'indexes').as_posix()}', "
                f"candidates_dir: '{candidates.as_posix()}', "
                f"log_dir: '{(root / 'logs').as_posix()}'}}",
                "io: {chunksize: 50, prepared_format: tsv, candidates_format: tsv}",
                "columns: {entity_id: entity_id, name: business_name, "
                "address: business_address, country: country}",
                "evaluation: {split: {enabled: true, val_fraction: 0.2, mode: hash}, "
                "zero_match_policy: exclude}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _candidate_frame(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "candidates" / "candidate_pairs.tsv", sep="\t", dtype=str)


def _val_ids() -> set[str]:
    ids = pd.Series([f"S1-{index}" for index in range(N_S1)], dtype=object)
    labels = assign_splits(ids, val_fraction=0.2, mode="hash", seed=42)
    return {entity_id for entity_id, label in zip(ids, labels) if label == "val"}


class _Fixture:
    """Temp tree plus a logger; ``close`` detaches the handlers before deleting."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="step3_features_test_"))
        self.config_path = _write_fixture(self.root)
        self.out = self.root / "out"
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(epf.LOG_NAME)
        self.log.setLevel(logging.CRITICAL)

    @property
    def config(self) -> dict:
        return epf.load_config(str(self.config_path))

    def argv(self, out_dir: Path | None = None, **overrides) -> list[str]:
        """A command line for this fixture.

        ``out_dir`` picks the output directory (defaults to the fixture's own), so
        one fixture can run the same command twice and compare the two results -
        which is how the ``--workers`` regression tests work. A ``True`` override
        becomes a bare flag (for ``--cleanup-shards``), and ``False``/``None``
        leave the flag off.
        """
        argv = ["--config", str(self.config_path), "--split", "train",
                "--output-dir", str(out_dir or self.out)]
        # These tests pin the val-only de-risk sample, so they ask for it explicitly;
        # the CLI default is now every labelled entity (see the population tests).
        overrides.setdefault("population", "val")
        for key, value in overrides.items():
            flag = f"--{key.replace('_', '-')}"
            if value is True:
                argv.append(flag)
            elif value is False or value is None:
                continue
            else:
                argv += [flag, str(value)]
        return argv

    def args(self, **overrides):
        return epf.parse_args(self.argv(**overrides))

    def run(self, **overrides):
        return self.run_into(self.out, **overrides)

    def run_into(self, out_dir: Path, **overrides):
        """Run the CLI into ``out_dir`` and read back its report."""
        out_dir.mkdir(parents=True, exist_ok=True)
        code = epf.main(self.argv(out_dir=out_dir, **overrides))
        report = json.loads((out_dir / "step3_features_report.json").read_text())
        return code, report

    def features(self) -> pd.DataFrame:
        return pd.read_csv(self.out / "features.tsv", sep="\t", dtype=str)

    def sample(self) -> pd.DataFrame:
        return pd.read_csv(self.out / "sample_candidates.tsv", sep="\t", dtype=str)

    def close(self) -> None:
        # The handler holds the log file open, and on Windows an open file cannot
        # be deleted - detach it so each fixture gets a clean logger and its own
        # directory really does go away.
        for handler in list(self.log.handlers):
            self.log.removeHandler(handler)
            handler.close()
        shutil.rmtree(self.root, ignore_errors=True)


def _fixture() -> _Fixture:
    return _Fixture()


SORT_KEY = (
    epf.CANDIDATE_S1_COLUMN,
    epf.CANDIDATE_TARGET_COLUMN,
    epf.CANDIDATE_SOURCE_COLUMN,
)


def _features_at(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "features.tsv", sep="\t", dtype=str)


def _sample_at(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "sample_candidates.tsv", sep="\t", dtype=str)


def _data_lines(path: Path) -> list[str]:
    """Every line of a TSV except the header, with the newline stripped.

    Read and split with ``newline=""`` so a line is a line whatever the platform's
    terminator is, which keeps the worker-order comparison honest.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        lines = handle.read().split("\n")
    return [line.rstrip("\r") for line in lines[1:] if line.strip()]


def _header_line(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.readline().rstrip("\r\n")


class _FakeCpuAllocation:
    """Pretend this process was allocated ``cpus`` CPUs, for one block.

    ``--workers`` is validated against the CPUs the process may actually use, so a
    two-CPU machine would reject the four- and eight-worker cases before they ran -
    and the tests would silently stop covering the partition at those counts. The
    limit is therefore set explicitly here. Every test that sets it also asserts the
    value it set is the one that was accepted, so this cannot hide a real clamp.
    """

    def __init__(self, cpus: int) -> None:
        self.cpus = int(cpus)

    def __enter__(self) -> _FakeCpuAllocation:
        self._real = epf.available_cpu_count
        epf.available_cpu_count = lambda: self.cpus
        return self

    def __exit__(self, *exc_info) -> bool:
        epf.available_cpu_count = self._real
        return False


def _counts(n_entities: int, seed: int = 7) -> dict[str, int]:
    """A deliberately skewed per-entity candidate count, like the real file's."""
    return {f"S1-{index}": 1 + (index * seed) % 23 for index in range(n_entities)}


# ---------------------------------------------------------------------------
# pure helpers: the semantics of every similarity live here
# ---------------------------------------------------------------------------
def test_token_set_splits_and_dedupes():
    assert epf.token_set("acme alpha limited") == frozenset({"acme", "alpha", "limited"})
    assert epf.token_set("alpha alpha alpha") == frozenset({"alpha"})
    assert epf.token_set("") == frozenset()


def test_set_jaccard_hand_computed():
    left, right = frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})
    assert epf.set_jaccard(left, right) == 0.5
    assert epf.set_jaccard(left, left) == 1.0
    assert epf.set_jaccard(left, frozenset({"z"})) == 0.0
    # An empty side means "nothing to compare", never a perfect match.
    assert epf.set_jaccard(left, frozenset()) == 0.0
    assert epf.set_jaccard(frozenset(), frozenset()) == 0.0


def test_length_ratio_hand_computed():
    assert epf.length_ratio("abcd", "abcd") == 1.0
    assert epf.length_ratio("abcd", "ab") == 0.5
    assert epf.length_ratio("", "abcd") == 0.0


def test_first_token_hand_computed():
    assert epf.first_token("acme alpha limited") == "acme"
    assert epf.first_token("   spaced   out  ") == "spaced"
    assert epf.first_token("") == ""


def test_parse_provenance_hand_computed():
    # (exact, token, char, dense, n_blockers, unknown)
    assert epf.parse_provenance("source2:exact_name") == (1, 0, 0, 0, 1, 0)
    assert epf.parse_provenance("source2:token") == (0, 1, 0, 0, 1, 0)
    assert epf.parse_provenance("source3:char_ngram") == (0, 0, 1, 0, 1, 0)
    assert epf.parse_provenance("source2:dense") == (0, 0, 0, 1, 1, 0)
    assert epf.parse_provenance("source2:exact_name,source2:token") == (1, 1, 0, 0, 2, 0)
    assert epf.parse_provenance("source2:exact_name,source2:token,source3:char_ngram") == (1, 1, 1, 0, 3, 0)
    assert epf.parse_provenance("source2:char_ngram,source2:dense") == (0, 0, 1, 1, 2, 0)
    assert epf.parse_provenance("") == (0, 0, 0, 0, 0, 0)
    # An unrecognised label is counted, never silently dropped.
    assert epf.parse_provenance("source2:levenshtein") == (0, 0, 0, 0, 0, 1)
    # The label after the LAST ":" identifies the blocker, whatever the source.
    assert epf.parse_provenance("source3:token") == (0, 1, 0, 0, 1, 0)


def test_evidence_columns_follow_the_union():
    assert list(epf.EVIDENCE_FLOAT_COLUMNS) == list(epf.evidence_columns_for(epf.UNION_BLOCKERS))
    assert set(epf.EVIDENCE_FLOAT_COLUMNS) == {"token_df", "char_jaccard", "dense_cosine"}


def test_population_all_samples_every_entity_not_only_val():
    """H1: the default train population is every labelled entity."""
    fixture = _fixture()
    try:
        code, report = fixture.run(sample_fraction=1.0, population="all")
        assert code == 0
        sampled = set(fixture.sample()[epf.CANDIDATE_S1_COLUMN])
        in_file = set(_candidate_frame(fixture.root)[epf.CANDIDATE_S1_COLUMN])
        assert sampled == in_file
        assert sampled - _val_ids(), "population=all must reach beyond the val split"
        assert report["inputs"]["population"] == "all"
    finally:
        fixture.close()


def test_default_population_is_all_and_test_split_is_forced_to_all():
    """C2: --split test used to keep only val-hashed ids (~20% of the entities)."""
    assert epf.parse_args(["--split", "train"]).population == "all"
    assert epf.parse_args(["--split", "test", "--sample-fraction", "1.0"]).population == "all"
    mask = epf._sample_mask_for_ids(
        np.array([f"S1-{i}" for i in range(N_S1)], dtype=object), {}, {}, 1_000_000,
        logging.getLogger("t"), population="all",
    )
    assert mask.all(), "population=all at fraction 1.0 must keep every entity"
    for argv in (
        ["--split", "test", "--sample-fraction", "1.0", "--population", "val"],
        ["--split", "test", "--sample-fraction", "0.5"],
    ):
        try:
            epf.parse_args(argv)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{argv} must be rejected: it would drop test entities")


def test_unit_interval_guard_excludes_the_document_frequency():
    # token_df is a document frequency (values in the thousands); putting it in the
    # [0, 1] guard would make every run fail its own sanity check.
    assert "token_df" not in epf.UNIT_INTERVAL_FEATURES
    assert "char_jaccard" in epf.UNIT_INTERVAL_FEATURES
    assert set(RATIO_FEATURES) <= set(epf.UNIT_INTERVAL_FEATURES)


# ---------------------------------------------------------------------------
# build_features: exact per-column expectations
# ---------------------------------------------------------------------------
def _lookup(source: str, rows: list[dict]) -> epf.PreparedLookup:
    ids = np.array([row["entity_id"] for row in rows], dtype=object)
    columns = {
        column: np.array([row[column] for row in rows], dtype=object)
        for column in (epf.PREPARED_NAME_NORM, epf.PREPARED_NAME_KEY,
                       epf.PREPARED_ADDRESS_NORM, epf.PREPARED_COUNTRY_NORM)
    }
    return epf.PreparedLookup(source, ids, columns)


def _mini_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Identical name and address; token_df measured, char_jaccard not.
            ("S1-1", "S2-1", "S2", "source2:token", "3", ""),
            # One name token differs; the char blocker proposed it, the token one did not.
            ("S1-1", "S2-2", "S2", "source2:char_ngram", "", "0.6"),
            # Target id absent from the prepared source -> join failure.
            ("S1-2", "S2-9", "S2", "source2:token", "12", ""),
            # Source label that was never loaded -> join failure, unknown source.
            ("S1-2", "S9-1", "S9", "source9:token", "12", ""),
        ],
        columns=CANDIDATE_COLUMNS,
    )


def _mini_built() -> tuple[pd.DataFrame, dict]:
    s1 = _lookup("source1", [
        _text_row("S1-1", "acme alpha limited", "1 main st bengaluru", "in"),
        _text_row("S1-2", "gamma delta limited", "", ""),
    ])
    targets = {
        "S2": _lookup("source2", [
            _text_row("S2-1", "acme alpha limited", "1 main st bengaluru", "in"),
            _text_row("S2-2", "acme beta limited", "1 main st bengaluru", "in"),
        ]),
        "S3": _lookup("source3", [_text_row("S3-1", "acme gamma limited", "1 main st bengaluru", "in")]),
    }
    integrity: dict = {}
    features = epf.build_features(
        _mini_frame(), targets, s1, {"S1-1": 2, "S1-2": 2}, integrity
    )
    return features, integrity


def test_build_features_keeps_one_row_per_candidate_pair():
    features, _ = _mini_built()
    assert len(features) == len(_mini_frame())
    # Same order and same ids: nothing dropped, reordered or deduplicated.
    assert list(features[epf.CANDIDATE_TARGET_COLUMN]) == ["S2-1", "S2-2", "S2-9", "S9-1"]
    assert list(features[epf.CANDIDATE_S1_COLUMN]) == ["S1-1", "S1-1", "S1-2", "S1-2"]


def _approx(value: float, expected: float, tolerance: float = 1e-5) -> bool:
    return abs(float(value) - expected) <= tolerance


def test_build_features_name_columns_hand_computed():
    features, _ = _mini_built()
    row = features.iloc[0]
    assert row["name_norm_equal"] == 1
    assert row["name_key_equal"] == 1
    assert row["name_token_jaccard"] == 1.0
    assert row["name_length_ratio"] == 1.0
    assert row["name_token_count_diff"] == 0
    assert row["name_first_token_equal"] == 1
    assert row["name_char3_jaccard"] == 1.0

    # "acme alpha limited" vs "acme beta limited": one token differs.
    # 17 characters vs 18, 2 of 4 distinct tokens shared.
    row = features.iloc[1]
    assert row["name_norm_equal"] == 0
    assert row["name_token_jaccard"] == 0.5
    assert row["name_token_count_diff"] == 0
    assert row["name_first_token_equal"] == 1
    assert _approx(row["name_length_ratio"], 17 / 18)


def test_build_features_rapidfuzz_ratios():
    features, integrity = _mini_built()
    assert integrity["rapidfuzz_available"] == (1 if RAPIDFUZZ_INSTALLED else 0)
    if not RAPIDFUZZ_INSTALLED:
        # The fallback must be NaN, never a fabricated 0.
        for column in RATIO_FEATURES:
            assert features[column].isna().all(), column
        return

    for column in RATIO_FEATURES:
        assert features.iloc[0][column] == 1.0, column
    # Measured values for "acme alpha limited" vs "acme beta limited" - asserted
    # exactly so a rapidfuzz upgrade that changes a score shows up as a failure
    # rather than as silently different training data.
    assert _approx(features.iloc[1]["name_token_set_ratio"], 0.82758623)
    assert _approx(features.iloc[1]["name_token_sort_ratio"], 0.8)
    assert _approx(features.iloc[1]["name_partial_ratio"], 0.7647059)
    assert features.iloc[1]["name_token_set_ratio"] >= features.iloc[1]["name_token_sort_ratio"]


def test_token_set_ratio_scores_containment_as_a_perfect_match():
    """Why the ratio features cannot be used alone.

    Measured, not assumed: ``token_set_ratio("acme alpha limited", "acme alpha")``
    is 1.0, because one token set contains the other. A short trading name that is
    a strict prefix of a long one therefore looks like a perfect name match to that
    feature. The accompanying jaccard and length ratio are what keep containment
    from reading as a match, so a test that forgot them would not notice if the
    ratio feature were quietly doing all the work.
    """
    if not RAPIDFUZZ_INSTALLED:  # pragma: no cover - depends on the environment
        return
    lookups = {"S2": _lookup("source2", [_text_row("S2-1", "acme alpha", "", "")])}
    s1 = _lookup("source1", [_text_row("S1-1", "acme alpha limited", "", "")])
    frame = pd.DataFrame(
        [("S1-1", "S2-1", "S2", "source2:token", "1", "0.0")],
        columns=CANDIDATE_COLUMNS,
    )
    row = epf.build_features(frame, lookups, s1, {"S1-1": 1}, {}).iloc[0]
    assert row["name_token_set_ratio"] == 1.0
    assert _approx(row["name_token_jaccard"], 2 / 3)
    assert _approx(row["name_length_ratio"], 10 / 18)
    assert row["name_norm_equal"] == 0


def test_build_features_address_columns_hand_computed():
    features, _ = _mini_built()
    row = features.iloc[0]
    assert row["address_norm_equal"] == 1
    assert row["address_token_jaccard"] == 1.0
    assert row["address_shared_token_count"] == 4     # 1 / main / st / bengaluru
    assert row["address_length_ratio"] == 1.0
    assert row["s1_address_missing"] == 0
    assert row["target_address_missing"] == 0
    assert row["both_address_missing"] == 0


def test_build_features_blank_evidence_becomes_nan_not_zero():
    features, integrity = _mini_built()
    # Row 0: token blocker measured token_df; char blocker did not propose the pair.
    assert features.iloc[0]["token_df"] == 3.0
    assert np.isnan(features.iloc[0]["char_jaccard"])
    # Row 1: the reverse.
    assert np.isnan(features.iloc[1]["token_df"])
    assert features.iloc[1]["char_jaccard"] == 0.6
    # One blank token_df (row 1) and three blank char_jaccard (rows 0, 2, 3).
    assert integrity["token_df_blank"] == 1
    assert integrity["char_jaccard_blank"] == 3


def test_build_features_join_failure_blanks_text_but_keeps_file_evidence():
    features, integrity = _mini_built()
    row = features.iloc[2]              # S2-9 is not in the prepared source
    assert row["text_join_ok"] == 0
    # Text-derived features are blanked...
    assert row["name_norm_equal"] == 0
    assert row["name_token_count_diff"] == 0
    assert np.isnan(row["name_token_jaccard"])
    assert np.isnan(row["name_char3_jaccard"])
    assert np.isnan(row["name_length_ratio"])
    assert np.isnan(row["address_token_jaccard"])
    # ...but provenance, evidence and the S1 candidate count come from the
    # candidate file, so they stay valid and must survive.
    assert row["blocker_token"] == 1
    assert row["n_blockers"] == 1
    assert row["token_df"] == 12.0
    assert row["s1_candidate_count"] == 2

    assert integrity["s1_join_failures"] == 0
    assert integrity["target_join_failures"] == 1     # row 2 only
    assert integrity["unknown_source_labels"] == 1    # row 3
    assert features.iloc[3]["text_join_ok"] == 0
    assert features.iloc[3]["token_df"] == 12.0


def test_build_features_blank_text_is_not_evidence_of_a_match():
    lookups = {"S2": _lookup("source2", [_text_row("S2-1", "acme alpha limited", "", "in")])}
    s1 = _lookup("source1", [_text_row("S1-1", "", "", "")])
    frame = pd.DataFrame(
        [("S1-1", "S2-1", "S2", "source2:exact_name", "1", "0.0")],
        columns=CANDIDATE_COLUMNS,
    )
    features = epf.build_features(frame, lookups, s1, {"S1-1": 1}, {})
    row = features.iloc[0]
    # The join SUCCEEDED, so this is not the blanked-on-failure path: a genuinely
    # empty field must still not read as "the two records agree".
    assert row["text_join_ok"] == 1
    assert row["name_norm_equal"] == 0
    assert row["name_key_equal"] == 0
    assert row["address_norm_equal"] == 0
    assert row["country_equal"] == 0
    assert row["country_missing"] == 1
    assert row["name_length_ratio"] == 0.0
    assert row["address_token_jaccard"] == 0.0


def test_build_features_dtypes_match_the_declaration():
    features, _ = _mini_built()
    for column, dtype in epf.FEATURE_DTYPES.items():
        assert str(features[column].dtype) == dtype, (column, features[column].dtype, dtype)


def test_build_features_unit_interval_features_stay_in_range():
    features, _ = _mini_built()
    for column in epf.UNIT_INTERVAL_FEATURES:
        values = features[column].to_numpy(dtype=np.float64)
        finite = values[~np.isnan(values)]
        assert finite.size, column
        assert finite.min() >= 0.0 and finite.max() <= 1.0, (column, finite.min(), finite.max())


def test_build_features_has_no_ground_truth_column():
    features, _ = _mini_built()
    forbidden = {"label", "is_match", "is_true", "target", "y", "ground_truth",
                 "true_match", "match", "score"}
    assert not (set(features.columns) & forbidden)
    # The frame is exactly the declared features plus the three id columns.
    extra = set(features.columns) - set(epf.FEATURE_DTYPES)
    assert extra == {epf.CANDIDATE_S1_COLUMN, epf.CANDIDATE_TARGET_COLUMN,
                     epf.CANDIDATE_SOURCE_COLUMN}


# ---------------------------------------------------------------------------
# PreparedLookup
# ---------------------------------------------------------------------------
def test_prepared_lookup_take_and_values():
    lookup = _lookup("source2", [
        _text_row("S2-1", "acme alpha limited", "1 main st", "in"),
        _text_row("S2-2", "beta limited", "", ""),
    ])
    positions, found = lookup.take(np.array(["S2-2", "S2-9", "S2-1"], dtype=object))
    assert list(positions) == [1, -1, 0]
    assert list(found) == [True, False, True]
    assert list(lookup.values(epf.PREPARED_NAME_NORM, positions, found)) == [
        "beta limited", "", "acme alpha limited"
    ]
    assert lookup.values(epf.PREPARED_NAME_NORM, np.array([-1]), np.array([False]))[0] == ""
    assert lookup.n_entities == 2


def test_prepared_lookup_memory_counts_the_string_payload():
    lookup = _lookup("source2", [
        _text_row(f"S2-{index}", "acme alpha limited", "1 main st bengaluru", "in")
        for index in range(500)
    ])
    # A naive nbytes-only estimate would be a few kB; the strings are the point.
    assert lookup.memory_bytes() > 500 * 40


# ---------------------------------------------------------------------------
# sampling: whole entities, deterministic, and a different hash slice than the split
# ---------------------------------------------------------------------------
def test_sample_mask_keeps_whole_entities_and_agrees_with_assign_splits():
    fixture = _fixture()
    try:
        config = fixture.config
        ids = np.array([f"S1-{index}" for index in range(N_S1)], dtype=object)

        labels = assign_splits(pd.Series(ids, dtype=object), val_fraction=0.2, mode="hash", seed=42)
        # Sample everything: the kept set is then exactly the validation split, which
        # is the split the evaluator uses - so the sample cannot straddle it.
        kept = epf._sample_mask_for_ids(ids.copy(), {}, config, 1_000_000, fixture.log)
        assert list(kept) == list(labels == "val")
        assert 0 < kept.sum() < len(ids)

        partial = epf._sample_mask_for_ids(ids.copy(), {}, config, 300_000, fixture.log)
        assert (partial & ~kept).sum() == 0, "the subsample must be a subset of the val set"
        assert 0 < partial.sum() < kept.sum()

        # Deterministic from a cold cache, and a warm cache never changes an answer.
        assert list(epf._sample_mask_for_ids(ids.copy(), {}, config, 300_000, fixture.log)) == list(partial)
        mixed: dict[str, int] = {}
        assert list(epf._sample_mask_for_ids(ids[:5].copy(), mixed, config, 300_000, fixture.log)) == list(partial[:5])
        assert list(epf._sample_mask_for_ids(ids.copy(), mixed, config, 300_000, fixture.log)) == list(partial)
    finally:
        fixture.close()


def test_subsample_bucket_is_a_different_hash_slice_than_the_split():
    """The two decisions must read different bits of the same hash.

    If they read the same bits, "val" would already imply "kept" and the
    subsample could not select a fraction of the val entities at all.
    """
    hashed = stable_hash64(pd.Series([f"S1-{index}" for index in range(400)], dtype=object))
    high = (hashed // np.uint64(1_000_000)) % np.uint64(1_000_000)
    low = hashed % np.uint64(1_000_000)
    assert list(high) != list(low)
    assert high.min() < 300_000 < high.max()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_main_writes_every_output_and_exits_zero():
    fixture = _fixture()
    try:
        code, report = fixture.run(sample_fraction=1.0)
        assert code == 0, report["integrity"]
        for name in ("sample_candidates.tsv", "features.tsv",
                     "feature_missingness.csv", "step3_features_report.json",
                     "extract_pair_features.log"):
            assert (fixture.out / name).is_file(), name
        assert report["integrity"]["count_mismatches"] == 0
        assert report["integrity"]["rapidfuzz_available"] == (1 if RAPIDFUZZ_INSTALLED else 0)
    finally:
        fixture.close()


def test_sample_is_exactly_the_val_entities():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        candidates = _candidate_frame(fixture.root)
        val = _val_ids()
        sample = fixture.sample()

        # Every sampled entity is in val, and every val entity is sampled.
        assert set(sample[epf.CANDIDATE_S1_COLUMN]) == val

        # Whole entities: each kept entity contributes exactly its file rows, which
        # is also the proof that no partial entity slipped through.
        expected = candidates[candidates[epf.CANDIDATE_S1_COLUMN].isin(val)]
        assert len(sample) == len(expected)
        assert sample[epf.CANDIDATE_S1_COLUMN].value_counts().to_dict() == (
            expected[epf.CANDIDATE_S1_COLUMN].value_counts().to_dict()
        )
    finally:
        fixture.close()


def test_whole_entities_survive_a_chunk_boundary():
    """A 7-row chunk size splits entities across chunks; the sample must not care."""
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0, chunksize=7)
        sample = fixture.sample()
        expected = _candidate_frame(fixture.root)
        val = _val_ids()
        assert set(sample[epf.CANDIDATE_S1_COLUMN]) == val
        assert len(sample) == len(expected[expected[epf.CANDIDATE_S1_COLUMN].isin(val)])
        # The per-entity count recorded for the feature came from the whole file,
        # not from the chunk it happened to arrive in, so it must match the file's
        # own count for that entity.
        file_counts = expected[epf.CANDIDATE_S1_COLUMN].value_counts().to_dict()
        seen = (fixture.features()[[epf.CANDIDATE_S1_COLUMN, "s1_candidate_count"]]
                .drop_duplicates().set_index(epf.CANDIDATE_S1_COLUMN)["s1_candidate_count"])
        assert set(seen.index) == val
        for entity_id, value in seen.items():
            assert int(value) == file_counts[entity_id], (entity_id, value, file_counts[entity_id])
    finally:
        fixture.close()


def test_one_row_per_candidate_pair_is_preserved_end_to_end():
    fixture = _fixture()
    try:
        code, report = fixture.run(sample_fraction=1.0)
        assert code == 0
        sample = fixture.sample()
        features = fixture.features()
        assert len(features) == len(sample)
        assert list(features[epf.CANDIDATE_TARGET_COLUMN]) == list(sample[epf.CANDIDATE_TARGET_COLUMN])
        assert list(features[epf.CANDIDATE_S1_COLUMN]) == list(sample[epf.CANDIDATE_S1_COLUMN])
        assert report["sample"]["n_candidate_pairs_sampled"] == len(features)
        assert report["sample"]["n_s1_entities_sampled"] == len(_val_ids())
    finally:
        fixture.close()


def test_duplicate_pairs_are_counted_not_collapsed():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        candidates = _candidate_frame(fixture.root)
        sample = fixture.sample()
        keys = list(zip(sample[epf.CANDIDATE_S1_COLUMN], sample[epf.CANDIDATE_TARGET_COLUMN]))
        expected_duplicates = len(keys) - len(set(keys))
        report = json.loads((fixture.out / "step3_features_report.json").read_text())
        assert report["integrity"]["duplicate_sampled_pairs"] == expected_duplicates
        # S1-0 is the entity carrying the duplicated pair, and both copies survive.
        if "S1-0" in _val_ids():
            assert expected_duplicates == 1
            assert keys.count(("S1-0", "S2-1")) == 2
        else:
            assert expected_duplicates == 0
    finally:
        fixture.close()


def test_join_failures_are_kept_and_counted():
    fixture = _fixture()
    try:
        code, report = fixture.run(sample_fraction=1.0)
        assert code == 0
        features = fixture.features()
        if "S1-0" in _val_ids():
            broken = features[features[epf.CANDIDATE_TARGET_COLUMN] == "S9-7"]
            assert len(broken) == 1, "a pair whose target is missing must still get a row"
            assert broken.iloc[0]["text_join_ok"] == "0"
            assert report["integrity"]["target_join_failures"] == 1
        else:
            assert report["integrity"]["target_join_failures"] == 0
        assert report["integrity"]["unknown_source_labels"] == 0
    finally:
        fixture.close()


def test_missingness_records_blank_evidence_not_zero():
    fixture = _fixture()
    try:
        _, report = fixture.run(sample_fraction=1.0)
        missingness = report["features"]["missingness"]
        # Every declared feature is reported.
        assert set(missingness) == set(epf.FEATURE_DTYPES)
        for column, entry in missingness.items():
            assert 0.0 <= entry["rate"] <= 1.0, column
            assert entry["dtype"] == epf.FEATURE_DTYPES[column], column
            assert entry["out_of_unit_range"] == 0, column

        # The report's missing counts must agree with the written matrix, so a
        # missingness figure can never describe a file other than the one on disk.
        features = fixture.features()
        for column in epf.FEATURE_DTYPES:
            assert missingness[column]["count"] == int(features[column].isna().sum()), column

        # Half the fixture's pairs have a blank token_df and the other half a blank
        # char_jaccard, so neither may be silently turned into 0.
        assert missingness["token_df"]["rate"] > 0.0
        assert missingness["char_jaccard"]["rate"] > 0.0
        # Integer features are never NaN: a blanked int is set to 0, deliberately,
        # because NaN is not representable in an integer column.
        for column, dtype in epf.FEATURE_DTYPES.items():
            if dtype != "float32":
                assert missingness[column]["count"] == 0, column
    finally:
        fixture.close()


def test_missingness_csv_lists_every_feature():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        csv = pd.read_csv(fixture.out / "feature_missingness.csv", sep="\t")
        assert list(csv["feature"]) == list(epf.FEATURE_DTYPES)
        assert set(csv.columns) >= {"feature", "dtype", "missing_rate", "missing_count"}
    finally:
        fixture.close()


def test_report_carries_every_required_measurement():
    fixture = _fixture()
    try:
        _, report = fixture.run(sample_fraction=1.0)
        assert report["sample"]["n_s1_entities_sampled"] > 0
        assert report["sample"]["n_candidate_pairs_sampled"] > 0
        assert report["timing"]["feature_pairs_per_sec"] > 0
        assert report["timing"]["total_seconds"] > 0
        assert report["memory"]["peak_rss_bytes"] is None or report["memory"]["peak_rss_bytes"] > 0
        assert report["memory"]["peak_rss_source"]
        assert report["features"]["dtypes"] and report["features"]["matrix_bytes"] > 0
        assert report["features"]["matrix_size"]
        assert report["features"]["missingness"]
        # The report must say which columns are features and which are diagnostics,
        # so a trainer never has to guess whether text_join_ok is trainable.
        assert set(report["features"]["non_feature_columns"]) == {"text_join_ok"}
        assert report["features"]["n_features"] == len(epf.FEATURE_DTYPES) - 1
        assert set(report["integrity"]["matrix_columns"]) == set(epf.FEATURE_DTYPES)
        assert not (set(report["features"]["non_feature_columns"])
                    & (set(epf.FEATURE_DTYPES) - {"text_join_ok"}))
        assert report["integrity"]["s1_join_failures"] == 0
        assert "duplicate_sampled_pairs" in report["integrity"]
        assert report["outputs"]["features"]["bytes"] > 0
        assert report["extrapolation"]["full_candidate_pairs"] == 336_056_756
        assert report["extrapolation"]["matrix_size_full"]
        assert report["extrapolation"]["matrix_fits_in_ram"] in (True, False)
        # The projections are labelled with how much they can be trusted, and the
        # caveats that make them upper bounds are stated, not implied.
        assert report["extrapolation"]["confidence"]
        assert len(report["extrapolation"]["assumptions"]) >= 2
        assert report["extrapolation"]["scan_timing_is_a_measurement"] is True
    finally:
        fixture.close()


def test_limit_rows_labels_the_scan_projection_as_an_extrapolation():
    fixture = _fixture()
    try:
        _, report = fixture.run(sample_fraction=1.0, limit_rows=40)
        assert report["extrapolation"]["scan_timing_is_a_measurement"] is False
        assert any("NOT a measurement" in text for text in report["extrapolation"]["assumptions"])
    finally:
        fixture.close()


def test_count_mismatch_is_detected():
    """The whole-entity invariant: phase 1's per-S1 count must equal phase 2's.

    Dropping a row out of the sample file is exactly what a partial entity would
    look like, and it must be reported rather than silently trained on.
    """
    fixture = _fixture()
    try:
        args = fixture.args(sample_fraction=1.0)
        config = fixture.config
        scan = epf.scan_and_sample(config, args, fixture.out, fixture.log)
        assert scan["rows_sampled"] > 1

        path = scan["sample_path"]
        frame = pd.read_csv(path, sep="\t", dtype=str)
        frame.iloc[:-1].to_csv(path, sep="\t", index=False)

        features = epf.extract_features(config, args, scan, fixture.out, fixture.log)
        assert features["integrity"]["count_mismatches"] == 1
        assert features["integrity"]["count_mismatch_examples"]
    finally:
        fixture.close()


def test_empty_sample_fails_loudly():
    fixture = _fixture()
    try:
        args = fixture.args(sample_fraction=0.0)
        try:
            epf.scan_and_sample(fixture.config, args, fixture.out, fixture.log)
        except RuntimeError as exc:
            assert "sample" in str(exc)
        else:  # pragma: no cover - the assertion below is the real check
            raise AssertionError("an empty sample must not pass silently")
    finally:
        fixture.close()


def test_two_runs_are_byte_identical():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0, chunksize=7)
        first_sample = (fixture.out / "sample_candidates.tsv").read_bytes()
        first_features = (fixture.out / "features.tsv").read_bytes()
        fixture.run(sample_fraction=1.0, chunksize=11)
        # Different chunk sizes must not change a single byte of the output.
        assert (fixture.out / "sample_candidates.tsv").read_bytes() == first_sample
        assert (fixture.out / "features.tsv").read_bytes() == first_features
    finally:
        fixture.close()


def test_subsample_is_a_subset_of_the_full_sample():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        small_out = fixture.root / "out_small"
        epf.main(["--config", str(fixture.config_path), "--split", "train",
                  "--sample-fraction", "0.30", "--population", "val",
                  "--output-dir", str(small_out)])
        small = pd.read_csv(small_out / "sample_candidates.tsv", sep="\t", dtype=str)
        whole = fixture.sample()
        assert 0 < len(small) < len(whole)
        assert set(small[epf.CANDIDATE_S1_COLUMN]) <= set(whole[epf.CANDIDATE_S1_COLUMN])
    finally:
        fixture.close()


def test_output_directory_defaults_outside_the_candidate_directory():
    """The experiment must never write into outputs/candidates."""
    fixture = _fixture()
    try:
        args = epf.parse_args(["--config", str(fixture.config_path)])
        assert args.output_dir is None
        candidate_dir = Path(fixture.config["resolved"]["candidates_dir"])
        default = candidate_dir.parent / "experiments" / "step3_features"
        assert candidate_dir not in default.parents
        assert default != candidate_dir
    finally:
        fixture.close()


# ---------------------------------------------------------------------------
# --workers: one computation, two execution layers
# ---------------------------------------------------------------------------
# Phase 1 (scan and sample) stays in the parent whatever --workers is, so the
# selection cannot depend on the worker count. Phase 2 is partitioned by whole S1
# entity into contiguous row-balanced shards, each worker loads only the prepared
# text its own shard can join to, and the merge concatenates the worker files in
# worker order. These tests hold the parallel path to that design.
def test_workers1_selects_the_same_s1_entities():
    """The sample is a function of the corpus alone - never of the worker count."""
    fixture = _fixture()
    try:
        one_dir, two_dir = fixture.root / "out_w1", fixture.root / "out_w2"
        with _FakeCpuAllocation(4):
            code_one, one = fixture.run_into(one_dir, sample_fraction=1.0, workers=1)
            code_two, two = fixture.run_into(two_dir, sample_fraction=1.0, workers=2)
        assert code_one == 0 and code_two == 0

        # Phase 1 writes the sample before a worker exists, so the two runs produce
        # byte-identical sample files - not merely the same set of entities.
        assert (two_dir / "sample_candidates.tsv").read_bytes() == (
            one_dir / "sample_candidates.tsv"
        ).read_bytes()
        assert two["sample"] == one["sample"]
        assert one["inputs"]["workers"] == 1 and two["inputs"]["workers"] == 2

        # And it is still exactly the validation entities: the sampling rule this
        # experiment shipped with, untouched by the parallel layer.
        sample = _sample_at(two_dir)
        assert set(sample[epf.CANDIDATE_S1_COLUMN]) == _val_ids()
        counts = sample[epf.CANDIDATE_S1_COLUMN].value_counts().to_dict()
        assert len(counts) == two["sample"]["n_s1_entities_sampled"] == len(_val_ids())
    finally:
        fixture.close()


def test_partition_has_no_duplicate_entities():
    """No S1 entity is handled twice, at any worker count - whole entities once."""
    counts = _counts(37)
    for workers in (1, 2, 4, 8, 10):
        groups = epf.partition_entities(counts, workers)
        assert len(groups) == workers
        flat = [entity for group in groups for entity in group]
        assert len(flat) == len(set(flat)), f"an entity went to two workers at {workers}"
        assert len(flat) == len(counts)
        assert set(flat) == set(counts)


def test_workers2_covers_exactly_the_workers1_entities():
    """The union of the workers is the whole selection, and nothing else."""
    counts = _counts(37)
    one = epf.partition_entities(counts, 1)
    two = epf.partition_entities(counts, 2)
    # workers=1 is the identity partition, in file order.
    assert one == [list(counts)]
    assert set(two[0]) | set(two[1]) == set(counts)
    assert not (set(two[0]) & set(two[1]))

    fixture = _fixture()
    try:
        one_dir, two_dir = fixture.root / "out_w1", fixture.root / "out_w2"
        with _FakeCpuAllocation(4):
            _, one_report = fixture.run_into(one_dir, sample_fraction=1.0, workers=1)
            _, two_report = fixture.run_into(two_dir, sample_fraction=1.0, workers=2)
        first = _features_at(one_dir)
        second = _features_at(two_dir)
        assert set(second[epf.CANDIDATE_S1_COLUMN]) == set(first[epf.CANDIDATE_S1_COLUMN])
        assert len(set(second[epf.CANDIDATE_S1_COLUMN])) == (
            one_report["sample"]["n_s1_entities_sampled"]
        )
        # Every entity's pairs are featurized too, not just the entity id written out.
        assert len(second) == len(first) == one_report["sample"]["n_candidate_pairs_sampled"]
        assert two_report["sample"]["n_candidate_pairs_sampled"] == (
            one_report["sample"]["n_candidate_pairs_sampled"]
        )
        # Each worker's own entity count is reported, and they sum to the selection.
        assert sum(two_report["parallel"]["n_s1_entities_per_worker"]) == (
            one_report["sample"]["n_s1_entities_sampled"]
        )
    finally:
        fixture.close()


def test_worker_outputs_share_the_single_process_schema():
    """A worker writes the same columns in the same order as the single process."""
    fixture = _fixture()
    try:
        one_dir, two_dir = fixture.root / "out_w1", fixture.root / "out_w2"
        with _FakeCpuAllocation(2):
            _, one = fixture.run_into(one_dir, sample_fraction=1.0, workers=1)
            code, two = fixture.run_into(two_dir, sample_fraction=1.0, workers=2)
        assert code == 0, two["integrity"]

        expected = [epf.CANDIDATE_S1_COLUMN, epf.CANDIDATE_TARGET_COLUMN,
                    epf.CANDIDATE_SOURCE_COLUMN, *epf.FEATURE_DTYPES]
        header = _header_line(two_dir / "features.tsv")
        assert header.split("\t") == expected
        assert header == _header_line(one_dir / "features.tsv")

        # Every worker's own file carries that header, so the merge only ever strips
        # a duplicate of a known header rather than inventing one.
        worker_dir = Path(two["parallel"]["shard_dir"])
        assert worker_dir.name == epf.WORKER_DIR_NAME
        assert worker_dir.parent.resolve() == two_dir.resolve(), "shards must stay in --output-dir"
        worker_files = sorted(worker_dir.glob("worker_*_features.tsv"))
        assert len(worker_files) >= 2
        for path in worker_files:
            assert _header_line(path) == header, path.name

        # Dtypes and the whole numeric summary are declared the same in both paths,
        # and no worker disagreed with the declaration.
        assert two["features"]["dtypes"] == one["features"]["dtypes"]
        assert two["features"]["n_features"] == one["features"]["n_features"]
        assert not two["integrity"].get("dtype_mismatches", 0)
    finally:
        fixture.close()


def test_merged_workers2_equals_workers1():
    """The strongest requirement: same rows, same values, whatever finished first."""
    fixture = _fixture()
    try:
        one_dir, two_dir = fixture.root / "out_w1", fixture.root / "out_w2"
        with _FakeCpuAllocation(2):
            code_one, one = fixture.run_into(one_dir, sample_fraction=1.0, workers=1)
            code_two, two = fixture.run_into(two_dir, sample_fraction=1.0, workers=2)
        assert code_one == 0 and code_two == 0, two["integrity"]

        lines_one = _data_lines(one_dir / "features.tsv")
        lines_two = _data_lines(two_dir / "features.tsv")
        assert len(lines_one) == len(lines_two) == two["sample"]["n_candidate_pairs_sampled"]
        # Same multiset of rows: nothing added, dropped, duplicated or altered.
        assert sorted(lines_two) == sorted(lines_one)
        # Same order too - the partition is contiguous in file order and the merge is
        # in worker order, so this holds without ever sorting. Comparing after a sort
        # (below) would pass even if the ordering rule were broken; this does not.
        assert lines_two == lines_one

        # And the numeric summary over the merged matrix is identical, sorted on the
        # deterministic key, which is what a trainer reads.
        first = _features_at(one_dir).sort_values(list(SORT_KEY), kind="stable")
        second = _features_at(two_dir).sort_values(list(SORT_KEY), kind="stable")
        assert first.reset_index(drop=True).equals(second.reset_index(drop=True))
        assert two["features"] == one["features"]
        assert two["integrity"] == one["integrity"]
    finally:
        fixture.close()


def test_empty_partitions_are_handled():
    """More workers than entities: the extra workers get nothing and must not fail."""
    counts = _counts(5)
    groups = epf.partition_entities(counts, 9)
    assert sum(len(group) for group in groups) == len(counts)
    assert sum(1 for group in groups if not group) >= 4

    fixture = _fixture()
    try:
        _, baseline = fixture.run(sample_fraction=1.0)
        n_entities = baseline["sample"]["n_s1_entities_sampled"]
        assert n_entities >= 2
        # A worker's share is a contiguous run of whole entities, so k entities can
        # fill at most k workers - four more workers than entities guarantees empties.
        workers = n_entities + 4
        out = fixture.root / "out_empty"
        with _FakeCpuAllocation(workers):
            code, report = fixture.run_into(out, sample_fraction=1.0, workers=workers)
        assert code == 0, report["integrity"]

        block = report["parallel"]
        assert len(block["n_s1_entities_per_worker"]) == workers
        assert sum(block["n_s1_entities_per_worker"]) == n_entities
        assert block["n_s1_entities_per_worker"].count(0) >= 4
        empty = [index for index, value in enumerate(block["n_s1_entities_per_worker"]) if not value]
        assert empty
        for index in empty:
            assert block["rows_per_worker"][index] == 0
            assert block["feature_seconds_per_worker"][index] == 0.0
            # No shard is written for an empty partition, and no feature file either -
            # the merge skips both rather than failing on a missing path.
            worker_dir = Path(block["shard_dir"])
            assert not (worker_dir / f"shard_{index:02d}.tsv").exists()
            assert not (worker_dir / f"worker_{index:02d}_features.tsv").exists()
            # The decision is still logged, so an empty worker is visible in the run.
            assert (worker_dir / f"{epf.LOG_NAME}_w{index:02d}.log").is_file()

        # The merged output is still exactly the single-process rows.
        assert _data_lines(out / "features.tsv") == _data_lines(fixture.out / "features.tsv")
        assert sum(block["rows_per_worker"]) == baseline["sample"]["n_candidate_pairs_sampled"]
        assert sum(block["shard_rows_per_worker"]) == baseline["sample"]["n_candidate_pairs_sampled"]
    finally:
        fixture.close()


def test_workers1_remains_the_single_process_path():
    """--workers 1 must be the original code path, not the parallel one with N=1."""
    fixture = _fixture()
    try:
        assert epf.DEFAULT_WORKERS == 1
        assert epf.parse_args(["--config", str(fixture.config_path)]).workers == 1
        code, report = fixture.run(sample_fraction=1.0)
        assert code == 0
        assert report["inputs"]["workers"] == 1
        # The parallel report block is absent, the phases that only the parallel path
        # has cost nothing, and nothing was partitioned onto disk.
        assert "parallel" not in report
        assert report["timing"]["shard_seconds"] == 0.0
        assert report["timing"]["merge_seconds"] == 0.0
        assert not (fixture.out / epf.WORKER_DIR_NAME).exists()
        assert (fixture.out / "features.tsv").is_file()
        # One process: the single-process figure is measured, not extrapolated, and
        # the 48-worker projection is still labelled unmeasured.
        assert report["extrapolation"]["assumed_workers"] == 1
        assert report["extrapolation"]["workers_are_measured"] is True
        assert report["extrapolation"]["single_process_equivalent"]["is_extrapolated"] is False
        assert report["extrapolation"]["theoretical_48_workers"]["is_theoretical"] is True
        assert "UNMEASURED" in report["extrapolation"]["theoretical_48_workers"]["note"]
    finally:
        fixture.close()


def test_invalid_worker_counts_are_rejected():
    """Rejected cleanly, against this process's allocation - never a clamp, never 48."""
    fixture = _fixture()
    try:
        with _FakeCpuAllocation(4):
            for bad in ("0", "-1", "5", "48", "many"):
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        epf.parse_args(["--config", str(fixture.config_path), "--workers", bad])
                except SystemExit as exc:
                    assert exc.code == 2, (bad, exc.code)
                else:
                    raise AssertionError(f"--workers {bad} must be rejected on a 4-CPU allocation")
            # The boundary is allowed, and the accepted count is the one used.
            assert epf.parse_args(["--config", str(fixture.config_path),
                                   "--workers", "4"]).workers == 4
            assert epf.available_cpu_count() == 4

            # The message names the allocation, so a rejected job says why.
            message = io.StringIO()
            try:
                with contextlib.redirect_stderr(message):
                    epf.parse_args(["--config", str(fixture.config_path), "--workers", "5"])
            except SystemExit:
                pass
            assert "5" in message.getvalue() and "4 CPU" in message.getvalue()
            assert "48" not in message.getvalue()

        # And the real limit is the allocation this process actually has.
        assert epf.available_cpu_count() >= 1
    finally:
        fixture.close()


def test_the_worker_limit_is_the_process_allocation():
    """On a scheduler-allocated node the limit is the affinity mask, not the machine."""
    if not hasattr(os, "sched_getaffinity"):  # pragma: no cover - Windows
        assert epf.available_cpu_count() == max(1, os.cpu_count() or 1)
        return
    assert epf.available_cpu_count() == max(1, len(os.sched_getaffinity(0)))


def test_partition_entities_is_contiguous_and_row_balanced():
    """The four properties the merge's determinism and the load balance rest on."""
    counts = _counts(37)
    workers = 7
    groups = epf.partition_entities(counts, workers)
    positions = {entity: index for index, entity in enumerate(counts)}
    flat = [entity for group in groups for entity in group]

    # Complete and disjoint, and each worker's entities are one contiguous run of the
    # file order, ascending - which is what makes the merge a plain concatenation.
    assert set(flat) == set(counts) and len(flat) == len(counts)
    blocks = []
    for group in groups:
        where = [positions[entity] for entity in group]
        assert where == sorted(where)
        blocks.append(where)
    occupied = sorted(index for where in blocks for index in where)
    assert occupied == list(range(len(counts)))

    # Balanced by rows, not by entity count: entities differ by an order of magnitude
    # in candidate count, so an even split of the entity list would not be an even
    # split of the work. A worker's window is one band of the row range, and it can
    # only overrun it by the entities that straddle its ends.
    total = sum(counts.values())
    largest = max(counts.values())
    rows_per_worker = [sum(counts[entity] for entity in group) for group in groups]
    assert max(rows_per_worker) <= total / workers + largest

    # Reproducible, and independent of dict order in any way that matters: the same
    # counts give the same partition, every time.
    assert epf.partition_entities(counts, workers) == groups
    assert epf.partition_entities(dict(counts), workers) == groups
    # Deterministic even when entities are far from uniform in size.
    lumpy = {**counts, "S1-heavy": 5000}
    assert epf.partition_entities(lumpy, 4) == epf.partition_entities(lumpy, 4)
    # Degenerate inputs: one entity, no entities, more workers than entities.
    assert epf.partition_entities({}, 4) == [[], [], [], []]
    assert epf.partition_entities({"S1-1": 0}, 3) == [["S1-1"], [], []]
    assert epf.partition_entities({"S1-1": 0}, 1) == [["S1-1"]]


def test_merge_features_copies_worker_files_in_order():
    """One header, worker order, empty and missing files skipped, bytes copied."""
    root = Path(tempfile.mkdtemp(prefix="step3_merge_test_"))
    try:
        def write(name: str, text: str) -> Path:
            path = root / name
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            return path

        header = "a\tb\n"
        payload = "1\t2\n3\t4\n"
        first = write("shard_00.tsv", header + "1\t2\n")
        empty = write("shard_01.tsv", "")
        second = write("shard_02.tsv", header + "3\t4\n")
        missing = root / "shard_03.tsv"
        target = root / "features.tsv"

        size = epf.merge_features([first, empty, second, missing], target)
        with open(target, encoding="utf-8", newline="") as handle:
            assert handle.read() == header + payload
        assert size == target.stat().st_size == len(header) + len(payload)
        # All-empty input is not an error either: it is an empty file, byte for byte.
        nothing = root / "empty.tsv"
        assert epf.merge_features([empty, missing], nothing) == 0
        with open(nothing, encoding="utf-8", newline="") as handle:
            assert handle.read() == ""
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_filtered_lookup_keeps_every_join_the_unfiltered_one_makes():
    """The per-worker filter is invisible in the values, which is why it is safe.

    The filter is only ever asked about ids the asking shard can join to:
    ``_collect_needed_ids`` builds ``keep_ids`` from the shard's own rows, so a
    worker's ``take`` never names an id outside it. That domain is what this test
    exercises - an id deliberately dropped by ``keep_ids`` is expected to report
    "not found", and is checked separately below.
    """
    fixture = _fixture()
    try:
        full = epf.load_lookup(fixture.config, "train", "source2", fixture.log)
        filtered = epf.load_lookup(fixture.config, "train", "source2", fixture.log,
                                   keep_ids={"S2-1", "S2-3"})
        ids = np.array(["S2-1", "S2-3", "S9-7", ""], dtype=object)
        positions_full, found_full = full.take(ids)
        positions_filtered, found_filtered = filtered.take(ids)
        # Same join verdict for every id: kept, and found exactly when it was before.
        assert list(found_full) == list(found_filtered)
        for column in (epf.PREPARED_NAME_NORM, epf.PREPARED_NAME_KEY,
                       epf.PREPARED_ADDRESS_NORM, epf.PREPARED_COUNTRY_NORM):
            assert list(full.values(column, positions_full, found_full)) == list(
                filtered.values(column, positions_filtered, found_filtered)
            ), column
        assert full.n_entities == 3 and filtered.n_entities == 2
        # The filter only ever *removes* rows: the id it dropped is absent rather than
        # altered, which is why a worker can never ask about one.
        probed = np.array(["S2-2"], dtype=object)
        assert bool(full.take(probed)[1][0])
        assert not bool(filtered.take(probed)[1][0])
        # The whole point: fewer rows resident, so W workers do not hold W copies.
        assert filtered.memory_bytes() < full.memory_bytes()
        # A shard with no pairs for a source loads that source as an empty lookup, and
        # an empty lookup joins nothing rather than raising.
        nothing = epf.load_lookup(fixture.config, "train", "source2", fixture.log, keep_ids=set())
        assert nothing.n_entities == 0
        assert not nothing.take(ids)[1].any()
        assert nothing.memory_bytes() >= 0
    finally:
        fixture.close()


def test_parallel_report_block_describes_the_run():
    """The benchmark is read off this block, so it must be complete and consistent."""
    fixture = _fixture()
    try:
        out = fixture.root / "out_w2"
        with _FakeCpuAllocation(2):
            code, report = fixture.run_into(out, sample_fraction=1.0, workers=2)
        assert code == 0, report["integrity"]
        block = report["parallel"]

        assert block["workers"] == 2 == report["inputs"]["workers"]
        assert "worker index" in block["merge_order"] and "completion" in block["merge_order"]
        for key in ("n_s1_entities_per_worker", "rows_per_worker", "shard_rows_per_worker",
                    "feature_seconds_per_worker", "lookup_bytes_per_worker"):
            assert len(block[key]) == 2, key
        assert sum(block["rows_per_worker"]) == report["sample"]["n_candidate_pairs_sampled"]
        assert sum(block["shard_rows_per_worker"]) == report["sample"]["n_candidate_pairs_sampled"]
        assert sum(block["n_s1_entities_per_worker"]) == report["sample"]["n_s1_entities_sampled"]
        assert block["total_lookup_bytes"] == sum(block["lookup_bytes_per_worker"])
        assert block["max_worker_lookup_bytes"] == max(block["lookup_bytes_per_worker"])
        assert block["unassigned_rows"] == 0
        assert block["s1_counts_entries"] == report["sample"]["n_s1_entities_sampled"]
        assert 0.0 <= block["worker_utilization"] <= 1.0
        assert block["total_peak_rss"] is None or block["total_peak_rss"]

        # The three parallel phases add up to the phase total the report projects from.
        assert abs(block["shard_seconds"] + block["worker_wall_seconds"]
                   + block["merge_seconds"] - block["parallel_phase_seconds"]) <= 0.05
        assert report["timing"]["parallel_phase_seconds"] == block["parallel_phase_seconds"]
        assert report["timing"]["shard_seconds"] == block["shard_seconds"]
        assert report["timing"]["merge_seconds"] == block["merge_seconds"]
        assert report["memory"]["prepared_lookup_estimate"] == block["total_lookup_size"]

        # The projection is labelled with the worker count it was measured at, and the
        # 48-worker number is quarantined as an assumption rather than reported as one.
        assert report["extrapolation"]["assumed_workers"] == 2
        assert report["extrapolation"]["workers_are_measured"] is True
        assert report["extrapolation"]["single_process_equivalent"]["is_extrapolated"] is True
        assert report["extrapolation"]["parallel_phase_seconds_measured"] == (
            block["parallel_phase_seconds"]
        )
    finally:
        fixture.close()


def test_cleanup_shards_removes_only_the_worker_directory():
    """Shards are kept by default (they make a disagreement traceable), removable on request."""
    fixture = _fixture()
    try:
        out = fixture.root / "out_clean"
        with _FakeCpuAllocation(2):
            code, report = fixture.run_into(out, sample_fraction=1.0, workers=2,
                                            cleanup_shards=True)
        assert code == 0, report["integrity"]
        assert report["parallel"]["shard_dir"]
        assert not (out / epf.WORKER_DIR_NAME).exists()
        for name in ("features.tsv", "sample_candidates.tsv", "feature_missingness.csv",
                     "step3_features_report.json", "extract_pair_features.log"):
            assert (out / name).is_file(), name
        # The report still says where the shards were and what they contained.
        assert sum(report["parallel"]["shard_rows_per_worker"]) == report["sample"][
            "n_candidate_pairs_sampled"
        ]
    finally:
        fixture.close()


# ---------------------------------------------------------------------------
# standalone runner (no pytest required)
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        else:
            print(f"[PASS] {test.__name__}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} tests failed")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
