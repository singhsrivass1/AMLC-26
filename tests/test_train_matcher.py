"""V1 matcher tests: pair-level labels, entity-grouped folds, OOF sweep, artifacts.

What is being guarded, and why each guard exists
------------------------------------------------
The matcher is the first stage in this pipeline whose output is a *prediction*
rather than a measurement, so its failure modes are silent: a leaky fold split, a
label shifted by one row against its features, or a threshold chosen from a metric
that is not the challenge's all produce a number that looks fine and is wrong. Each
test below pins one of those.

The fixture is small enough to verify by hand, and it is built to contain the cases
that actually break label generation:

* **one-to-many**: ``S1-5`` has three true matches (two S2, one S3) and two
  candidates that are not matches, so its five rows must label as
  ``[1, 1, 0, 1, 0]`` - not "entity is right, all 1" and not "one was wrong, all 0".
* **a missed match must not poison its entity**: ``S1-3`` truly matches ``S2-5`` and
  ``S2-6``, but the blocker only proposed ``S2-5``. Its single row must label 1.
* **an empty ground truth**: ``S1-16`` has no true match at all and two candidates, so
  both rows label 0, and under ``score_zero`` the entity scores 1 only if the
  matcher predicts nothing for it.
* **the same target for two different S1s**: ``S2-1`` is a true match for both
  ``S1-5`` and ``S1-10``, so labels must key on the pair, not on the target alone.
* **blank evidence**: ``token_df`` is blank on the pairs the token blocker did not
  propose, which must arrive as NaN and never as 0.
* **a chunk boundary through an entity**: the fixture is read with
  ``chunksize=4`` over 10 rows, so ``S1-5``'s five rows straddle two chunks.

The end-to-end test asserts an F0.5 that can be computed by hand from this fixture
(0.958333) *and* that the repository's own
``CandidateEvaluation.evaluate_file`` returns the same number when asked to score
the predicted pairs from a file - so the matcher cannot drift from the graded metric.

Runs standalone (``python tests/test_train_matcher.py``) and under pytest.
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import GroundTruth, load_ground_truth  # noqa: E402
from src.evaluation import CandidateEvaluation  # noqa: E402
from src.matching_model import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_DTYPES,
    ID_COLUMNS,
    MAX_FOLDS,
    NON_FEATURE_COLUMNS,
    ModelBundle,
    aggregate_matches,
    assert_fold_purity,
    assign_folds,
    build_label_artifacts,
    decide,
    default_params,
    entity_metrics,
    evaluate_at_threshold,
    load_bundle,
    log_grid_positions,
    macro_f05,
    predict,
    resolve_feature_columns,
    resolve_fold_mode,
    sweep_thresholds,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMP_DIRS: list[Path] = []

# The fixture's entity ids are not arbitrary. The train/val split is a pure hash of the
# S1 id (``assign_splits``) and ``val_fraction`` must be strictly below 1, so these four
# ids were chosen because they all hash into the val bucket at ``VAL_SPLIT_FRACTION``.
# ``test_fixture_entities_are_in_the_val_split`` pins that, so if the hash or the
# fraction ever changes the failure says why, instead of the metric silently being
# computed over a subset of the fixture's entities.
VAL_SPLIT_FRACTION = 0.2

# ``assign_folds`` refuses to leave a fold empty, and with four entities a hash
# assignment can only fill two folds (it gives [2, 2]; three would give [0, 1, 3]). The
# real run uses DEFAULT_FOLDS = 5, where 2.2M entities make an empty fold impossible -
# so this is a fixture-scale constraint, not a property of the matcher.
FIXTURE_FOLDS = 2


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
GROUND_TRUTH_ROWS = [
    ("S1-5", "S2-1,S2-2,S3-1"),
    ("S1-10", "S2-1"),
    ("S1-16", ""),
    ("S1-3", "S2-5,S2-6"),
]

# (s1, target, source, name_token_set_ratio, expected_label, token_df)
FEATURE_ROWS = [
    ("S1-5", "S2-1", "S2", 0.95, 1, 42.0),  # true, and carries a real token_df
    ("S1-5", "S2-2", "S2", 0.95, 1, None),
    ("S1-5", "S2-3", "S2", 0.30, 0, None),
    ("S1-5", "S3-1", "S3", 0.95, 1, None),
    ("S1-5", "S3-2", "S3", 0.30, 0, None),
    ("S1-10", "S2-1", "S2", 0.95, 1, None),  # same target as S1-5's, different pair
    ("S1-10", "S2-9", "S2", 0.30, 0, None),
    ("S1-16", "S2-4", "S2", 0.30, 0, None),  # empty ground truth
    ("S1-16", "S3-4", "S3", 0.30, 0, None),
    ("S1-3", "S2-5", "S2", 0.95, 1, None),  # S2-6 was never proposed; must still be 1
]

EXPECTED_LABELS = np.array([row[4] for row in FEATURE_ROWS], dtype=np.int8)
# Ground-truth row order is GROUND_TRUTH_ROWS, so owners are positions into it.
EXPECTED_OWNERS = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3], dtype=np.int32)
# True matches per ground-truth entity: S1-5 has 3, S1-10 has 1, S1-16 has 0, S1-3 has 2.
EXPECTED_LENGTHS = np.array([3, 1, 0, 2], dtype=np.int64)
EXPECTED_SOURCE_IS_S2 = np.array([row[2] == "S2" for row in FEATURE_ROWS], dtype=np.int8)


def _write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            cells = ["" if value is None else str(value) for value in row]
            handle.write("\t".join(cells) + "\n")


def _temp_dir() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="matcher_test_"))
    _TEMP_DIRS.append(directory)
    return directory


def _config() -> dict:
    """The fixture's config: the default hash split, and the challenge's own policy.

    The split is left at the repository default shape (``hash`` mode, ``val_fraction``
    below 1) rather than being bent to make the fixture convenient: the fixture's ids
    were picked to fall in val under exactly this config, which is the same code path a
    real run takes.
    """
    return {
        "project": {"seed": 42},
        "columns": {"gt_source1_id": "source1_entity_id", "gt_matched_ids": "matched_entity_ids"},
        "io": {"chunksize": 4},
        "evaluation": {
            "split": {"enabled": True, "val_fraction": VAL_SPLIT_FRACTION, "mode": "hash"},
            "zero_match_policy": "score_zero",
        },
    }


def _make_fixture(root: Path, skip_s1: tuple[str, ...] = ()) -> dict:
    """Write the ground truth and the feature file, with the chunk boundary inside S1-5.

    ``skip_s1`` drops those entities' *candidate rows* only - the ground truth keeps
    them. That is the real shape of a run: the ground truth is every S1 entity in the
    competition, while a feature file is the candidates the blockers proposed, so some
    entities have no rows at all. It is also what made the purity check fail on HPC
    (see ``test_fold_purity_ignores_ground_truth_entities_without_candidate_rows``).
    """
    ground_truth_path = root / "ground_truth.tsv"
    _write_tsv(
        ground_truth_path,
        ["source1_entity_id", "matched_entity_ids"],
        [[s1, matches] for s1, matches in GROUND_TRUTH_ROWS],
    )

    features_path = root / "features.tsv"
    header = list(ID_COLUMNS) + list(FEATURE_COLUMNS) + list(NON_FEATURE_COLUMNS)
    rows = []
    for s1, target, source, ratio, _label, token_df in FEATURE_ROWS:
        if s1 in skip_s1:
            continue
        values = {name: 0 for name in FEATURE_COLUMNS}
        values["name_token_set_ratio"] = ratio
        values["source_is_s2"] = 1 if source == "S2" else 0
        # Blank stays blank: this is the "the blocker that measures this did not
        # propose this pair" case, which must reach the model as NaN.
        values["token_df"] = token_df if token_df is not None else None
        rows.append([s1, target, source] + [values[name] for name in FEATURE_COLUMNS] + [1])
    _write_tsv(features_path, header, rows)

    config = _config()
    ground_truth = load_ground_truth(config, path=ground_truth_path)
    return {
        "root": root,
        "config": config,
        "ground_truth": ground_truth,
        "ground_truth_path": ground_truth_path,
        "features_path": features_path,
    }


def _load_artifacts(fixture: dict) -> dict:
    """Run the labeler against the fixture and load what it wrote."""
    out_dir = _temp_dir() / "labels"
    report = build_label_artifacts(
        fixture["features_path"], fixture["ground_truth"], out_dir, chunksize=4
    )
    return {
        "out_dir": out_dir,
        "report": report,
        "labels": np.load(out_dir / "val_labels.npy"),
        "owners": np.load(out_dir / "val_owner_index.npy"),
        "sources": np.load(out_dir / "val_source_is_s2.npy"),
        "matrix": np.load(out_dir / "val_features.npy"),
    }


def _has_lightgbm() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def _sklearn_available() -> bool:
    return importlib.util.find_spec("sklearn") is not None


def test_fixture_entities_are_in_the_val_split():
    """Guards the fixture's load-bearing assumption, not the split itself.

    The end-to-end tests compare against an F0.5 computed by hand over all four
    entities. That only holds while all four are val, and the split is a hash of the
    id - so if this ever fails, the fixture ids need re-picking rather than the
    expected number being adjusted.
    """
    from src.evaluation import split_mask_for

    fixture = _make_fixture(_temp_dir())
    mask = split_mask_for(fixture["ground_truth"], fixture["config"], "val")
    ids = [s1 for s1, _ in GROUND_TRUTH_ROWS]
    assert mask.all(), (
        f"fixture ids {ids} must all be in the val split at val_fraction="
        f"{VAL_SPLIT_FRACTION}; they are not, so every end-to-end macro F0.5 below would "
        "be computed over a subset of the fixture"
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_feature_columns_match_the_extractor_exactly():
    """The 29 features must match the extractor's declaration in name AND order.

    The extractor is parsed, not imported, so this stays a pure schema check. A
    reordering here would silently train every model on a permuted matrix while
    the metrics still looked plausible.
    """
    source = (REPO_ROOT / "scripts" / "extract_pair_features.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Two passes: the extractor builds some constants out of other module-level string
    # constants (``PREPARED_ID_COLUMN`` and friends), so those have to be collected
    # before anything that references them can be resolved. Only the names actually
    # needed are resolved, so an unrelated module-level expression cannot break this.
    constants: dict[str, str] = {}
    declarations: dict[str, ast.AST] = {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        for target in targets:
            if node.value is None:
                continue
            declarations[target.id] = node.value
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                constants[target.id] = node.value.value

    def resolve(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in constants:
                raise AssertionError(
                    f"the extractor's declaration references {node.id!r}, which this test "
                    "cannot resolve; teach it that constant rather than skipping the check"
                )
            return constants[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(resolve(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            return {resolve(k): resolve(v) for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)

    for required in ("FEATURE_DTYPES", "NON_FEATURE_COLUMNS"):
        assert required in declarations, f"the extractor no longer declares {required}"

    declared = resolve(declarations["FEATURE_DTYPES"])
    non_features = set(resolve(declarations["NON_FEATURE_COLUMNS"]))
    expected = [name for name in declared if name not in non_features]

    assert list(FEATURE_COLUMNS) == expected, (
        "FEATURE_COLUMNS drifted from the extractor:\n"
        f"  extractor: {expected}\n  matcher:   {list(FEATURE_COLUMNS)}"
    )
    assert len(FEATURE_COLUMNS) == 29, f"expected 29 features, got {len(FEATURE_COLUMNS)}"
    for name in FEATURE_COLUMNS:
        assert FEATURE_DTYPES[name] == declared[name], (
            f"dtype for {name!r}: matcher says {FEATURE_DTYPES[name]!r}, "
            f"extractor says {declared[name]!r}"
        )
    assert non_features == {"text_join_ok"}
    assert "text_join_ok" not in FEATURE_COLUMNS


def test_resolve_feature_columns_rejects_drift():
    header = list(ID_COLUMNS) + list(FEATURE_COLUMNS) + ["text_join_ok"]
    assert resolve_feature_columns(header) == list(FEATURE_COLUMNS)

    try:
        resolve_feature_columns([c for c in header if c != "token_df"])
    except ValueError as exc:
        assert "token_df" in str(exc), f"the missing column should be named: {exc}"
    else:
        raise AssertionError("a feature file missing a feature must not be accepted")

    try:
        resolve_feature_columns(header + ["leaked_label"])
    except ValueError as exc:
        assert "leaked_label" in str(exc), f"the unknown column should be named: {exc}"
    else:
        raise AssertionError("an unknown column must not be silently accepted")


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def test_labels_are_pair_level_and_exactly_as_hand_computed():
    """The five-row one-to-many case, the missed match, and the empty entity."""
    fixture = _make_fixture(_temp_dir())
    artifacts = _load_artifacts(fixture)

    assert np.array_equal(artifacts["labels"], EXPECTED_LABELS), (
        f"labels: got {artifacts['labels'].tolist()}, want {EXPECTED_LABELS.tolist()}"
    )

    # S1-5: two S2 matches, one S3 match, two non-matches.
    assert artifacts["labels"][:5].tolist() == [1, 1, 0, 1, 0]
    # S1-3's second true match was never proposed - its one row must still be positive.
    assert artifacts["labels"][9] == 1, "a missed match must not negate the pairs that were found"
    # S1-16 has no true match at all.
    assert artifacts["labels"][7:9].tolist() == [0, 0]
    # S2-1 is a true match for both S1-5 and S1-10.
    assert artifacts["labels"][0] == 1 and artifacts["labels"][5] == 1


def test_labels_owners_and_sources_align_row_for_row():
    fixture = _make_fixture(_temp_dir())
    artifacts = _load_artifacts(fixture)
    report = artifacts["report"]

    assert len(artifacts["labels"]) == len(FEATURE_ROWS)
    assert len(artifacts["owners"]) == len(FEATURE_ROWS)
    assert len(artifacts["matrix"]) == len(FEATURE_ROWS)
    assert np.array_equal(artifacts["owners"], EXPECTED_OWNERS), (
        f"owners: got {artifacts['owners'].tolist()}, want {EXPECTED_OWNERS.tolist()}"
    )
    assert np.array_equal(artifacts["sources"], EXPECTED_SOURCE_IS_S2)
    assert report["rows_labelled"] == len(FEATURE_ROWS)
    assert report["rows_dropped_unknown_s1"] == 0
    assert report["positive_labels"] == int(EXPECTED_LABELS.sum()) == 5
    assert report["negative_labels"] == len(FEATURE_ROWS) - 5 == 5
    assert abs(report["positive_rate"] - 0.5) < 1e-12
    assert report["n_s1_entities_with_rows"] == 4
    # Rows belonging to the one entity with no true match (S1-16's two rows).
    assert report["rows_for_entities_with_no_true_match"] == 2
    assert report["n_features"] == 29


def test_blank_evidence_reaches_the_matrix_as_nan_not_zero():
    fixture = _make_fixture(_temp_dir())
    artifacts = _load_artifacts(fixture)
    matrix = artifacts["matrix"]
    column = list(FEATURE_COLUMNS).index("token_df")

    assert matrix.dtype == np.float32
    assert matrix.shape == (len(FEATURE_ROWS), 29)
    assert matrix[0, column] == np.float32(42.0), "a present token_df must survive parsing"
    for row in (1, 2, 3):
        assert np.isnan(matrix[row, column]), (
            f"row {row}: blank token_df became {matrix[row, column]!r}; blank means "
            "'this blocker did not propose the pair', which is not zero evidence"
        )
    # A blank in one column must not blank the rest of the row.
    assert matrix[1, list(FEATURE_COLUMNS).index("name_token_set_ratio")] == np.float32(0.95)


def test_rows_with_an_unknown_s1_id_are_dropped_counted_and_kept_aligned(tmp_path=None):
    """The anomalous path: a dropped row must not shift every row after it.

    The evaluator drops unknown S1 ids and counts them, and so does the labeler - but
    the artifacts have to stay row-aligned with each other while doing it, so this
    checks the compaction rather than just the count.
    """
    root = _temp_dir()
    fixture = _make_fixture(root)

    features_path = root / "features_with_unknown.tsv"
    original = fixture["features_path"].read_text(encoding="utf-8").splitlines()
    # Inject an unknown entity in the middle, where a naive drop would misalign.
    parts = original[4].split("\t")
    parts[0] = "S1-999"
    original.insert(4, "\t".join(parts))
    features_path.write_text("\n".join(original) + "\n", encoding="utf-8")

    out_dir = root / "labels_unknown"
    report = build_label_artifacts(features_path, fixture["ground_truth"], out_dir, chunksize=4)

    labels = np.load(out_dir / "val_labels.npy")
    owners = np.load(out_dir / "val_owner_index.npy")
    matrix = np.load(out_dir / "val_features.npy")

    assert report["rows_dropped_unknown_s1"] == 1
    assert report["rows_dropped_unknown_s1_examples"] == ["S1-999"]
    assert report["rows_labelled"] == len(FEATURE_ROWS)
    assert len(labels) == len(owners) == len(FEATURE_ROWS)
    assert matrix.shape == (len(FEATURE_ROWS), 29)
    # The labels after the injected row are the same as without it - no off-by-one.
    assert np.array_equal(labels, EXPECTED_LABELS)
    assert np.array_equal(owners, EXPECTED_OWNERS)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def test_no_s1_entity_is_split_across_train_and_holdout():
    """The leakage guarantee: an entity's rows all share one fold."""
    fixture = _make_fixture(_temp_dir())
    ground_truth = fixture["ground_truth"]
    artifacts = _load_artifacts(fixture)
    owners = artifacts["owners"].astype(np.int64)

    entity_folds = assign_folds(ground_truth, n_folds=FIXTURE_FOLDS, mode="hash")
    row_folds = entity_folds[owners]
    assert_fold_purity(row_folds, owners, FIXTURE_FOLDS)  # must not raise

    for owner in np.unique(owners):
        folds = np.unique(row_folds[owners == owner])
        assert len(folds) == 1, f"entity at gt position {owner} spans folds {folds.tolist()}"

    # The same statement in the form that matters: no training row may come from an
    # entity whose rows are being predicted by that fold.
    for fold in range(FIXTURE_FOLDS):
        held_out = row_folds == fold
        trained_on = ~held_out
        assert not np.intersect1d(owners[held_out], owners[trained_on]).size, (
            f"fold {fold} trained on entities it is being scored on"
        )


def test_fold_assignment_is_deterministic_and_covers_every_entity():
    fixture = _make_fixture(_temp_dir())
    ground_truth = fixture["ground_truth"]

    first = assign_folds(ground_truth, n_folds=FIXTURE_FOLDS, mode="hash")
    second = assign_folds(ground_truth, n_folds=FIXTURE_FOLDS, mode="hash")
    assert np.array_equal(first, second), "fold assignment must be reproducible"

    assert len(first) == ground_truth.n_entities
    assert first.min() >= 0 and first.max() < FIXTURE_FOLDS
    counts = np.bincount(first.astype(np.int64), minlength=FIXTURE_FOLDS)
    assert (counts > 0).all(), f"every fold must own at least one entity, got {counts.tolist()}"

    # The same entity id must always land in the same fold, run to run and mode to mode
    # for the hash mode (which is a pure function of the id).
    assert first[0] == second[0]


def test_groupkfold_mode_is_entity_pure_and_covers_entities_without_matches():
    """The literal ``GroupKFold(n_splits=5)`` path, and the gap it has to be patched for.

    ``GroupKFold`` is given one pseudo-row per *true match*, so it balances folds by
    sample count - but an entity with an empty ground truth contributes no pseudo-row
    and therefore gets no fold at all, while still owning candidate rows that must be
    predicted out of fold. The matcher fills those in by hash; this checks that it does,
    because the failure mode is silent: the entity would sit in fold 0 next to nothing
    that says it was never assigned.
    """
    if not _sklearn_available():
        assert resolve_fold_mode("auto") == "hash", "auto must fall back to hash"
        try:
            assign_folds(_make_fixture(_temp_dir())["ground_truth"], n_folds=2, mode="groupkfold")
        except ImportError as exc:
            assert "pip install scikit-learn" in str(exc), f"no install hint in: {exc}"
        else:
            raise AssertionError("explicit groupkfold must not silently fall back")
        return

    assert resolve_fold_mode("auto") == "groupkfold", "auto must prefer GroupKFold"

    fixture = _make_fixture(_temp_dir())
    ground_truth = fixture["ground_truth"]
    report: dict = {}
    entity_folds = assign_folds(
        ground_truth, n_folds=FIXTURE_FOLDS, mode="auto", report=report
    )
    assert report["mode"] == "groupkfold" and report["requested_mode"] == "auto"
    assert len(entity_folds) == ground_truth.n_entities
    assert (entity_folds >= 0).all(), "every entity needs a fold, including empty-GT ones"
    assert not (np.asarray(report["entities_per_fold"]) == 0).any()

    # Position 2 is the fixture's empty-ground-truth entity: it owns candidate rows but
    # no true match, so it is the one GroupKFold itself cannot place.
    empty_gt_entity = int(np.flatnonzero(ground_truth.lengths() == 0)[0])
    assert empty_gt_entity == 2
    assert entity_folds[empty_gt_entity] >= 0

    owners = _load_artifacts(fixture)["owners"].astype(np.int64)
    assert_fold_purity(entity_folds[owners], owners, FIXTURE_FOLDS)


def test_fold_purity_assertion_catches_a_row_level_split():
    """The guard has to fire on a row-level split, or it is not a guard.

    A row-level fold assignment is the exact bug that would inflate the score, so the
    assertion is given one deliberately. The count must be exactly one entity: entity 1
    is pure and must not be swept up with it, or a real leak's size becomes unreadable.
    """
    owners = np.array([0, 0, 0, 1], dtype=np.int64)
    split_by_row = np.array([0, 1, 0, 0], dtype=np.int8)  # entity 0 spans folds 0 and 1
    try:
        assert_fold_purity(split_by_row, owners, 2)
    except ValueError as exc:
        message = str(exc)
        assert "more than one fold" in message, f"unexpected message: {exc}"
        assert message.startswith("1 S1 entities"), f"exactly one entity is impure: {message}"
        assert "2 entities own candidate rows in this run" in message, (
            f"the message must state what it actually examined: {message}"
        )
    else:
        raise AssertionError("a row-level split must be rejected")

    assert_fold_purity(np.array([1, 1, 1, 0], dtype=np.int8), owners, 2)  # pure is fine


def test_fold_purity_ignores_ground_truth_entities_without_candidate_rows():
    """The HPC shakedown failure, in miniature - the root cause of it.

    A feature file holds one split's candidates; the ground truth holds every S1 entity
    in the competition. So most ground-truth entities own no candidate rows, and one of
    them sitting *inside* the range the check walks leaves a hole in it. The body used
    to start its min/max accumulators at ``(n_folds, -1)``, so every hole read as "this
    entity spans two folds" and a perfectly grouped assignment was rejected, reporting

        (highest owner position + 1) - (entities owning rows)

    which grows with the ground truth, not with any leak. On the HPC run that was
    2,194,111 "impure" entities. Here S1-10 (position 1) is the entity the feature file
    has no rows for, and it is the hole the old body counted.
    """
    fixture = _make_fixture(_temp_dir(), skip_s1=("S1-10",))
    ground_truth = fixture["ground_truth"]
    owners = _load_artifacts(fixture)["owners"].astype(np.int64)
    entity_folds = assign_folds(ground_truth, n_folds=FIXTURE_FOLDS, mode="hash")
    row_folds = entity_folds[owners]

    # Preconditions for the regression. The ground truth must keep the entity the
    # feature file skips, and that entity's position must be *inside* the checked range:
    # put it past the end and the old code missed it too, so the test would pass for the
    # wrong reason.
    assert ground_truth.n_entities == len(GROUND_TRUTH_ROWS), "the ground truth keeps all four"
    covered = set(np.unique(owners).tolist())
    assert 1 not in covered, "S1-10 must own no candidate rows"
    assert max(covered) > 1, "the hole has to be interior, not past the end of the range"

    # What the old body reported, reconstructed: exactly the one hole, nothing else.
    old_low = np.full(int(owners.max()) + 1, FIXTURE_FOLDS, dtype=np.int8)
    old_high = np.full(int(owners.max()) + 1, -1, dtype=np.int8)
    np.minimum.at(old_low, owners, row_folds)
    np.maximum.at(old_high, owners, row_folds)
    assert int(np.count_nonzero(old_low != old_high)) == 1, (
        "the false positive is the entity with no rows, and it is the only one"
    )

    counts = assert_fold_purity(row_folds, owners, FIXTURE_FOLDS)  # must not raise
    assert counts["rows"] == len(owners)
    assert counts["entity_index_span"] == int(owners.max()) + 1
    assert counts["entities_checked"] == 3, "three of the four entities own rows"

    # The grouping itself is unchanged and still holds per entity.
    for position in np.unique(owners):
        assert len(np.unique(row_folds[owners == position])) == 1


def test_fold_purity_rejects_misaligned_rows_unknown_owners_and_bad_fold_ids():
    """The three inputs that would make the reductions above lie, rather than raise.

    A negative owner wraps onto a real entity, and a fold id outside ``[0, n_folds)``
    collides with the ``n_folds`` sentinel - either would turn a leak into a pass, so
    both are rejected instead of being reduced over.
    """
    owners = np.array([0, 0, 1], dtype=np.int64)
    folds = np.array([0, 0, 1], dtype=np.int8)

    try:
        assert_fold_purity(folds[:2], owners, 2)
    except ValueError as exc:
        assert "row-aligned" in str(exc), f"unexpected message: {exc}"
    else:
        raise AssertionError("misaligned inputs must be rejected")

    unknown = np.array([0, -1, 1], dtype=np.int64)
    try:
        assert_fold_purity(folds, unknown, 2)
    except ValueError as exc:
        assert "negative owner index" in str(exc), f"unexpected message: {exc}"
    else:
        raise AssertionError("an unknown S1 id must be rejected, not wrapped onto entity -1")

    for bad in (np.array([0, 0, 2], dtype=np.int8), np.array([0, 0, -1], dtype=np.int8)):
        try:
            assert_fold_purity(bad, owners, 2)
        except ValueError as exc:
            assert "fold ids must be in" in str(exc), f"unexpected message: {exc}"
        else:
            raise AssertionError(f"fold ids {bad.tolist()} must be rejected")


def test_one_to_many_entities_keep_every_row_in_one_fold():
    """One S1 with several candidates, checked per entity *and* per fold.

    An entity that legitimately matches three targets has three candidate rows, and a
    partial-fold bug is easy to miss: each row still reads as "some fold", and only the
    reverse statement - no fold holds a strict subset of an entity's rows - shows it.
    Both fold modes are checked, including ``auto``, because ``auto`` is what a real run
    takes.
    """
    root = _temp_dir()
    ground_truth_rows = [
        ("S1-1", "S2-1,S2-2,S2-3"),
        ("S1-2", "S2-4,S2-5"),
        ("S1-3", "S2-6"),
        ("S1-4", "S2-7,S2-8"),
        ("S1-5", "S2-9,S2-10"),
    ]
    # (s1, target) - several candidates per entity, true and false mixed.
    candidate_rows = [
        ("S1-1", "S2-1", 1),
        ("S1-1", "S2-2", 1),
        ("S1-1", "S2-3", 1),
        ("S1-1", "S2-90", 0),
        ("S1-2", "S2-4", 1),
        ("S1-2", "S2-5", 1),
        ("S1-3", "S2-6", 1),
        ("S1-3", "S2-91", 0),
        ("S1-4", "S2-7", 1),
        ("S1-4", "S2-8", 1),
        ("S1-4", "S2-92", 0),
        ("S1-5", "S2-9", 1),
        ("S1-5", "S2-10", 1),
    ]
    ground_truth_path = root / "ground_truth.tsv"
    features_path = root / "features.tsv"
    _write_tsv(
        ground_truth_path,
        ["source1_entity_id", "matched_entity_ids"],
        [[s1, matches] for s1, matches in ground_truth_rows],
    )
    header = list(ID_COLUMNS) + list(FEATURE_COLUMNS) + list(NON_FEATURE_COLUMNS)
    rows = []
    for s1, target, ratio in candidate_rows:
        values = {name: 0 for name in FEATURE_COLUMNS}
        values["name_token_set_ratio"] = float(ratio)
        values["source_is_s2"] = 1
        rows.append([s1, target, "S2"] + [values[name] for name in FEATURE_COLUMNS] + [1])
    _write_tsv(features_path, header, rows)

    ground_truth = load_ground_truth(_config(), path=ground_truth_path)
    artifacts = build_label_artifacts(features_path, ground_truth, _temp_dir() / "labels", chunksize=5)
    owners = np.load(Path(artifacts["owner_index_path"])).astype(np.int64)
    labels = np.load(Path(artifacts["labels_path"]))

    # The fixture really is one-to-many, in the rows and in the labels.
    rows_per_entity = [int((owners == position).sum()) for position in range(ground_truth.n_entities)]
    assert rows_per_entity == [4, 2, 2, 3, 2], f"unexpected rows per entity: {rows_per_entity}"
    assert int(labels.sum()) == 10, "ten of the thirteen candidates are true pairs"

    modes = ["hash", "auto"] if _sklearn_available() else ["hash"]
    for mode in modes:
        entity_folds = assign_folds(ground_truth, n_folds=FIXTURE_FOLDS, mode=mode)
        row_folds = entity_folds[owners]
        assert_fold_purity(row_folds, owners, FIXTURE_FOLDS)  # must not raise

        for position in range(ground_truth.n_entities):
            rows_of_entity = owners == position
            folds_of_entity = np.unique(row_folds[rows_of_entity])
            assert len(folds_of_entity) == 1, (
                f"mode={mode}: entity at position {position} spans folds "
                f"{folds_of_entity.tolist()} across its {int(rows_of_entity.sum())} rows"
            )
            # Reverse statement: for every fold, this entity is either entirely in it or
            # entirely out of it - never partly.
            for fold in range(FIXTURE_FOLDS):
                in_fold = row_folds[rows_of_entity] == fold
                assert in_fold.all() or not in_fold.any(), (
                    f"mode={mode}: fold {fold} holds {int(in_fold.sum())} of entity "
                    f"{position}'s {int(rows_of_entity.sum())} rows"
                )

        # And the form that matters for training: the entities a fold is scored on are
        # absent from everything it trains on.
        for fold in range(FIXTURE_FOLDS):
            held_out = row_folds == fold
            assert not np.intersect1d(owners[held_out], owners[~held_out]).size, (
                f"mode={mode}: fold {fold} trained on entities it is being scored on"
            )


def test_fold_ids_survive_more_folds_than_int8_can_hold():
    """``--folds`` past 127 used to wrap the fold id into a negative one.

    ``assign_folds`` reads a negative id as "this entity was never assigned" and re-places
    it by hash, which is a silent entity split - the same class of bug as the HPC one,
    reachable from a CLI flag. Fold ids are ``int16`` and the count is bounded explicitly.

    The entity count is what makes 130 folds fillable by hash at all: ``assign_folds``
    refuses to leave a fold empty, and ~15 entities per fold is the average here.
    """
    n_entities = 2_000
    lengths = np.ones(n_entities, dtype=np.int64)
    offsets = np.zeros(n_entities + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    codes = np.arange(1, n_entities + 1, dtype=np.int64)
    entity_ids = np.array([f"S1-{i + 1}" for i in range(n_entities)], dtype=object)
    ground_truth = GroundTruth(entity_ids, offsets, codes)

    folds = assign_folds(ground_truth, n_folds=130, mode="hash")
    assert folds.min() >= 0, "a wrapped fold id reads as unassigned"
    assert folds.max() < 130
    counts = np.bincount(folds.astype(np.int64), minlength=130)
    assert (counts > 0).all(), "every fold must own an entity"

    try:
        assign_folds(ground_truth, n_folds=MAX_FOLDS + 1, mode="hash")
    except ValueError as exc:
        assert "n_folds must be <=" in str(exc), f"unexpected message: {exc}"
    else:
        raise AssertionError("an out-of-range fold count must be rejected, not wrapped")


# ---------------------------------------------------------------------------
# Sweep and threshold selection
# ---------------------------------------------------------------------------
def test_sweep_matches_a_direct_evaluation_when_scores_are_distinct():
    """Every sweep point must equal the metric computed from a direct >= filter.

    The sweep accumulates per-entity counts incrementally from a sorted order; this
    recomputes each point from scratch with the repository's own metric, so an error
    in the accumulation cannot hide.
    """
    probabilities = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    is_true = EXPECTED_LABELS.astype(bool)
    owners = EXPECTED_OWNERS.astype(np.int64)
    lengths = EXPECTED_LENGTHS
    entity_mask = np.ones(len(lengths), dtype=bool)

    sweep = sweep_thresholds(probabilities, is_true, owners, lengths, entity_mask)

    assert sweep["positions"][0] == 0
    assert np.isinf(sweep["thresholds"][0])
    assert sweep["predicted_rows"][0] == 0

    for index, threshold in enumerate(sweep["thresholds"]):
        predicted = probabilities >= np.float32(threshold)
        counts = np.zeros(len(lengths), dtype=np.int64)
        hits = np.zeros(len(lengths), dtype=np.int64)
        for row in range(len(predicted)):
            if predicted[row]:
                counts[owners[row]] += 1
                if is_true[row]:
                    hits[owners[row]] += 1
        expected = CandidateEvaluation._macro_f05(
            lengths, counts, hits, policy="score_zero"
        )
        assert abs(sweep["macro_f05_score_zero"][index] - expected) < 1e-12, (
            f"point {index} (threshold {threshold}): sweep says "
            f"{sweep['macro_f05_score_zero'][index]}, direct says {expected}"
        )
        assert sweep["predicted_rows"][index] == int(counts.sum())

    # Predicting nothing scores the empty-GT share of the macro average, which the
    # fixture makes exactly 1/4: S1-16 is the only entity of four with no true match.
    assert abs(sweep["macro_f05_score_zero"][0] - 0.25) < 1e-12


def test_ties_are_resolved_by_ge_in_the_final_measurement():
    """Every point on the sweep curve must be a threshold the deployed rule can produce.

    The sweep walks rank prefixes ("the top k rows") while the rule applied to test data
    is ``p >= t``, which also predicts every row tied with the k-th. Where scores tie - and
    LightGBM's leaf values tie in very large groups, so this is the normal case rather
    than a corner - a rank prefix describes predictions no threshold can make, and a
    "flat plateau" spanning several such points is an artifact of subdividing one tie
    group. This test is the guard: it pins that the curve the threshold is chosen from is
    the curve the deployed rule actually has.
    """
    probabilities = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    is_true = EXPECTED_LABELS.astype(bool)
    owners = EXPECTED_OWNERS.astype(np.int64)
    lengths = EXPECTED_LENGTHS
    entity_mask = np.ones(len(lengths), dtype=bool)

    sweep = sweep_thresholds(probabilities, is_true, owners, lengths, entity_mask)

    # Two distinct scores, so at most three realizable points: predict nothing, the 0.9
    # group, or everything. A raw 200-point log grid over ten rows would claim ten.
    assert len(sweep["positions"]) <= 3, (
        f"the grid must collapse onto tie boundaries, got {sweep['positions'].tolist()}"
    )
    assert np.isinf(sweep["thresholds"][0]) and sweep["predicted_rows"][0] == 0

    for index in range(len(sweep["positions"])):
        threshold = float(sweep["thresholds"][index])
        predicted = probabilities >= np.float32(threshold)
        counts = np.zeros(len(lengths), dtype=np.int64)
        hits = np.zeros(len(lengths), dtype=np.int64)
        for row in range(len(predicted)):
            if predicted[row]:
                counts[owners[row]] += 1
                if is_true[row]:
                    hits[owners[row]] += 1

        # The row count must be what the threshold selects, not what the rank selected.
        assert sweep["predicted_rows"][index] == int(counts.sum()), (
            f"point {index} (threshold {threshold}): the curve claims "
            f"{sweep['predicted_rows'][index]} predictions, but p >= t selects {int(counts.sum())}"
        )
        assert abs(
            sweep["macro_f05_score_zero"][index]
            - CandidateEvaluation._macro_f05(lengths, counts, hits, policy="score_zero")
        ) < 1e-12

        measured = evaluate_at_threshold(
            threshold, probabilities, is_true, owners, lengths, entity_mask
        )
        assert measured["predicted_pairs"] == sweep["predicted_rows"][index]
        assert (
            abs(measured["macro_f05_score_zero"] - sweep["macro_f05_score_zero"][index]) < 1e-12
        ), "the re-measured operating point must equal the curve it was chosen from"

    # Predicting the whole 0.9 group is a point the rule can produce, and it is the best
    # one here - the top five rows are exactly the fixture's five true pairs.
    assert any(int(p) == 5 for p in sweep["predicted_rows"])
    assert max(sweep["macro_f05_score_zero"]) > sweep["macro_f05_score_zero"][0]


def test_nan_scores_are_never_predicted_at_any_threshold():
    """A blank feature must not be ranked as if it were a low score.

    ``p >= t`` is False when ``p`` is NaN, so a NaN row is never predicted whatever the
    threshold - but sorting NaN last would place it inside low-threshold prefixes, making
    the sweep claim predictions the rule cannot make. The threshold arm scores a raw
    feature column, which is exactly where blanks live.
    """
    probabilities = np.array([0.9, np.nan, 0.8, 0.7, np.nan, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    is_true = EXPECTED_LABELS.astype(bool)
    # Make the NaN rows true pairs, the worst case: if they were ranked in, the curve
    # would show recall it cannot deliver.
    is_true = is_true.copy()
    is_true[1] = True
    is_true[4] = True
    owners = EXPECTED_OWNERS.astype(np.int64)
    lengths = EXPECTED_LENGTHS
    entity_mask = np.ones(len(lengths), dtype=bool)

    sweep = sweep_thresholds(probabilities, is_true, owners, lengths, entity_mask)
    assert sweep["thresholds"][0] == np.inf

    for index in range(len(sweep["positions"])):
        threshold = float(sweep["thresholds"][index])
        assert not np.isnan(threshold), "a NaN score can never be a threshold"
        expected = int(np.count_nonzero(probabilities >= np.float32(threshold)))
        assert sweep["predicted_rows"][index] == expected
        measured = evaluate_at_threshold(
            threshold, probabilities, is_true, owners, lengths, entity_mask
        )
        assert measured["predicted_pairs"] == expected

    # Predicting everything usable still never counts a NaN row: eight finite scores.
    everything = np.nextafter(np.float32(0.2), np.float32(0))
    assert int(np.count_nonzero(probabilities >= everything)) == 8
    assert sweep["predicted_rows"].max() <= 8, (
        "NaN rows must never be in a predicted prefix: p >= t is False for NaN"
    )


def test_score_zero_singleton_semantics():
    """An empty-GT entity scores 1 only if nothing is predicted for it."""
    lengths = np.array([0, 0, 1], dtype=np.int64)
    empty_candidates = np.array([0, 0, 1], dtype=np.int64)
    empty_hits = np.array([0, 0, 1], dtype=np.int64)
    assert abs(macro_f05(lengths, empty_candidates, empty_hits) - 1.0) < 1e-12

    # One spurious pair on one of the two empty entities: that entity scores 0, so the
    # macro average is (1 + 0 + 1) / 3 - not "3/4 pairs correct".
    noisy = np.array([1, 0, 1], dtype=np.int64)
    assert abs(macro_f05(lengths, noisy, empty_hits) - (2.0 / 3.0)) < 1e-12

    # Both empty entities merged: 1/3.
    both = np.array([1, 1, 1], dtype=np.int64)
    assert abs(macro_f05(lengths, both, empty_hits) - (1.0 / 3.0)) < 1e-12


def test_plateau_selection_prefers_the_middle_over_the_edge_and_reports_a_spike():
    thresholds = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float64)
    scores = np.array([0.50, 0.60, 0.75, 0.75, 0.75, 0.60, 0.40], dtype=np.float64)
    sweep = {
        "thresholds": thresholds,
        "macro_f05_score_zero": scores,
        "macro_f05_exclude": scores,
        "predicted_rows": np.arange(7, dtype=np.int64),
        "positions": np.arange(7, dtype=np.int64),
    }
    from src.matching_model import select_operating_point

    choice = select_operating_point(sweep, tolerance=0.001)
    assert choice["best_index"] == 2 and choice["plateau_start_index"] == 2
    assert choice["plateau_end_index"] == 4
    assert choice["chosen_index"] == 3, "the middle of the plateau, not its edge"
    assert choice["plateau_points"] == 3
    assert choice["plateau_is_single_point"] is False
    assert abs(choice["chosen_macro_f05"] - 0.75) < 1e-12

    spike = dict(sweep)
    spike["macro_f05_score_zero"] = np.array([0.50, 0.90, 0.50, 0.50, 0.50, 0.50, 0.50])
    isolated = select_operating_point(spike, tolerance=0.001)
    assert isolated["plateau_is_single_point"] is True
    assert isolated["plateau_points"] == 1
    assert isolated["chosen_index"] == 1


def test_log_grid_covers_nothing_to_everything():
    grid = log_grid_positions(1_000_000, 50)
    assert grid[0] == 0
    assert grid[-1] == 1_000_000
    assert np.all(np.diff(grid) > 0), "positions must be strictly increasing"
    assert grid.min() >= 0 and grid.max() <= 1_000_000
    assert log_grid_positions(0).tolist() == [0]
    assert log_grid_positions(3, 100).tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Decisions and per-source diagnostics
# ---------------------------------------------------------------------------
def test_every_pair_above_the_threshold_is_predicted_no_top_1():
    bundle = ModelBundle(model="threshold", threshold=0.5, score_feature="name_token_set_ratio")
    probabilities = np.array([0.9, 0.8, 0.2, 0.7, 0.1], dtype=np.float32)
    decisions = decide({}, bundle, probabilities)
    assert decisions.tolist() == [True, True, False, True, False]

    s1_ids = ["S1-5", "S1-5", "S1-5", "S1-5", "S1-16"]
    target_ids = ["S2-1", "S2-2", "S2-3", "S3-1", "S2-4"]
    grouped = aggregate_matches(s1_ids, target_ids, decisions)
    assert grouped["S1-5"] == ["S2-1", "S2-2", "S3-1"], "one-to-many matches must all survive"
    # One row per S1 entity is required by the submission format, so an entity whose
    # candidates all fell below the threshold maps to an empty list rather than going
    # missing. (A test entity with no candidate rows at all is not in s1_ids at all -
    # that is what s1_universe is for.)
    assert grouped["S1-16"] == []

    with_universe = aggregate_matches(s1_ids, target_ids, decisions, s1_universe=["S1-9"])
    assert with_universe["S1-9"] == [], "an entity with no candidate rows still needs a row"
    assert with_universe["S1-5"] == grouped["S1-5"]

    # Duplicated and unsorted target ids must come out deduplicated and sorted, so a
    # rerun is byte-identical.
    repeated = aggregate_matches(
        ["S1-5"] * 3, ["S3-1", "S2-1", "S3-1"], np.array([True, True, True])
    )
    assert repeated["S1-5"] == ["S2-1", "S3-1"]


def test_training_without_the_matrix_fails_with_a_clear_message():
    """``write_matrix=False`` has nothing to train on, so it must say so, not crash.

    The old behaviour was a raw FileNotFoundError from deep inside numpy when the
    training code looked for a matrix the labelling pass had been told not to write.
    """
    fixture = _make_fixture(_temp_dir())
    try:
        train(
            fixture["config"],
            fixture["features_path"],
            _temp_dir() / "run_nomatrix",
            ground_truth=fixture["ground_truth"],
            model="threshold",
            folds=FIXTURE_FOLDS,
            fold_mode="hash",
            write_matrix=False,
        )
    except ValueError as exc:
        assert "write_matrix" in str(exc)
        assert "build_label_artifacts" in str(exc), f"no alternative offered: {exc}"
    else:
        raise AssertionError("training without the feature matrix must not proceed")


def test_per_source_diagnostics_split_s2_and_s3():
    fixture = _make_fixture(_temp_dir())
    from src.matching_model import per_source_diagnostics

    artifacts = _load_artifacts(fixture)
    owners = artifacts["owners"].astype(np.int64)
    is_true = artifacts["labels"].astype(bool)
    source_is_s2 = artifacts["sources"].astype(bool)
    lengths = EXPECTED_LENGTHS
    entity_mask = np.ones(len(lengths), dtype=bool)

    predicted = np.ones(len(is_true), dtype=bool)  # predict every candidate
    stats = per_source_diagnostics(
        fixture["ground_truth"], lengths, owners, is_true, predicted, source_is_s2, entity_mask
    )

    # True pairs: S1-5 has S2-1, S2-2 (S2) and S3-1 (S3); S1-10 has S2-1; S1-3 has
    # S2-5 and S2-6 (S2). So 5 S2 pairs and 1 S3 pair.
    assert stats["S2"]["n_true_pairs"] == 5
    assert stats["S3"]["n_true_pairs"] == 1
    # Candidate rows: 7 with source S2, 3 with source S3.
    assert stats["S2"]["predictions"] == 7
    assert stats["S3"]["predictions"] == 3
    assert stats["S2"]["true_positives"] == 4  # rows 0,1,5,9
    assert stats["S3"]["true_positives"] == 1  # row 3
    assert abs(stats["S2"]["pair_precision"] - 4 / 7) < 1e-12
    assert abs(stats["S3"]["pair_precision"] - 1 / 3) < 1e-12


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_end_to_end_threshold_arm_produces_every_artifact_and_the_hand_checked_f0_5():
    """The dependency-free arm, end to end, against a number computed by hand.

    From the fixture, with the score being ``name_token_set_ratio`` (0.95 on true
    pairs, 0.30 otherwise) and every true pair except S2-6 present as a candidate:

    * S1-16 (no true match) must predict nothing -> 1.0
    * S1-10: 1 true, 1 predicted -> 1.0
    * S1-5: 3 true, 3 predicted -> 1.0
    * S1-3: 2 true, 1 predicted -> precision 1.0, recall 0.5 -> 1.25*0.5/(0.25+0.5) = 5/6
    * macro = (1 + 1 + 1 + 5/6) / 4 = 0.9583333...
    """
    fixture = _make_fixture(_temp_dir())
    out_dir = _temp_dir() / "run"

    bundle = train(
        fixture["config"],
        fixture["features_path"],
        out_dir,
        ground_truth=fixture["ground_truth"],
        model="threshold",
        score_feature="name_token_set_ratio",
        folds=FIXTURE_FOLDS,
        fold_mode="hash",
        chunksize=4,
    )

    expected_f05 = (1.0 + 1.0 + 1.0 + (5.0 / 6.0)) / 4.0
    metrics = bundle.metrics
    final = metrics["val_metrics"]

    assert abs(bundle.threshold - 0.95) < 1e-6, f"threshold {bundle.threshold} != 0.95"
    assert abs(final["macro_f05_score_zero"] - expected_f05) < 1e-9, (
        f"macro F0.5 {final['macro_f05_score_zero']} != hand-computed {expected_f05}"
    )
    assert metrics["positive_labels"] == 5
    assert metrics["negative_labels"] == 5
    assert metrics["rows"] == 10
    assert metrics["fold_mode"] == "hash"
    assert metrics["n_folds"] == FIXTURE_FOLDS
    assert final["predicted_pairs"] == 5
    assert final["max_predictions_per_s1"] == 3, "one-to-many must survive to the metric"
    assert final["n_empty_gt_entities_with_predictions"] == 0

    # Baselines frame the result, and the accept-everything one is checked against the
    # metric formula computed by hand - a second point of the curve, independent of the
    # threshold the sweep chose:
    #   S1-5: 5 candidates / 3 true -> p=.6, r=1.0
    #   S1-10: 2 candidates / 1 true -> p=.5, r=1.0
    #   S1-16: empty ground truth, 2 predictions -> 0.0 under score_zero
    #   S1-3: 1 candidate / 1 hit, 2 true -> p=1.0, r=0.5
    def _f05(precision: float, recall: float) -> float:
        return 1.25 * precision * recall / (0.25 * precision + recall)

    expected_accept_all = (_f05(0.6, 1.0) + _f05(0.5, 1.0) + 0.0 + _f05(1.0, 0.5)) / 4.0
    assert abs(metrics["baselines"]["predict_nothing_macro_f05_score_zero"] - 0.25) < 1e-12
    assert (
        abs(
            metrics["baselines"]["accept_every_candidate_macro_f05_score_zero"]
            - expected_accept_all
        )
        < 1e-9
    ), "merging every candidate must score the hand-computed baseline"
    assert expected_accept_all > 0.25, "the fixture's baselines must differ, or they frame nothing"

    for name in (
        "v1_metrics.json",
        "v1_labels_report.json",
        "threshold_sweep.csv",
        "oof_probabilities.npy",
        "val_labels.npy",
        "val_owner_index.npy",
        "val_source_is_s2.npy",
        "val_features.npy",
        "model/model_meta.json",
    ):
        assert (out_dir / name).is_file(), f"missing artifact: {name}"

    oof = np.load(out_dir / "oof_probabilities.npy")
    assert len(oof) == 10 and (oof >= 0).all(), "every row must carry an out-of-fold score"

    # The prediction rule must reproduce the reported count exactly.
    assert int(decide(fixture["config"], bundle, oof).sum()) == final["predicted_pairs"]

    # A saved bundle reloads and scores identically.
    reloaded = load_bundle(out_dir)
    assert reloaded.model == "threshold"
    assert abs(reloaded.threshold - bundle.threshold) < 1e-12
    assert np.allclose(reloaded.metrics["threshold"], bundle.threshold)
    assert np.allclose(
        predict(fixture["config"], reloaded, np.load(out_dir / "val_features.npy")), oof
    )


def test_the_repositorys_own_evaluator_agrees_with_the_reported_macro_f0_5():
    """Score the predicted pairs through evaluation.py's file-based path.

    This is the check that the matcher and the graded metric cannot drift: the pairs
    the matcher would emit are written as a candidate TSV and handed to
    ``CandidateEvaluation.evaluate_file``, which must return the same macro F0.5.
    """
    fixture = _make_fixture(_temp_dir())
    out_dir = _temp_dir() / "run_eval"
    bundle = train(
        fixture["config"],
        fixture["features_path"],
        out_dir,
        ground_truth=fixture["ground_truth"],
        model="threshold",
        folds=FIXTURE_FOLDS,
        fold_mode="hash",
        chunksize=4,
    )

    oof = np.load(out_dir / "oof_probabilities.npy")
    kept = decide(fixture["config"], bundle, oof)
    predictions_path = out_dir / "predicted_pairs.tsv"
    header = ["source1_entity_id", "matched_entity_id", "source", "score"]
    rows = [
        [
            FEATURE_ROWS[row][0],
            FEATURE_ROWS[row][1],
            FEATURE_ROWS[row][2],
            float(oof[row]),
        ]
        for row in range(len(FEATURE_ROWS))
        if kept[row]
    ]
    _write_tsv(predictions_path, header, rows)

    evaluator = CandidateEvaluation(fixture["ground_truth"])
    reported = evaluator.evaluate_file(predictions_path, chunksize=4)

    assert reported["n_s1_entities"] == len(EXPECTED_LENGTHS)
    assert reported["n_candidate_pairs"] == int(kept.sum())
    assert reported["n_true_pairs"] == int(EXPECTED_LENGTHS.sum())
    assert abs(reported["f05_accept_all_macro_score_zero"] - bundle.metrics["val_metrics"]["macro_f05_score_zero"]) < 1e-12, (
        "the matcher's reported macro F0.5 must equal the evaluator's for the same pairs"
    )
    assert abs(reported["f05_accept_all_macro"] - bundle.metrics["val_metrics"]["macro_f05_exclude"]) < 1e-12


def test_end_to_end_lightgbm_when_available_or_a_usable_error_when_not():
    """V1 proper: LightGBM trains per fold and fills the out-of-fold scores.

    Where LightGBM is absent - as on the machine this was written - the matcher must
    fail with something actionable rather than a bare ImportError, so that path is
    asserted too instead of being skipped silently.
    """
    fixture = _make_fixture(_temp_dir())
    out_dir = _temp_dir() / "run_lgbm"

    if not _has_lightgbm():
        try:
            train(
                fixture["config"],
                fixture["features_path"],
                out_dir,
                ground_truth=fixture["ground_truth"],
                model="lightgbm",
                folds=FIXTURE_FOLDS,
                chunksize=4,
            )
        except ImportError as exc:
            message = str(exc)
            assert "lightgbm" in message.lower()
            assert "pip install lightgbm" in message, f"no install hint in: {message}"
            assert "--model threshold" in message, f"no dependency-free fallback in: {message}"
        else:
            raise AssertionError("model=lightgbm must not succeed without LightGBM installed")
        return

    bundle = train(
        fixture["config"],
        fixture["features_path"],
        out_dir,
        ground_truth=fixture["ground_truth"],
        model="lightgbm",
        folds=FIXTURE_FOLDS,
        fold_mode="hash",
        n_estimators=20,
        # The fixture has ten rows, so the production leaf minimum (200) would leave
        # every tree a single stump. This is the one place a parameter is loosened, and
        # only because the fixture is tiny - see default_params for the real settings.
        params={**default_params(seed=42), "min_data_in_leaf": 1},
        chunksize=4,
    )

    assert len(bundle.boosters) == FIXTURE_FOLDS, "one model per fold"
    oof = np.load(out_dir / "oof_probabilities.npy")
    assert len(oof) == 10
    assert (oof >= 0).all() and (oof <= 1).all(), "probabilities must be filled and in range"
    assert np.isfinite(bundle.threshold)
    for fold in range(FIXTURE_FOLDS):
        assert (out_dir / "model" / f"fold_{fold}.txt").is_file()

    reloaded = load_bundle(out_dir)
    assert len(reloaded.boosters) == FIXTURE_FOLDS
    assert abs(reloaded.threshold - bundle.threshold) < 1e-12

    # The saved models must score identically to the ones still in memory, and twice in
    # a row. Note this is the *ensemble* prediction (every fold's model averaged), which
    # is deliberately not the out-of-fold vector above: an out-of-fold row is scored by
    # the one model that never saw its entity, and using the ensemble instead is exactly
    # the leak the folds exist to prevent.
    matrix = np.load(out_dir / "val_features.npy")
    ensemble = predict(fixture["config"], bundle, matrix)
    reloaded_ensemble = predict(fixture["config"], reloaded, matrix)
    assert np.allclose(ensemble, reloaded_ensemble, atol=1e-9), "save/load must round-trip"
    assert np.allclose(ensemble, predict(fixture["config"], reloaded, matrix), atol=1e-12)
    # The ensemble path (every fold's model averaged) is a different number from the
    # out-of-fold vector, by design: an out-of-fold row is scored by the one model that
    # never saw its entity, and using the ensemble there is exactly the leak the folds
    # exist to prevent. So each fold's out-of-fold scores are checked against the single
    # saved booster that produced them - a direct test of the out-of-fold wiring.
    row_folds = assign_folds(
        fixture["ground_truth"], n_folds=FIXTURE_FOLDS, mode="hash"
    )[np.load(out_dir / "val_owner_index.npy").astype(np.int64)]
    for fold in range(FIXTURE_FOLDS):
        rows = np.flatnonzero(row_folds == fold)
        assert len(rows), f"fold {fold} holds no rows out"
        from_that_model = bundle.boosters[fold].predict(matrix[rows])
        assert np.allclose(oof[rows], from_that_model, atol=1e-6), (
            f"fold {fold}: the out-of-fold scores must come from fold {fold}'s own model, "
            "trained without these rows"
        )


def test_oof_probabilities_are_filled_for_every_row_and_never_negative():
    """The sentinel check: an unfilled row would mean a model scored its own training data."""
    fixture = _make_fixture(_temp_dir())
    artifacts = _load_artifacts(fixture)
    owners = artifacts["owners"].astype(np.int64)
    entity_folds = assign_folds(fixture["ground_truth"], n_folds=FIXTURE_FOLDS, mode="hash")
    row_folds = entity_folds[owners]

    probabilities = np.full(len(owners), -1.0, dtype=np.float32)
    assert int(np.count_nonzero(probabilities < 0)) == len(owners)
    # Every fold's held-out rows must be filled by that fold, and nothing else.
    for fold in range(FIXTURE_FOLDS):
        probabilities[row_folds == fold] = 0.5
    assert int(np.count_nonzero(probabilities < 0)) == 0
    assert np.allclose(probabilities, 0.5)


def test_every_entity_is_scored_by_a_model_that_never_saw_the_entity():
    """Out-of-fold stated per S1 entity, against the *saved* models.

    This follows from fold purity, which is why it is worth asserting on its own: it is
    the property the whole design exists to protect, and it is checked against the models
    that produced the scores rather than against the fold array. A prediction step that
    quietly used the ensemble - fitted on every row - would pass a fold-array check and
    fail here.

    The ``threshold`` arm fits nothing at all, so there is no training set for it to
    leak from; it is covered by its own end-to-end test.
    """
    if not _has_lightgbm():
        return

    fixture = _make_fixture(_temp_dir())
    out_dir = _temp_dir() / "run_oof_entity"
    bundle = train(
        fixture["config"],
        fixture["features_path"],
        out_dir,
        ground_truth=fixture["ground_truth"],
        model="lightgbm",
        folds=FIXTURE_FOLDS,
        fold_mode="hash",
        n_estimators=20,
        params={**default_params(seed=42), "min_data_in_leaf": 1},
        chunksize=4,
    )

    oof = np.load(out_dir / "oof_probabilities.npy")
    owners = np.load(out_dir / "val_owner_index.npy").astype(np.int64)
    matrix = np.load(out_dir / "val_features.npy")
    entity_folds = assign_folds(fixture["ground_truth"], n_folds=FIXTURE_FOLDS, mode="hash")
    row_folds = entity_folds[owners]

    for position in np.unique(owners):
        rows = np.flatnonzero(owners == position)
        fold = int(entity_folds[position])
        assert len(np.unique(row_folds[rows])) == 1, (
            f"entity at position {position} spans folds {np.unique(row_folds[rows]).tolist()}"
        )
        # Its scores come from that fold's model - and no other model could have
        # produced them, because the ensemble is a different function.
        assert np.allclose(
            oof[rows], bundle.boosters[fold].predict(matrix[rows]), atol=1e-6
        ), f"entity {position} was not scored by fold {fold}'s model"
        # And that model did not train on this entity: none of its rows is in the mask
        # the fold's own booster was fitted on.
        trained_on = row_folds != fold
        assert not trained_on[rows].any(), (
            f"entity {position} is inside fold {fold}'s training mask while being scored by it"
        )

    # The same statement the other way round: no entity is on both sides of a fold.
    for fold in range(FIXTURE_FOLDS):
        held_out = row_folds == fold
        assert not np.intersect1d(owners[held_out], owners[~held_out]).size


def test_training_survives_a_ground_truth_entity_the_features_do_not_cover():
    """The failing HPC command in miniature, through the whole ``train()`` path.

    The ground truth has four entities and the feature file has rows for three, because
    the blockers proposed nothing for S1-10. That is ordinary - it is what the blocking
    recall number measures - and the run must complete. It also must not drop the
    uncovered entity from the metric: its one true match is unreachable, so it scores 0
    rather than being averaged away.
    """
    fixture = _make_fixture(_temp_dir(), skip_s1=("S1-10",))
    out_dir = _temp_dir() / "run_uncovered"

    bundle = train(
        fixture["config"],
        fixture["features_path"],
        out_dir,
        ground_truth=fixture["ground_truth"],
        model="threshold",
        score_feature="name_token_set_ratio",
        folds=FIXTURE_FOLDS,
        fold_mode="hash",
        chunksize=4,
    )
    metrics = bundle.metrics

    # The check reports what it examined, so this state is visible in the run rather
    # than implied: three of the four ground-truth entities own candidate rows.
    assert metrics["fold_purity"] == {
        "rows": 8,
        "entities_checked": 3,
        "entity_index_span": 4,
    }
    assert metrics["labels"]["n_s1_entities_in_ground_truth"] == 4
    assert metrics["labels"]["n_s1_entities_with_rows"] == 3
    assert metrics["rows"] == 8

    # S1-5 -> 1.0, S1-16 (no true match, nothing predicted) -> 1.0, S1-3 -> 5/6,
    # S1-10 -> 0.0: nothing was predicted and it has a true match, so it is a miss.
    expected = (1.0 + 0.0 + 1.0 + (5.0 / 6.0)) / 4.0
    assert abs(metrics["val_metrics"]["macro_f05_score_zero"] - expected) < 1e-9, (
        f"macro F0.5 {metrics['val_metrics']['macro_f05_score_zero']} != {expected}"
    )
    assert metrics["val_metrics"]["predicted_pairs"] == 4


def test_entity_metrics_report_one_to_many_without_truncating():
    lengths = np.array([3, 0, 2], dtype=np.int64)
    candidates = np.array([3, 0, 1], dtype=np.int64)
    hits = np.array([3, 0, 1], dtype=np.int64)
    stats = entity_metrics(lengths, candidates, hits)
    assert stats["max_predictions_per_s1"] == 3
    assert stats["true_positives"] == 4
    assert stats["false_positives"] == 0
    assert stats["false_negatives"] == 1
    assert abs(stats["pair_precision"] - 1.0) < 1e-12
    assert abs(stats["pair_recall"] - 4 / 5) < 1e-12
    assert stats["n_empty_gt_entities"] == 1
    assert stats["n_empty_gt_entities_with_predictions"] == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    try:
        for name, test in tests:
            try:
                test()
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
            else:
                print(f"[PASS] {name}")
    finally:
        for directory in _TEMP_DIRS:
            shutil.rmtree(directory, ignore_errors=True)

    print()
    if failures:
        print(f"{failures} of {len(tests)} tests failed")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
