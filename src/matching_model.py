"""Match classifier.  **V1 implemented: entity-grouped LightGBM + out-of-fold threshold sweep.**

This is the stage that turns candidate pairs into match decisions. V1 is the first
honest baseline, not a finished model: it trains on the features
``scripts/extract_pair_features.py`` already wrote, out-of-fold, and reports the
challenge metric of a threshold chosen on those out-of-fold scores.

Why the metric shapes the design
--------------------------------
The challenge scores F0.5, computed **per S1 entity** and then macro-averaged.
Two consequences drive every choice here:

1. **Precision is worth 4x recall.** ``beta = 0.5`` means a false positive costs
   four times a false negative. With 7,638,365 true matches spread over
   22.8 trillion possible pairs, and 123,247 S1 entities that have no match at
   all, a classifier with mediocre precision destroys the macro average: one
   spurious match on an empty-GT entity takes that entity's F0.5 to zero.

2. **Per-entity aggregation, not per-pair.** Averaging over entities stops a
   single S1 with 11 matches from dominating, and it means a per-entity decision
   policy (e.g. a threshold tuned per candidate-count bucket) is legitimate and
   often better than one global threshold.

Interface
---------
``train(config, features_path, out_dir, ...) -> ModelBundle``
``predict(config, ModelBundle, candidate_features) -> np.ndarray``
``decide(config, ModelBundle, probabilities) -> np.ndarray``
``tune_threshold(config, sweep, ...) -> dict``

Recommended progression (measure at each step, do not skip ahead):

1. **Threshold on a lexical score** - a single feature (e.g. token-set ratio)
   with a threshold tuned on the val split. Gives a real F0.5 number and a floor
   to beat. Deliberately CPU: thresholding one score on a few million pairs
   needs no accelerator, and adding one would only add a transfer cost. Available
   here as ``--model threshold``, which needs no third-party dependency.
2. **Gradient-boosted trees** on the lexical + address features. This is V1
   (``--model lightgbm``), CPU by default - for 27 features the GPU histogram
   path often loses to a well-threaded CPU build - so benchmark ``device=cuda``
   on a real sample before switching. This is expected to be the bulk of the
   score. Handle missing addresses natively.
3. **Add semantic features** from multilingual embeddings, GPU when available.
4. **Re-ranking with a cross-encoder** on the top candidates only, if it still
   pays for itself after step 3.

What V1 does, and what it deliberately does not
-----------------------------------------------
Implemented here:

* **Pair-level labels, from the evaluator's own membership test.** A candidate
  pair is positive iff its packed ``(S1, target)`` code is present in
  ``build_true_pair_codes(ground_truth)``. ``label_pairs`` reproduces
  ``CandidateEvaluation.evaluate_file``'s own lines (``positions_of`` ->
  ``_encode_target_codes`` -> ``owners * PAIR_MULTIPLIER + target_codes`` ->
  ``_contains_sorted``) rather than restating what a true pair is, so the training
  label and the graded metric cannot drift apart. ``PAIR_MULTIPLIER`` is imported
  from ``src/blocking.py`` and ``ID_NUMERIC_MODULUS`` from ``src/utils.py`` - the
  same constants the evaluator and the blockers use, not copies of them.
* **Pair-level, never entity-level.** An S1 with three true matches of which the
  blocker proposed two gets labels ``[1, 1, 0]`` on its three candidate rows. It
  is never labelled 0 because one of its matches was missed; marking the whole
  entity negative would teach the model to reject true pairs.
* **Entity-grouped folds.** Every candidate row of one S1 entity lands in one fold
  (``assign_folds``), so an out-of-fold prediction comes from a model that never
  saw that entity's name, address or competitor set. A row-level split would leak
  exactly that and quietly inflate the score.
* **Out-of-fold probabilities** for every row, so the threshold sweep and the
  reported F0.5 are measured on scores the model never fit.
* **A threshold sweep scored by the challenge metric** - the official
  ``CandidateEvaluation._macro_f05``, under the ``score_zero`` policy that charges
  a false merge on an empty-GT entity the full 0.
* **A stable operating point**, chosen from a plateau rather than the single best
  grid point, then re-measured with the exact ``probability >= t`` rule so the
  reported F0.5 is the one the threshold actually achieves.

Not implemented, on purpose (see README "Roadmap"):

* no embeddings, no cross-encoder, no LambdaRank, no new features
* no resampling and no ``scale_pos_weight``: the natural class distribution is
  preserved, because resampling moves the score distribution the threshold is
  later read off
* no top-1 assumption anywhere: every pair at or above the threshold is predicted,
  so a one-to-many entity keeps all of its surviving matches
* no test-split pipeline and no ``matching_results.tsv`` (``scripts/predict.py``
  stays a stub until the test features exist and the submission format is
  confirmed). ``aggregate_matches`` is here because the decision rule and the
  submission shape are the same fact.

Memory
------
This module never loads the 67.3M-row feature matrix into RAM. One streaming pass
writes it to a memory-mapped ``float32`` file sized from the row count, so peak RSS
is one chunk plus the artifacts:

===========================  =========  ==========================
artifact                     size       where
===========================  =========  ==========================
``val_features.npy``         7.27 GiB   67.3M x 27 x float32, mmap
``val_labels.npy``           64 MiB     int8, row-aligned
``val_owner_index.npy``      257 MiB    int32 S1 position per row
``val_source_is_s2.npy``     64 MiB     int8
``oof_probabilities.npy``    257 MiB    float32, row-aligned
one fold's training copy     ~5.8 GiB   released before the next fold
LightGBM's binned dataset    ~1.5 GiB   per fold, released with it
===========================  =========  ==========================

Peak ~10-12 GiB, against the ~500 GiB the target node has. The alternative - one
dense RAM array for all folds plus a training copy - would be ~20 GiB and buys
nothing.

The candidate-cap interaction
-----------------------------
``blocking.max_candidates_per_source`` controls how many candidates the matcher
sees per S1. This is a recall/precision trade-off, not a tuning detail: capping
too aggressively silently deletes true matches before the model ever sees them,
which shows up as a hard ceiling on per-entity F0.5. Choose the cap from
``recall_at_k_file_order`` once blockers produce ranked scores, and re-check the
blocking report after every blocker change.

Constraints
-----------
* CPU must remain sufficient for the whole pipeline. A GPU is an accelerator for
  specific stages, never a requirement: resolve it with
  ``utils.resolve_device_from_config(config)`` rather than hardcoding a device
  string. See README "Compute architecture" for which stages benefit.
* Model license and parameter-count constraints apply to whatever is finally
  submitted - record them here before shipping. LightGBM is MIT-licensed.
* No external data or internet augmentation.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from .blocking import PAIR_MULTIPLIER
from .data_loader import TSV_QUOTING, GroundTruth, load_ground_truth
from .evaluation import (
    CANDIDATE_S1_COLUMN,
    CANDIDATE_SOURCE_COLUMN,
    CANDIDATE_TARGET_COLUMN,
    CandidateEvaluation,
    _contains_sorted,
    _encode_target_codes,
    _macro_entity_recall,
    build_true_pair_codes,
    split_mask_for,
)
from .utils import (
    ID_NUMERIC_MODULUS,
    ensure_dir,
    fmt_int,
    human_bytes,
    read_json,
    stable_hash64,
    write_json,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# The 41 model features, in the order ``scripts/extract_pair_features.py`` writes
# them (its FEATURE_DTYPES minus NON_FEATURE_COLUMNS). ``dense_cosine`` and
# ``blocker_dense`` carry the dense blocker's evidence: without them a dense-only
# (e.g. cross-script) pair would be judged on lexical features alone, which are ~0
# for it by construction, and rejected. The twelve ``s1ctx_*`` columns (V2) place
# each candidate within its S1's candidate group. Kept here as a tuple rather
# than imported from the script - ``src`` does not import from ``scripts`` - and
# checked against the feature file's own header at read time by
# ``resolve_feature_columns``, so a change on either side fails loudly instead of
# silently handing the model a different matrix.
FEATURE_COLUMNS: tuple[str, ...] = (
    "name_norm_equal",
    "name_key_equal",
    "name_token_jaccard",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_char3_jaccard",
    "name_length_ratio",
    "name_token_count_diff",
    "name_first_token_equal",
    "address_norm_equal",
    "address_token_jaccard",
    "address_shared_token_count",
    "address_length_ratio",
    "s1_address_missing",
    "target_address_missing",
    "both_address_missing",
    "token_df",
    "char_jaccard",
    "dense_cosine",
    "blocker_exact_name",
    "blocker_token",
    "blocker_char_ngram",
    "blocker_dense",
    "n_blockers",
    "s1_candidate_count",
    "source_is_s2",
    "country_equal",
    "country_missing",
    "s1ctx_name_token_set_ratio_rank",
    "s1ctx_name_token_set_ratio_gap_to_best",
    "s1ctx_name_token_set_ratio_ratio_to_max",
    "s1ctx_name_token_set_ratio_ratio_to_mean",
    "s1ctx_name_char3_jaccard_rank",
    "s1ctx_name_char3_jaccard_gap_to_best",
    "s1ctx_name_char3_jaccard_ratio_to_max",
    "s1ctx_name_char3_jaccard_ratio_to_mean",
    "s1ctx_dense_cosine_rank",
    "s1ctx_dense_cosine_gap_to_best",
    "s1ctx_dense_cosine_ratio_to_max",
    "s1ctx_dense_cosine_ratio_to_mean",
)

# dtypes to parse those columns with, mirrored from the extractor. Reading them as
# float32 up front is what keeps the matrix at 7.27 GiB on disk rather than ~15 GiB.
FEATURE_DTYPES: dict[str, str] = {
    "name_norm_equal": "uint8",
    "name_key_equal": "uint8",
    "name_token_jaccard": "float32",
    "name_token_set_ratio": "float32",
    "name_token_sort_ratio": "float32",
    "name_partial_ratio": "float32",
    "name_char3_jaccard": "float32",
    "name_length_ratio": "float32",
    "name_token_count_diff": "int16",
    "name_first_token_equal": "uint8",
    "address_norm_equal": "uint8",
    "address_token_jaccard": "float32",
    "address_shared_token_count": "int16",
    "address_length_ratio": "float32",
    "s1_address_missing": "uint8",
    "target_address_missing": "uint8",
    "both_address_missing": "uint8",
    "token_df": "float32",
    "char_jaccard": "float32",
    "dense_cosine": "float32",
    "blocker_exact_name": "uint8",
    "blocker_token": "uint8",
    "blocker_char_ngram": "uint8",
    "blocker_dense": "uint8",
    "n_blockers": "uint8",
    "s1_candidate_count": "int32",
    "source_is_s2": "uint8",
    "country_equal": "uint8",
    "country_missing": "uint8",
    "s1ctx_name_token_set_ratio_rank": "float32",
    "s1ctx_name_token_set_ratio_gap_to_best": "float32",
    "s1ctx_name_token_set_ratio_ratio_to_max": "float32",
    "s1ctx_name_token_set_ratio_ratio_to_mean": "float32",
    "s1ctx_name_char3_jaccard_rank": "float32",
    "s1ctx_name_char3_jaccard_gap_to_best": "float32",
    "s1ctx_name_char3_jaccard_ratio_to_max": "float32",
    "s1ctx_name_char3_jaccard_ratio_to_mean": "float32",
    "s1ctx_dense_cosine_rank": "float32",
    "s1ctx_dense_cosine_gap_to_best": "float32",
    "s1ctx_dense_cosine_ratio_to_max": "float32",
    "s1ctx_dense_cosine_ratio_to_mean": "float32",
}

# Id / provenance columns carried by the feature file. Never features: they identify
# the pair rather than describe it.
ID_COLUMNS: tuple[str, ...] = (
    CANDIDATE_S1_COLUMN,
    CANDIDATE_TARGET_COLUMN,
    CANDIDATE_SOURCE_COLUMN,
)

# Written so a failed text join stays diagnosable, and deliberately dropped before
# training - a column saying "the feature pipeline worked" is not a property of the
# pair and would be a label-adjacent shortcut.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("text_join_ok",)

SOURCE_IS_S2_COLUMN = "source_is_s2"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
# Absolute F0.5 slack that still counts as "the same operating point". The curve is
# flat over a wide threshold range, so the midpoint of the flat region is a far more
# stable choice than the argmax, which can be a one-grid-point spike.
DEFAULT_PLATEAU_TOLERANCE = 0.002
DEFAULT_SWEEP_POINTS = 200
DEFAULT_REFINE_POINTS = 50
DEFAULT_PREDICT_CHUNK = 5_000_000
DEFAULT_CHUNKSIZE = 500_000

# Two arms. ``lightgbm`` is the V1 model; ``threshold`` needs no dependency and
# exists so the whole label -> fold -> out-of-fold -> sweep -> metric path can be
# exercised (and regression-tested) without LightGBM installed. It is also the
# documented first step of the progression above, i.e. the floor to beat.
MODEL_LIGHTGBM = "lightgbm"
MODEL_THRESHOLD = "threshold"
MODELS = (MODEL_LIGHTGBM, MODEL_THRESHOLD)

FOLD_MODE_AUTO = "auto"
FOLD_MODE_HASH = "hash"
FOLD_MODE_GROUPKFOLD = "groupkfold"
FOLD_MODES = (FOLD_MODE_AUTO, FOLD_MODE_HASH, FOLD_MODE_GROUPKFOLD)

# The policy the threshold is tuned under. ``score_zero`` is the challenge's own
# treatment of the 123,247 S1 entities with no true match: predicting nothing for
# them scores 1, predicting anything scores 0. Tuning under ``exclude`` instead
# would ignore that penalty entirely and pick a threshold that merges entities.
ZERO_MATCH_POLICY = "score_zero"
OTHER_POLICY = "exclude"

# Per-entity fold ids. ``int16`` rather than ``int8``: with ``int8`` a fold count above
# 127 wraps to a negative id, which ``assign_folds`` reads as "this entity was never
# assigned" and silently re-places by hash - an entity split across folds, from a CLI
# flag. 32,767 folds is far past any real use, and the guard in ``assign_folds`` says so
# explicitly rather than letting the wrap happen.
FOLD_DTYPE = np.int16
MAX_FOLDS = int(np.iinfo(FOLD_DTYPE).max)

LIGHTGBM_MISSING_MESSAGE = (
    "LightGBM is required for --model lightgbm but is not installed.\n"
    "  pip install lightgbm            # MIT licence, CPU by default\n"
    "or run the dependency-free arm to exercise the pipeline end to end:\n"
    "  --model threshold --score-feature name_token_set_ratio"
)


# ---------------------------------------------------------------------------
# Reading the feature matrix
# ---------------------------------------------------------------------------
def resolve_feature_columns(header: Sequence[str]) -> list[str]:
    """The feature columns present in a feature file, verified against the header.

    Raises rather than guessing: if the extractor's feature set and this module's
    disagree, the model would train on a different matrix than the one it reports.
    """
    available = set(header)
    missing = [c for c in FEATURE_COLUMNS if c not in available]
    if missing:
        raise ValueError(
            "feature file is missing columns the matcher expects: "
            f"{missing}\n  present: {sorted(available)}"
        )
    extra = sorted(
        available - set(FEATURE_COLUMNS) - set(ID_COLUMNS) - set(NON_FEATURE_COLUMNS)
    )
    if extra:
        raise ValueError(
            f"feature file carries columns this matcher does not know: {extra}\n"
            "  Add them to FEATURE_COLUMNS (and FEATURE_DTYPES) deliberately - do not "
            "let the model pick up an unnamed column."
        )
    return list(FEATURE_COLUMNS)


def iter_feature_chunks(
    path: str | Path,
    columns: Sequence[str],
    chunksize: int = DEFAULT_CHUNKSIZE,
    sample_rows: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Stream a feature TSV with the declared dtypes.

    Deliberately not ``data_loader.iter_tsv``: that forces ``dtype=str`` with
    ``na_filter=False``, under which a blank evidence cell raises
    (``ValueError: could not convert string to float: ''``) instead of becoming
    NaN. The blanks are load-bearing - ``token_df`` is blank on every pair the
    token blocker did not propose - so ``na_values=[""]`` with numeric dtypes
    applied is what makes them arrive as NaN and reach LightGBM as "missing".
    """
    path = Path(path)
    dtypes = {c: FEATURE_DTYPES[c] for c in columns if c in FEATURE_DTYPES}
    remaining = sample_rows
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=list(columns),
        dtype=dtypes,
        na_values=[""],
        keep_default_na=False,
        na_filter=True,
        quoting=TSV_QUOTING,
        chunksize=chunksize,
        on_bad_lines="warn",
    )
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                return
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        yield chunk


def count_data_rows(path: str | Path) -> int:
    """Data rows in a TSV, by a byte scan (no parsing) - the memmap needs the size."""
    path = Path(path)
    if path.suffix == ".gz":  # pragma: no cover - the extractor writes plain TSV
        import gzip

        with gzip.open(path, "rb") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    with open(path, "rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


# ---------------------------------------------------------------------------
# Labelling (pair-level, official membership test)
# ---------------------------------------------------------------------------
def label_pairs(
    frame: pd.DataFrame,
    ground_truth: GroundTruth,
    true_pair_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(owner_position, is_true)`` for one chunk of candidate feature rows.

    The body is ``CandidateEvaluation.evaluate_file``'s own four lines, reproduced
    rather than reinterpreted - so "true pair" has exactly one definition in this
    repository::

        owners = ground_truth.positions_of(chunk[CANDIDATE_S1_COLUMN])
        target_codes = _encode_target_codes(chunk[CANDIDATE_TARGET_COLUMN])
        packed = owners * PAIR_MULTIPLIER + target_codes
        is_true = _contains_sorted(true_pair_codes, packed)

    ``owner_position`` indexes the ground-truth table (``-1`` for an S1 id absent
    from the ground truth, which the caller drops). Where the evaluator filters
    unknown ids out and counts them, this keeps them long enough to be counted and
    then drops them, because the label artifact has to stay row-aligned with the
    matrix.

    Pair-level, never entity-level: a positive is this exact ``(S1, target)`` being
    in the ground truth. An S1 with three true matches and two of them proposed by
    the blocker gets labels ``[1, 1, 0]`` on its three candidate rows - not "entity
    is wrong, everything 0".
    """
    owners = ground_truth.positions_of(frame[CANDIDATE_S1_COLUMN])
    target_codes = _encode_target_codes(frame[CANDIDATE_TARGET_COLUMN])
    known = owners >= 0
    # Unknown ids pack to owner 0, whose pairs could collide with a real code, so the
    # membership result is discarded for them rather than relied upon.
    packed = np.where(known, owners, 0).astype(np.int64) * PAIR_MULTIPLIER + target_codes
    is_true = _contains_sorted(true_pair_codes, packed) & known
    return owners, is_true


def build_label_artifacts(
    features_path: str | Path,
    ground_truth: GroundTruth,
    out_dir: str | Path,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    sample_rows: Optional[int] = None,
    write_matrix: bool = True,
    log: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """One streaming pass over the feature file: labels, group keys, and the matrix.

    Writes, all row-for-row aligned with the feature file

    * ``val_labels.npy``        int8,  1 = the pair is in the ground truth
    * ``val_owner_index.npy``   int32, ground-truth row of the pair's S1 entity
    * ``val_source_is_s2.npy``  int8,  1 when the candidate is an S2 record
    * ``val_features.npy``      float32 ``(rows, 27)``, written through a memmap

    The matrix goes straight to a memory-mapped file whose size is known from the row
    count, so peak RSS stays at one chunk: 7.27 GiB on disk and nothing like it in
    RAM. ``write_matrix=False`` reads only the columns labelling needs.

    The feature TSV is read, never rewritten - it is the input of record.
    """
    log = log or logger
    features_path = Path(features_path)
    out_dir = ensure_dir(out_dir)

    header = _read_header(features_path)
    feature_columns = resolve_feature_columns(header)

    total_rows = count_data_rows(features_path)
    rows_to_read = min(total_rows, int(sample_rows)) if sample_rows else total_rows
    if rows_to_read <= 0:
        raise ValueError(f"feature file has no data rows: {features_path}")

    # ``source_is_s2`` is one of the 27 features *and* the per-source diagnostic key,
    # so it is always read; the others are read only when the matrix is written.
    if write_matrix:
        columns = list(ID_COLUMNS) + feature_columns
        matrix_path: Optional[Path] = out_dir / "val_features.npy"
        partial = out_dir / "val_features.npy.partial"
        matrix = np.lib.format.open_memmap(
            partial, mode="w+", dtype="float32", shape=(rows_to_read, len(feature_columns))
        )
    else:
        columns = list(ID_COLUMNS) + [SOURCE_IS_S2_COLUMN]
        matrix_path, partial, matrix = None, None, None

    labels = np.zeros(rows_to_read, dtype=np.int8)
    owners = np.full(rows_to_read, -1, dtype=np.int32)
    sources = np.zeros(rows_to_read, dtype=np.int8)

    true_pair_codes = build_true_pair_codes(ground_truth)
    log.info(
        "labelling %s feature rows against %s ground-truth pairs (matrix=%s)",
        fmt_int(rows_to_read),
        fmt_int(len(true_pair_codes)),
        "on" if write_matrix else "off",
    )

    started = time.time()
    written = 0
    read = 0
    dropped = 0
    dropped_examples: list[str] = []
    positives = 0
    rows_for_empty_entities = 0
    lengths_all = ground_truth.lengths()
    for chunk in iter_feature_chunks(
        features_path, columns, chunksize=chunksize, sample_rows=rows_to_read
    ):
        read += len(chunk)
        owner_chunk, true_chunk = label_pairs(chunk, ground_truth, true_pair_codes)
        known = owner_chunk >= 0
        if not known.all():
            dropped += int((~known).sum())
            if len(dropped_examples) < 5:
                unknown_ids = chunk.loc[~known, CANDIDATE_S1_COLUMN].astype(str)
                dropped_examples.extend(unknown_ids.head(5 - len(dropped_examples)).tolist())
            chunk = chunk.loc[known]
            owner_chunk = owner_chunk[known]
            true_chunk = true_chunk[known]
        if not len(chunk):
            continue

        take = min(len(chunk), rows_to_read - written)
        if take <= 0:
            break
        labels[written : written + take] = true_chunk[:take].astype(np.int8)
        owners[written : written + take] = owner_chunk[:take].astype(np.int32)
        source_values = chunk[SOURCE_IS_S2_COLUMN].to_numpy(dtype=np.float32)
        sources[written : written + take] = np.nan_to_num(source_values[:take], nan=0.0).astype(
            np.int8
        )
        if matrix is not None:
            matrix[written : written + take, :] = chunk[feature_columns].to_numpy(
                dtype=np.float32
            )[:take]
        # Candidate rows whose S1 entity has no true match at all: every one of them is
        # a false merge waiting to happen, which is why score_zero punishes them so
        # heavily.
        rows_for_empty_entities += int(np.count_nonzero(lengths_all[owner_chunk[:take]] == 0))
        written += take
        positives += int(true_chunk[:take].sum())

        if written and (written // (chunksize * 10)) != ((written - take) // (chunksize * 10)):
            log.info(
                "  %s rows labelled (%s positive, %.4f%%)",
                fmt_int(written),
                fmt_int(positives),
                100.0 * positives / max(written, 1),
            )
        if written >= rows_to_read:
            break

    elapsed = time.time() - started
    if matrix is not None:
        matrix.flush()
        # Release the memmap handle before touching either path: Windows refuses to
        # rename a file that is still mapped.
        del matrix
        gc.collect()
        if written != rows_to_read:
            # Rows were dropped, so the matrix has to be rewritten without them.
            # Keeping the hole would shift every row after it by one against the
            # labels - a silent mislabelling, which is the one outcome this function
            # must never produce.
            _compact_matrix(written, partial, matrix_path)
        else:
            partial.replace(matrix_path)

    if written != rows_to_read:
        # The row count came from this same file, so a short read means it changed
        # underneath us mid-run, or rows were dropped for unknown S1 ids.
        log.warning(
            "expected %s rows, labelled %s (%s dropped: S1 id absent from the ground truth)",
            fmt_int(rows_to_read),
            fmt_int(written),
            fmt_int(dropped),
        )
        if dropped_examples:
            log.warning("  examples of dropped S1 ids: %s", ", ".join(dropped_examples))

    # Sliced to `written` so the artifacts stay exactly row-aligned with each other
    # and with the matrix, whether or not anything was dropped.
    _save_array(out_dir / "val_labels.npy", labels[:written])
    _save_array(out_dir / "val_owner_index.npy", owners[:written])
    _save_array(out_dir / "val_source_is_s2.npy", sources[:written])

    entities_with_rows = np.unique(owners[owners >= 0])
    negatives = int(written) - positives
    report = {
        "features_path": str(features_path),
        "rows_in_file": int(total_rows),
        "rows_read": int(read),
        "rows_labelled": int(written),
        "rows_dropped_unknown_s1": int(dropped),
        "rows_dropped_unknown_s1_examples": dropped_examples,
        "positive_labels": int(positives),
        "negative_labels": negatives,
        "positive_rate": float(positives / max(written, 1)),
        "n_s1_entities_with_rows": int(len(entities_with_rows)),
        "n_s1_entities_in_ground_truth": int(ground_truth.n_entities),
        "n_ground_truth_pairs": int(len(true_pair_codes)),
        "n_features": len(feature_columns),
        "feature_columns": list(feature_columns),
        "rows_for_entities_with_no_true_match": int(rows_for_empty_entities),
        "matrix_written": bool(matrix_path),
        "matrix_path": str(matrix_path) if matrix_path else None,
        "matrix_bytes": int(written * len(feature_columns) * 4) if matrix_path else 0,
        "labels_path": str(out_dir / "val_labels.npy"),
        "owner_index_path": str(out_dir / "val_owner_index.npy"),
        "elapsed_seconds": round(elapsed, 1),
        "is_smoke": bool(sample_rows and sample_rows < total_rows),
    }
    log.info(
        "labels: %s positive / %s total (%.4f%%), %s S1 entities, %s dropped, %.0f rows/s",
        fmt_int(positives),
        fmt_int(written),
        100.0 * report["positive_rate"],
        fmt_int(len(entities_with_rows)),
        fmt_int(dropped),
        written / max(elapsed, 1e-9),
    )
    write_json(out_dir / "v1_labels_report.json", report)
    return report


def _read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.readline().rstrip("\n").split("\t")


def _compact_matrix(rows: int, partial: Path, final: Path, block: int = 2_000_000) -> None:
    """Rewrite ``partial`` keeping only its first ``rows`` rows, then replace ``final``.

    Only needed when rows were dropped for unknown S1 ids, which is the unusual path;
    it costs one pass over the disk at ~2M rows per block, with no extra RAM.

    This opens and closes ``partial`` itself rather than taking the caller's memmap:
    on Windows a memory-mapped file cannot be renamed while a handle is open on it, so
    the caller has to have released it first (``del matrix``) and this function has to
    release its own before the final rename.
    """
    compact_partial = Path(str(partial) + ".compact")
    source = np.load(partial, mmap_mode="r")
    n_features = source.shape[1]
    compact = np.lib.format.open_memmap(
        compact_partial, mode="w+", dtype="float32", shape=(rows, n_features)
    )
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        compact[start:stop] = source[start:stop]
    compact.flush()
    del compact, source
    gc.collect()
    compact_partial.replace(final)
    partial.unlink(missing_ok=True)


def _save_array(path: Path, array: np.ndarray) -> None:
    """Atomic ``.npy`` write - the temp-file + replace the repo uses elsewhere.

    ``np.save`` appends ``.npy`` to a path that lacks it, which would silently write
    ``<name>.npy.partial.npy`` and leave the rename below with nothing to move, so the
    handle is opened here instead and the suffix question never arises.
    """
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as handle:
        np.save(handle, array)
    partial.replace(path)


# ---------------------------------------------------------------------------
# Entity-grouped folds
# ---------------------------------------------------------------------------
def resolve_fold_mode(mode: str) -> str:
    """The fold mode that will actually be used, with ``auto`` resolved."""
    if mode not in FOLD_MODES:
        raise ValueError(f"unknown fold mode {mode!r}; expected one of {FOLD_MODES}")
    if mode == FOLD_MODE_AUTO:
        return FOLD_MODE_GROUPKFOLD if _sklearn_available() else FOLD_MODE_HASH
    return mode


def assign_folds(
    ground_truth: GroundTruth,
    n_folds: int = DEFAULT_FOLDS,
    mode: str = FOLD_MODE_AUTO,
    seed: int = DEFAULT_SEED,
    log: Optional[logging.Logger] = None,
    report: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Per-ground-truth-entity fold id in ``[0, n_folds)`` - grouped by S1 entity.

    GroupKFold's *semantics* are what matter, and both modes give them: all candidate
    rows of one S1 entity share a fold, so an out-of-fold prediction comes from a
    model that never saw that entity's name, address or competitor set. A row-level
    split would leak exactly that.

    ``groupkfold`` ``sklearn.model_selection.GroupKFold`` over one pseudo-row per true
                   match, so folds balance by sample count. This is the literal
                   ``GroupKFold(n_splits=5)`` the design calls for, and ``auto`` picks
                   it whenever scikit-learn is importable.
    ``hash``       fold = ``stable_hash64(entity_id) % n_folds``. A pure function of
                   the id - the same device ``assign_splits`` uses - so the group
                   invariant holds by construction and the assignment is reproducible
                   on any machine, without scikit-learn. This is what ``auto`` falls
                   back to when scikit-learn is absent, and it gives the same
                   entity-disjoint guarantee by a different route.

    ``report``, if given, receives the resolved mode and the per-fold entity counts.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_folds > MAX_FOLDS:
        # Fold ids have to survive the round trip through FOLD_DTYPE. Checked here
        # because an overflowing id reads as "unassigned" downstream and would be
        # re-placed by hash - splitting an entity across folds instead of failing.
        raise ValueError(f"n_folds must be <= {MAX_FOLDS}, got {n_folds}")
    resolved = resolve_fold_mode(mode)

    fold_of_entity: Optional[np.ndarray] = None
    if resolved == FOLD_MODE_GROUPKFOLD:
        try:
            fold_of_entity = _folds_groupkfold(ground_truth, n_folds)
        except ImportError:
            if mode != FOLD_MODE_AUTO:
                raise
            resolved = FOLD_MODE_HASH

    if fold_of_entity is None:
        fold_of_entity = _folds_from_hash(ground_truth.entity_ids, n_folds)

    missing = fold_of_entity < 0
    if missing.any():
        # GroupKFold only assigns entities that own at least one true match. An entity
        # with an empty ground-truth list still owns candidate rows and still needs a
        # fold, so the remainder is placed by the hash rule.
        fold_of_entity[missing] = _folds_from_hash(ground_truth.entity_ids[missing], n_folds)

    counts = np.bincount(fold_of_entity.astype(np.int64), minlength=n_folds)
    if len(counts) != n_folds or (counts == 0).any():
        raise ValueError(f"fold assignment left empty folds: {counts.tolist()} (n_folds={n_folds})")
    if log:
        log.info(
            "folds: %s (%s), entities per fold: %s",
            n_folds,
            resolved,
            ", ".join(fmt_int(int(c)) for c in counts),
        )
    if report is not None:
        report.update(
            {
                "mode": resolved,
                "requested_mode": mode,
                "entities_per_fold": counts.astype(int).tolist(),
            }
        )
    return fold_of_entity


def _folds_from_hash(entity_ids: np.ndarray, n_folds: int) -> np.ndarray:
    hashed = stable_hash64(entity_ids).astype(np.uint64) % np.uint64(n_folds)
    return hashed.astype(FOLD_DTYPE)


def _folds_groupkfold(ground_truth: GroundTruth, n_folds: int) -> np.ndarray:
    """``GroupKFold`` over one pseudo-row per true match, balancing by sample count."""
    try:
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:  # pragma: no cover - the caller falls back to hash
        raise ImportError(
            "fold mode 'groupkfold' needs scikit-learn: pip install scikit-learn"
        ) from exc

    lengths = ground_truth.lengths()
    n_entities = len(lengths)
    rows = np.repeat(np.arange(n_entities, dtype=np.int64), lengths)
    fold_of_entity = np.full(n_entities, -1, dtype=FOLD_DTYPE)
    if len(rows) < n_folds:
        raise ValueError(f"cannot make {n_folds} folds from {len(rows)} ground-truth pairs")
    splitter = GroupKFold(n_splits=n_folds)
    placeholder = np.zeros((len(rows), 1), dtype=np.uint8)
    for fold, (_, test_rows) in enumerate(splitter.split(placeholder, groups=rows)):
        fold_of_entity[rows[test_rows]] = fold
    return fold_of_entity


def _sklearn_available() -> bool:
    try:
        import sklearn.model_selection  # noqa: F401
    except ImportError:
        return False
    return True


def assert_fold_purity(
    row_folds: np.ndarray,
    owners: np.ndarray,
    n_folds: int,
    log: Optional[logging.Logger] = None,
) -> dict[str, int]:
    """Fail loudly if any S1 entity's rows span more than one fold.

    The construction (``row_folds = entity_folds[owner]``) makes this true by
    definition, which is exactly why it is worth asserting: if a future refactor ever
    computes folds per row, this is the check that catches the leak instead of a
    validation score that is quietly too good.

    **Only entities that own candidate rows are examined.** The ground truth holds
    every S1 entity in the competition - 2.2M of them in this dataset, including
    123,247 that match nothing - while a feature file is one split's worth of
    candidates and covers a fraction of that. An entity with no rows has nothing to
    keep together and cannot leak, but it does have an entry in the per-entity
    reduction, so counting it is a false positive. That is what this check used to do:
    ``low`` started at ``n_folds`` and ``high`` at ``-1``, so every *untouched*
    position read as "spans two folds", and a perfectly grouped assignment failed with:

        impure == (highest owner position + 1) - (entities owning rows)

    - a number that grows with the ground truth rather than with any leak. On the
    2M-row HPC shakedown it reported 2,194,111 "impure" entities while every entity
    sat in exactly one fold. ``high >= 0`` marks the entities that were actually
    reduced over, so nothing outside that set is counted.

    Returns the counts, so a run's log and metrics can state how much of the ground
    truth the check actually covered instead of implying all of it.
    """
    row_folds = np.asarray(row_folds)
    owners = np.asarray(owners)
    if row_folds.shape != owners.shape:
        raise ValueError(
            f"row_folds and owners must be row-aligned: {row_folds.shape} vs {owners.shape}"
        )
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    n_rows = int(owners.size)
    if not n_rows:
        return {"rows": 0, "entities_checked": 0, "entity_index_span": 0}

    # -1 is what label_pairs returns for an S1 id the ground truth does not contain.
    # Such a row must be dropped before this point, because a negative index wraps onto
    # a real entity in the two reductions below and would look like a leak.
    unknown = int(np.count_nonzero(owners < 0))
    if unknown:
        raise ValueError(
            f"{fmt_int(unknown)} rows have a negative owner index (an S1 id absent from "
            "the ground truth); they must be dropped before folds are checked"
        )
    # Every entity must have been assigned a fold, and a fold id must be in range: the
    # ``n_folds`` sentinel below is only distinguishable from a real id while this holds.
    if int(row_folds.min()) < 0 or int(row_folds.max()) >= n_folds:
        raise ValueError(
            f"fold ids must be in [0, {n_folds}), got [{int(row_folds.min())}, "
            f"{int(row_folds.max())}] - an out-of-range id means an entity was never assigned"
        )

    n_entities = int(owners.max()) + 1
    low = np.full(n_entities, n_folds, dtype=np.int32)
    high = np.full(n_entities, -1, dtype=np.int32)
    np.minimum.at(low, owners, row_folds)
    np.maximum.at(high, owners, row_folds)

    with_rows = high >= 0
    entities_with_rows = int(np.count_nonzero(with_rows))
    impure = int(np.count_nonzero(with_rows & (low != high)))
    counts = {
        "rows": n_rows,
        "entities_checked": entities_with_rows,
        # The highest owner position this run reaches, +1. Equal to the ground truth's
        # entity count only when its last position happens to own a row; the ground
        # truth's own size is logged by train() and recorded in the labels report.
        "entity_index_span": n_entities,
    }
    if impure:
        raise ValueError(
            f"{fmt_int(impure)} S1 entities have candidates in more than one fold - "
            "folds must be grouped by entity, never by pair "
            f"({fmt_int(entities_with_rows)} entities own candidate rows in this run, and "
            "those are the ones checked)"
        )
    if log:
        log.info(
            "entity-fold purity: %s entities own candidate rows in this run, and all %s of "
            "them sit in exactly one fold",
            fmt_int(entities_with_rows),
            fmt_int(entities_with_rows),
        )
    return counts


# ---------------------------------------------------------------------------
# The metric: always the evaluator's own implementation
# ---------------------------------------------------------------------------
def macro_f05(
    lengths: np.ndarray,
    candidates: np.ndarray,
    hits: np.ndarray,
    policy: str = ZERO_MATCH_POLICY,
) -> float:
    """The challenge metric, from ``CandidateEvaluation``'s implementation.

    Delegated rather than restated: the sweep must optimise exactly the number the
    evaluator reports, and a second implementation of the same formula is only a
    second thing to get wrong.
    """
    return float(
        CandidateEvaluation._macro_f05(
            np.asarray(lengths), np.asarray(candidates), np.asarray(hits), policy=policy
        )
    )


def entity_metrics(
    lengths: np.ndarray, candidates: np.ndarray, hits: np.ndarray
) -> dict[str, float]:
    """Per-entity precision/recall plus the pair-level view, for diagnostics only.

    The pair-level figures are easy to reason about and are explicitly **not** the
    objective: with beta=0.5 and a macro average over entities they disagree with the
    graded metric, sometimes by a lot. They are reported to explain *why* a threshold
    was chosen, never to choose it.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    hits = np.asarray(hits, dtype=np.float64)
    true_positive = float(hits.sum())
    false_positive = float((candidates - hits).sum())
    false_negative = float((lengths - hits).sum())

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    pair_f05 = _safe_div(1.25 * precision * recall, 0.25 * precision + recall)

    with_predictions = candidates > 0
    empty_gt = lengths == 0
    return {
        "true_positives": int(true_positive),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f05": pair_f05,
        "macro_precision_entity": float(
            (hits[with_predictions] / candidates[with_predictions]).mean()
        )
        if with_predictions.any()
        else 0.0,
        "macro_recall_entity": float(_macro_entity_recall(lengths, hits)),
        "n_entities_with_predictions": int(with_predictions.sum()),
        "n_empty_gt_entities": int(empty_gt.sum()),
        "n_empty_gt_entities_with_predictions": int((empty_gt & with_predictions).sum()),
        "n_matched_entities_with_no_prediction": int(((lengths > 0) & ~with_predictions).sum()),
        "max_predictions_per_s1": int(candidates.max()) if len(candidates) else 0,
        "mean_predictions_per_s1": float(candidates.mean()) if len(candidates) else 0.0,
        "mean_predictions_per_s1_with_predictions": float(
            candidates[with_predictions].mean()
        )
        if with_predictions.any()
        else 0.0,
    }


def per_source_diagnostics(
    ground_truth: GroundTruth,
    lengths: np.ndarray,
    owners: np.ndarray,
    is_true: np.ndarray,
    predicted: np.ndarray,
    source_is_s2: np.ndarray,
    entity_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Entity-level diagnostics split by target source (S2 / S3).

    The candidate file mixes S2 and S3 targets, and the two sources differ in how many
    candidates they generate, so one global threshold can be right for one and wrong
    for the other. The true-pair side is split from the packed ground-truth codes with
    ``code // ID_NUMERIC_MODULUS`` - the same recovery ``src/blocking.py`` uses - so no
    second pass over the ground-truth file is needed; the candidate side is split by
    the ``source_is_s2`` column the extractor already wrote.

    ``macro_f05_source_restricted`` is a **diagnostic, not the graded metric**: the
    challenge scores one F0.5 over all of an entity's targets, not one per source.
    """
    n_entities = ground_truth.n_entities
    lengths = np.asarray(lengths)
    owners = np.asarray(owners, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=bool)
    is_true = np.asarray(is_true, dtype=bool)
    entity_mask = np.asarray(entity_mask, dtype=bool)
    source_is_s2 = np.asarray(source_is_s2, dtype=bool)

    entity_of_pair = np.repeat(np.arange(n_entities, dtype=np.int64), lengths)
    pair_source = (ground_truth.codes // ID_NUMERIC_MODULUS).astype(np.int8)

    out: dict[str, dict[str, float]] = {}
    for label, code in (("S2", 2), ("S3", 3)):
        true_counts = np.bincount(entity_of_pair[pair_source == code], minlength=n_entities)
        rows = source_is_s2 if code == 2 else ~source_is_s2
        candidates = np.bincount(owners[predicted & rows], minlength=n_entities)[entity_mask]
        hits = np.bincount(
            owners[predicted & rows & is_true], minlength=n_entities
        )[entity_mask]
        true_lengths = true_counts[entity_mask]
        out[label] = {
            "n_true_pairs": int(true_lengths.sum()),
            "predictions": int(candidates.sum()),
            "true_positives": int(hits.sum()),
            "pair_precision": _safe_div(float(hits.sum()), float(candidates.sum())),
            "pair_recall": _safe_div(float(hits.sum()), float(true_lengths.sum())),
            "macro_recall_entity": float(_macro_entity_recall(true_lengths, hits)),
            "macro_f05_source_restricted": macro_f05(
                true_lengths, candidates, hits, ZERO_MATCH_POLICY
            ),
        }
    return out


def _safe_div(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def log_grid_positions(n_rows: int, points: int = DEFAULT_SWEEP_POINTS) -> np.ndarray:
    """Candidate prefixes to evaluate, log-spaced over how many pairs are predicted.

    A log grid rather than a linear one on the score: the interesting region is the
    high-precision tail, where a linear grid of 200 points would spend almost all of
    its resolution on thresholds that predict a third of the corpus. Position 0
    (predict nothing) is always included - it scores the empty-GT fraction, which is
    the number a real threshold has to beat.
    """
    if n_rows <= 1:
        return np.arange(max(n_rows, 1), dtype=np.int64)
    top = int(n_rows)
    grid = np.unique(
        np.round(np.logspace(0.0, np.log10(top), max(2, int(points)))).astype(np.int64)
    )
    grid = np.clip(grid, 1, top)
    return np.concatenate([[0], grid]).astype(np.int64)


def _snap_to_tie_boundaries(grid: np.ndarray, sorted_scores: np.ndarray) -> np.ndarray:
    """Move each rank position up to the first position realizable by ``score >= t``.

    The sweep counts rows by rank - "the top k" - but the rule applied to new data is
    ``p >= threshold``, and that also sweeps in every row sharing the k-th row's score.
    LightGBM scores tie in large groups (one leaf value per group: a 3,910-row smoke run
    had ten distinct scores with a 713-row tie group), so a rank prefix can describe a
    set of predictions no threshold can produce - and a "flat plateau" spanning several
    such points is an artifact of subdividing one tie group. Snapping each position to
    the end of its tie group makes every point on the curve exactly realizable, so the
    sweep, the plateau it selects, and the final re-measurement all describe the same
    predictions.

    ``sorted_scores`` must be descending and free of NaN.
    """
    zero = grid[grid == 0]
    base = grid[grid > 0]
    if not len(base):
        return np.unique(zero)
    if not len(sorted_scores):
        return np.unique(zero)
    thresholds = sorted_scores[base - 1]
    # sorted_scores descends, so its negation ascends and is searchable.
    snapped = np.searchsorted(-sorted_scores, -thresholds, side="right")
    return np.unique(np.concatenate([zero, snapped]))


def sweep_thresholds(
    probabilities: np.ndarray,
    is_true: np.ndarray,
    owners: np.ndarray,
    lengths: np.ndarray,
    entity_mask: np.ndarray,
    *,
    positions: Optional[Sequence[int]] = None,
    sweep_points: int = DEFAULT_SWEEP_POINTS,
    log: Optional[logging.Logger] = None,
) -> dict[str, np.ndarray]:
    """Macro F0.5 (and diagnostics) for every realizable threshold on the ranked pairs.

    Rows are sorted by score once, descending, so "predict every pair scoring at least
    t" is a prefix. Walking the grid in increasing prefix order means each row is
    counted exactly once in total and the per-entity counters only ever grow - so the
    whole sweep is one pass over the rows plus, per grid point, one reduction over the
    val entities.

    Every grid point is snapped to a tie boundary (see ``_snap_to_tie_boundaries``), so
    the score at each point is the score the deployed ``p >= threshold`` rule produces,
    and the reported ``predicted_rows`` is what that rule predicts. Rows whose score is
    NaN are never predicted by ``>=`` at any threshold, so they are excluded from the
    ranking entirely rather than being ranked below the real scores.

    On ``np.bincount(minlength=n_entities)`` here: ``evaluate_file`` explicitly warns
    against it, and is right to - there it runs once per *chunk*, so its 2.2M-wide
    zeroing dominates. Here it runs once per *grid point* (a few hundred calls total,
    each over only the delta rows since the previous point), which is the regime where
    it is the cheapest option rather than the most expensive. The comment there is
    about chunking, not about this.

    Returns arrays aligned to ``positions``. ``macro_f05_score_zero`` is the graded
    metric; ``macro_f05_exclude`` sits beside it because the gap between them is the
    cost of false merges on empty-GT entities.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    is_true = np.asarray(is_true, dtype=bool)
    owners = np.asarray(owners, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    entity_mask = np.asarray(entity_mask, dtype=bool)
    if not (len(probabilities) == len(is_true) == len(owners)):
        raise ValueError("probabilities, is_true and owners must be row-aligned")

    # NaN scores are never predicted by ``p >= t`` for any t, so they are ranked out
    # entirely: ranking them below the real scores would put them in prefixes that no
    # threshold can select.
    valid = ~np.isnan(probabilities)
    ranked = np.flatnonzero(valid)
    order = ranked[np.argsort(-probabilities[valid], kind="stable")]
    sorted_scores = probabilities[order]
    sorted_owners = owners[order]
    true_positions = np.flatnonzero(is_true[order])

    grid = (
        np.unique(np.asarray(positions, dtype=np.int64))
        if positions is not None
        else log_grid_positions(len(order), sweep_points)
    )
    grid = grid[(grid >= 0) & (grid <= len(order))]
    grid = _snap_to_tie_boundaries(grid, sorted_scores)

    n_entities = len(lengths)
    counts = np.zeros(n_entities, dtype=np.int64)
    hits = np.zeros(n_entities, dtype=np.int64)
    cursor = 0
    true_cursor = 0

    collected: dict[str, list[Any]] = {
        name: []
        for name in (
            "positions",
            "thresholds",
            "predicted_rows",
            "macro_f05_score_zero",
            "macro_f05_exclude",
            "pair_precision",
            "pair_recall",
            "pair_f05",
            "macro_precision_entity",
            "macro_recall_entity",
            "mean_predictions_per_s1",
            "n_empty_gt_entities_with_predictions",
            "n_matched_entities_with_no_prediction",
        )
    }

    for target in grid:
        target = int(target)
        if target > cursor:
            counts += np.bincount(sorted_owners[cursor:target], minlength=n_entities)
            end = int(np.searchsorted(true_positions, target))
            if end > true_cursor:
                hits += np.bincount(
                    sorted_owners[true_positions[true_cursor:end]], minlength=n_entities
                )
                true_cursor = end
            cursor = target

        threshold = float(sorted_scores[cursor - 1]) if cursor > 0 else float("inf")
        val_lengths = lengths[entity_mask]
        val_candidates = counts[entity_mask]
        val_hits = hits[entity_mask]
        diagnostics = entity_metrics(val_lengths, val_candidates, val_hits)

        collected["positions"].append(cursor)
        collected["thresholds"].append(threshold)
        collected["predicted_rows"].append(int(val_candidates.sum()))
        collected["macro_f05_score_zero"].append(
            macro_f05(val_lengths, val_candidates, val_hits, ZERO_MATCH_POLICY)
        )
        collected["macro_f05_exclude"].append(
            macro_f05(val_lengths, val_candidates, val_hits, OTHER_POLICY)
        )
        collected["pair_precision"].append(diagnostics["pair_precision"])
        collected["pair_recall"].append(diagnostics["pair_recall"])
        collected["pair_f05"].append(diagnostics["pair_f05"])
        collected["macro_precision_entity"].append(diagnostics["macro_precision_entity"])
        collected["macro_recall_entity"].append(diagnostics["macro_recall_entity"])
        collected["mean_predictions_per_s1"].append(diagnostics["mean_predictions_per_s1"])
        collected["n_empty_gt_entities_with_predictions"].append(
            diagnostics["n_empty_gt_entities_with_predictions"]
        )
        collected["n_matched_entities_with_no_prediction"].append(
            diagnostics["n_matched_entities_with_no_prediction"]
        )

    if log:
        log.info("swept %s operating points", fmt_int(len(grid)))
    return {name: np.asarray(values) for name, values in collected.items()}


def sweep_frame(sweep: dict[str, np.ndarray]) -> pd.DataFrame:
    """The sweep as a table, for the CSV saved next to the metrics."""
    return pd.DataFrame({name: values for name, values in sweep.items()})


def select_operating_point(
    sweep: dict[str, np.ndarray],
    tolerance: Optional[float] = None,
    policy: str = ZERO_MATCH_POLICY,
    log: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Pick a threshold from the middle of the best plateau, not the best spike.

    The curve is flat over a wide range of thresholds; the argmax of a 200-point grid
    is one point on that plateau and moves with the grid. Taking the midpoint of the
    contiguous region within ``tolerance`` of the best score gives a threshold a small
    change in the score surface cannot move much - which matters, because the test
    split is a different sample of entities.

    Between two points that score the same, the midpoint is preferred to either edge:
    the high-threshold edge is needlessly conservative, and the low-threshold edge
    spends precision the metric charges four times for.
    """
    scores = np.asarray(sweep[_policy_column(policy)])
    if not len(scores):
        raise ValueError("empty sweep")
    best = int(np.argmax(scores))
    slack = DEFAULT_PLATEAU_TOLERANCE if tolerance is None else float(tolerance)
    within = scores >= scores[best] - slack
    low = best
    while low - 1 >= 0 and within[low - 1]:
        low -= 1
    high = best
    while high + 1 < len(scores) and within[high + 1]:
        high += 1
    chosen = (low + high) // 2

    choice = {
        "policy": policy,
        "tolerance": slack,
        "chosen_index": int(chosen),
        "best_index": int(best),
        "plateau_start_index": int(low),
        "plateau_end_index": int(high),
        "plateau_points": int(high - low + 1),
        "plateau_is_single_point": bool(high == low),
        "chosen_threshold": float(sweep["thresholds"][chosen]),
        "chosen_macro_f05": float(scores[chosen]),
        "best_macro_f05": float(scores[best]),
        "chosen_predicted_rows": int(sweep["predicted_rows"][chosen]),
        "plateau_low_threshold": float(sweep["thresholds"][low]),
        "plateau_high_threshold": float(sweep["thresholds"][high]),
    }
    if log:
        log.info(
            "plateau: %s points within %.4f of the best (%.4f), thresholds %.6g..%.6g -> chosen %.6g",
            fmt_int(choice["plateau_points"]),
            slack,
            choice["best_macro_f05"],
            choice["plateau_high_threshold"],
            choice["plateau_low_threshold"],
            choice["chosen_threshold"],
        )
        if choice["plateau_is_single_point"]:
            log.warning(
                "the best operating point is a single grid point - the curve around it "
                "is not flat; prefer a wider --plateau-tolerance before trusting it"
            )
    return choice


def _policy_column(policy: str) -> str:
    if policy == ZERO_MATCH_POLICY:
        return "macro_f05_score_zero"
    if policy == OTHER_POLICY:
        return "macro_f05_exclude"
    raise ValueError(f"unknown policy {policy!r}; expected {ZERO_MATCH_POLICY!r} or {OTHER_POLICY!r}")


def tune_threshold(
    config: dict,
    sweep: dict[str, np.ndarray],
    *,
    tolerance: Optional[float] = None,
    policy: str = ZERO_MATCH_POLICY,
    log: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Choose the decision threshold by macro F0.5 on out-of-fold val scores.

    Optimising the *macro per-entity* F0.5 - not accuracy, not pair-level F1, not
    ROC-AUC - is the point: those do not agree on this dataset. Pair-level accuracy
    would happily accept a threshold that merges thousands of empty-GT entities,
    because those entities own so few of the 67.3M candidate rows.
    """
    choice = select_operating_point(sweep, tolerance=tolerance, policy=policy, log=log)
    choice["zero_match_policy_configured"] = config.get("evaluation", {}).get("zero_match_policy")
    return choice


def evaluate_at_threshold(
    threshold: float,
    probabilities: np.ndarray,
    is_true: np.ndarray,
    owners: np.ndarray,
    lengths: np.ndarray,
    entity_mask: np.ndarray,
) -> dict[str, Any]:
    """Exact per-entity accounting for one threshold, with ties resolved by ``>=``.

    The sweep identifies the operating point by *rank*; this re-measures it with the
    rule that will actually be applied to the test data, so a score shared by many rows
    cannot make the reported F0.5 describe a slightly different set of predictions than
    the threshold produces.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    is_true = np.asarray(is_true, dtype=bool)
    owners = np.asarray(owners, dtype=np.int64)
    entity_mask = np.asarray(entity_mask, dtype=bool)
    predicted = probabilities >= np.float32(threshold)
    n_entities = len(lengths)
    counts = np.bincount(owners[predicted], minlength=n_entities)
    hits = np.bincount(owners[predicted & is_true], minlength=n_entities)

    val_lengths = lengths[entity_mask]
    val_candidates = counts[entity_mask]
    val_hits = hits[entity_mask]
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "predicted_pairs": int(val_candidates.sum()),
        "macro_f05_score_zero": macro_f05(val_lengths, val_candidates, val_hits, ZERO_MATCH_POLICY),
        "macro_f05_exclude": macro_f05(val_lengths, val_candidates, val_hits, OTHER_POLICY),
    }
    out.update(entity_metrics(val_lengths, val_candidates, val_hits))
    return out


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    """Everything needed to score new candidate features and to explain the run."""

    model: str = MODEL_LIGHTGBM
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    params: dict[str, Any] = field(default_factory=dict)
    folds: int = DEFAULT_FOLDS
    fold_mode: str = FOLD_MODE_HASH
    threshold: float = float("nan")
    score_feature: str = "name_token_set_ratio"
    boosters: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    out_dir: Optional[Path] = None

    def is_ready(self) -> bool:
        """True when the bundle can score pairs."""
        if self.model == MODEL_THRESHOLD:
            return True
        return bool(self.boosters)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_features": len(self.feature_columns),
            "n_boosters": len(self.boosters),
            "folds": self.folds,
            "fold_mode": self.fold_mode,
            "threshold": self.threshold,
            "score_feature": self.score_feature,
            "params": self.params,
            "out_dir": str(self.out_dir) if self.out_dir else None,
        }


def _import_lightgbm():
    try:
        import lightgbm
    except ImportError as exc:
        raise ImportError(LIGHTGBM_MISSING_MESSAGE) from exc
    return lightgbm


def default_params(seed: int = DEFAULT_SEED, num_threads: int = 0) -> dict[str, Any]:
    """Conservative, reproducible LightGBM parameters.

    Deliberately no ``feature_fraction`` < 1 and no bagging: LightGBM's
    ``deterministic=True`` is incompatible with subsampling, and with 27 features and
    tens of millions of rows the subsampling buys nothing anyway. What is left is
    exactly reproducible for a fixed seed and thread count, and preserves the natural
    class distribution - no ``scale_pos_weight``, no resampling.
    """
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "num_threads": int(num_threads) if num_threads else 0,
        "seed": int(seed),
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------
def train(
    config: dict,
    features_path: str | Path,
    out_dir: str | Path,
    *,
    ground_truth: Optional[GroundTruth] = None,
    ground_truth_path: Optional[str | Path] = None,
    model: str = MODEL_LIGHTGBM,
    score_feature: str = "name_token_set_ratio",
    folds: int = DEFAULT_FOLDS,
    fold_mode: str = FOLD_MODE_AUTO,
    seed: int = DEFAULT_SEED,
    chunksize: int = DEFAULT_CHUNKSIZE,
    sample_rows: Optional[int] = None,
    write_matrix: bool = True,
    n_estimators: int = 300,
    params: Optional[dict[str, Any]] = None,
    num_threads: int = 0,
    predict_chunk: int = DEFAULT_PREDICT_CHUNK,
    sweep_points: int = DEFAULT_SWEEP_POINTS,
    refine_points: int = DEFAULT_REFINE_POINTS,
    plateau_tolerance: Optional[float] = None,
    policy: str = ZERO_MATCH_POLICY,
    final_model: str = "ensemble",
    log: Optional[logging.Logger] = None,
) -> ModelBundle:
    """Label, fold, train out-of-fold, sweep the threshold, and report.

    The order matters and is not incidental: the out-of-fold probabilities are produced
    by the per-fold models *before* the optional final retrain, so the threshold sweep
    and every reported number come from scores no model was fit on.

    Args:
        config: loaded config (used for the val split mask).
        features_path: ``features.tsv`` from ``scripts/extract_pair_features.py``.
        out_dir: every artifact of the run lands here.
        ground_truth: pre-loaded ground truth, or ``ground_truth_path``, or config.
        model: ``lightgbm`` (V1) or ``threshold`` (dependency-free baseline arm).
        final_model: ``ensemble`` averages the per-fold models at predict time;
            ``retrain`` fits one more model on every row. Both are legitimate; the
            ensemble is free, needs no second pass over the data, and is the default.

    Returns:
        A :class:`ModelBundle` carrying the fitted models, the chosen threshold and the
        full metrics dict.
    """
    log = log or logger
    out_dir = ensure_dir(out_dir)
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    if final_model not in ("ensemble", "retrain"):
        raise ValueError(
            f"unknown final_model {final_model!r}; expected 'ensemble' or 'retrain'"
        )

    if ground_truth is None:
        ground_truth = load_ground_truth(config, path=ground_truth_path, log=log)
    lengths_all = ground_truth.lengths()
    entity_mask = split_mask_for(ground_truth, config, "val")
    n_entities = ground_truth.n_entities
    log.info(
        "ground truth: %s entities (%s val), %s true pairs",
        fmt_int(n_entities),
        fmt_int(int(entity_mask.sum())),
        fmt_int(int(lengths_all.sum())),
    )

    # -- 1. labels + matrix -------------------------------------------------
    if not write_matrix:
        raise ValueError(
            "write_matrix=False labels the file without the feature matrix, and every "
            "model here trains from it - so there is nothing to train on. Use "
            "build_label_artifacts() directly for a labels-only pass."
        )
    label_report = build_label_artifacts(
        features_path,
        ground_truth,
        out_dir,
        chunksize=chunksize,
        sample_rows=sample_rows,
        write_matrix=write_matrix,
        log=log,
    )
    feature_columns = tuple(label_report["feature_columns"])
    labels = np.load(out_dir / "val_labels.npy")
    owners = np.load(out_dir / "val_owner_index.npy").astype(np.int64)
    source_is_s2 = np.load(out_dir / "val_source_is_s2.npy").astype(bool)
    n_rows = len(labels)
    is_true = labels.astype(bool)

    # The feature file normally covers every labelled entity (extract_pair_features
    # --population all): every row trains out-of-fold, so every entity's score comes
    # from a model that never saw it, and the threshold is tuned and reported on the
    # val entities only. `sampled_mask` narrows that to the entities this file actually has
    # rows for, and is reported separately: an entity the blocker proposed nothing for
    # is a blocking miss, not a matcher miss.
    covered_mask = np.zeros(n_entities, dtype=bool)
    covered_mask[np.unique(owners)] = True
    sampled_mask = entity_mask & covered_mask
    log.info(
        "val entities: %s in the split mask, %s of them covered by the feature file",
        fmt_int(int(entity_mask.sum())),
        fmt_int(int(sampled_mask.sum())),
    )

    # -- 2. folds -----------------------------------------------------------
    fold_report: dict[str, Any] = {}
    entity_folds = assign_folds(
        ground_truth, n_folds=folds, mode=fold_mode, seed=seed, log=log, report=fold_report
    )
    row_folds = entity_folds[owners]
    # The check covers the entities this run has candidates for, not the whole ground
    # truth - see assert_fold_purity for why counting the rest is a false positive.
    purity = assert_fold_purity(row_folds, owners, folds, log=log)

    # -- 3. out-of-fold scores ---------------------------------------------
    probabilities = np.full(n_rows, -1.0, dtype=np.float32)
    per_fold: list[dict[str, Any]] = []
    resolved_params: dict[str, Any] = {}
    boosters: list[Any] = []
    matrix = None

    if model == MODEL_THRESHOLD:
        probabilities = _threshold_scores(
            out_dir, feature_columns, score_feature, n_rows, write_matrix
        )
        log.info(
            "model=threshold on %r: nothing is fitted, so every score is out-of-sample "
            "by construction and no folds are trained",
            score_feature,
        )
    else:
        lightgbm = _import_lightgbm()
        resolved_params = (
            dict(params) if params else default_params(seed=seed, num_threads=num_threads)
        )
        matrix, boosters = _fit_oof(
            lightgbm,
            out_dir,
            feature_columns,
            is_true,
            row_folds,
            folds,
            resolved_params,
            n_estimators,
            predict_chunk,
            probabilities,
            per_fold,
            log,
        )
        if final_model == "retrain":
            # After the out-of-fold pass, never before: the model below has seen every
            # row, so it must not be the source of the out-of-fold scores.
            log.info("retraining the final model on every row (final_model=retrain)")
            started = time.time()
            full = lightgbm.Dataset(
                np.asarray(matrix),
                label=is_true.astype(np.float32),
                feature_name=list(feature_columns),
                free_raw_data=True,
            )
            boosters = [lightgbm.train(resolved_params, full, num_boost_round=int(n_estimators))]
            del full
            log.info("retrained in %.1f s", time.time() - started)

    unfilled = int(np.count_nonzero(probabilities < 0))
    if unfilled:
        raise RuntimeError(
            f"{fmt_int(unfilled)} rows have no out-of-fold score - every row must be "
            "predicted by a model that did not train on its entity"
        )

    # -- 4. sweep + operating point ----------------------------------------
    sweep = sweep_thresholds(
        probabilities, is_true, owners, lengths_all, entity_mask, sweep_points=sweep_points, log=log
    )
    choice = tune_threshold(config, sweep, tolerance=plateau_tolerance, policy=policy, log=log)
    sweep = _refine_around(
        sweep, probabilities, is_true, owners, lengths_all, entity_mask, choice, refine_points
    )
    choice = tune_threshold(config, sweep, tolerance=plateau_tolerance, policy=policy, log=log)
    threshold = float(choice["chosen_threshold"])

    configured_policy = (config.get("evaluation", {}) or {}).get("zero_match_policy")
    if configured_policy and configured_policy != policy:
        # Surfaced because the two metrics reward different behaviour on empty-GT
        # entities, so a log reader seeing "exclude" in the config should know which
        # number the threshold was chosen for.
        log.info(
            "tuning under policy=%r although the config sets zero_match_policy=%r: "
            "score_zero is the challenge's own rule (the graded "
            "f05_accept_all_macro_score_zero), and both are reported below",
            policy,
            configured_policy,
        )

    # Re-measure with the rule that will be applied to the test data.
    predicted = probabilities >= np.float32(threshold)
    final = evaluate_at_threshold(
        threshold, probabilities, is_true, owners, lengths_all, entity_mask
    )

    # The sweep is what chose this threshold, so its value at the chosen point and this
    # re-measurement must agree - they are the same accounting reached two different
    # ways. They can only diverge if the grid stops landing on tie boundaries, which is
    # the bug that made a rank-prefix plateau describe predictions the deployed rule
    # could not produce. Checked rather than assumed, and reported in the metrics.
    sweep_macro = float(sweep["macro_f05_score_zero"][choice["chosen_index"]])
    sweep_rows = int(sweep["predicted_rows"][choice["chosen_index"]])
    consistency = {
        "sweep_macro_f05_score_zero": sweep_macro,
        "measured_macro_f05_score_zero": float(final["macro_f05_score_zero"]),
        "sweep_predicted_rows": sweep_rows,
        "measured_predicted_pairs": int(final["predicted_pairs"]),
    }
    consistency["macro_f05_agrees"] = abs(sweep_macro - final["macro_f05_score_zero"]) < 1e-9
    consistency["predicted_rows_agree"] = sweep_rows == int(final["predicted_pairs"])
    if not (consistency["macro_f05_agrees"] and consistency["predicted_rows_agree"]):
        log.warning(
            "the sweep and the deployed rule disagree at the chosen threshold: sweep "
            "%.6f on %s rows, measured %.6f on %s rows - the threshold is being selected "
            "from a curve the deployed rule does not have",
            sweep_macro,
            fmt_int(sweep_rows),
            final["macro_f05_score_zero"],
            fmt_int(final["predicted_pairs"]),
        )

    final_sampled = evaluate_at_threshold(
        threshold, probabilities, is_true, owners, lengths_all, sampled_mask
    )
    final["per_source"] = per_source_diagnostics(
        ground_truth, lengths_all, owners, is_true, predicted, source_is_s2, entity_mask
    )
    final["per_source_covered_entities_only"] = per_source_diagnostics(
        ground_truth, lengths_all, owners, is_true, predicted, source_is_s2, sampled_mask
    )

    # -- 5. baselines, so the chosen threshold has something to beat --------
    all_counts = np.bincount(owners, minlength=n_entities)
    all_hits = np.bincount(owners[is_true], minlength=n_entities)
    baselines = {
        "predict_nothing_macro_f05_score_zero": macro_f05(
            lengths_all[entity_mask],
            np.zeros_like(all_counts[entity_mask]),
            np.zeros_like(all_hits[entity_mask]),
            ZERO_MATCH_POLICY,
        ),
        "accept_every_candidate_macro_f05_score_zero": macro_f05(
            lengths_all[entity_mask], all_counts[entity_mask], all_hits[entity_mask], ZERO_MATCH_POLICY
        ),
        "accept_every_candidate_macro_f05_exclude": macro_f05(
            lengths_all[entity_mask], all_counts[entity_mask], all_hits[entity_mask], OTHER_POLICY
        ),
        "candidate_precision_of_the_feature_file": _safe_div(
            float(all_hits[entity_mask].sum()), float(all_counts[entity_mask].sum())
        ),
        "candidate_pair_recall_of_the_feature_file": _safe_div(
            float(all_hits[entity_mask].sum()), float(lengths_all[entity_mask].sum())
        ),
    }

    # -- 6. persist ---------------------------------------------------------
    metrics: dict[str, Any] = {
        "out_dir": str(out_dir),
        "model": model,
        "final_model": final_model if model == MODEL_LIGHTGBM else None,
        "score_feature": score_feature if model == MODEL_THRESHOLD else None,
        "features_path": str(features_path),
        "rows": int(n_rows),
        "positive_labels": int(is_true.sum()),
        "negative_labels": int(n_rows - is_true.sum()),
        "positive_rate": float(is_true.mean()) if n_rows else 0.0,
        "n_s1_entities_with_rows": int(len(np.unique(owners))),
        "n_s1_entities_in_val_split": int(entity_mask.sum()),
        "n_s1_entities_covered_by_features": int(sampled_mask.sum()),
        "n_folds": int(folds),
        "fold_mode": fold_report.get("mode", fold_mode),
        "fold_mode_requested": fold_report.get("requested_mode", fold_mode),
        "fold_entities": fold_report.get("entities_per_fold", []),
        "fold_rows": _counts_per_fold(row_folds, folds),
        "fold_positives": [int(is_true[row_folds == k].sum()) for k in range(folds)],
        "fold_purity": purity,
        "params": _jsonable(resolved_params),
        "n_estimators": int(n_estimators) if model == MODEL_LIGHTGBM else 0,
        "per_fold": per_fold,
        "threshold": threshold,
        "operating_point": choice,
        "operating_point_consistency": consistency,
        "val_metrics": final,
        "val_metrics_covered_entities_only": final_sampled,
        "baselines": baselines,
        "labels": {k: v for k, v in label_report.items()},
        "feature_columns": list(feature_columns),
        "is_smoke": bool(label_report.get("is_smoke")),
        "sweep_points": int(len(sweep["positions"])),
    }
    write_json(out_dir / "v1_metrics.json", metrics)
    sweep_frame(sweep).to_csv(out_dir / "threshold_sweep.csv", index=False)
    _save_array(out_dir / "oof_probabilities.npy", probabilities)

    bundle = ModelBundle(
        model=model,
        feature_columns=feature_columns,
        params=resolved_params,
        folds=int(folds),
        fold_mode=str(fold_report.get("mode", fold_mode)),
        threshold=threshold,
        score_feature=score_feature,
        boosters=boosters,
        metrics=metrics,
        out_dir=out_dir,
    )
    save_bundle(bundle, out_dir)
    _log_summary(log, metrics)
    return bundle


def _counts_per_fold(values: np.ndarray, folds: int) -> list[int]:
    return [int(np.count_nonzero(values == k)) for k in range(folds)]


def _jsonable(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value
            if isinstance(value, (int, float, str, bool)) or value is None
            else str(value)
        )
        for key, value in mapping.items()
    }


def _threshold_scores(
    out_dir: Path,
    feature_columns: Sequence[str],
    score_feature: str,
    n_rows: int,
    write_matrix: bool,
) -> np.ndarray:
    """The dependency-free arm: the raw value of one feature is the score.

    Not clipped to ``[0, 1]``: several features are raw counts (``token_df``,
    ``name_token_count_diff``), and clipping them would silently rewrite the score. The
    sweep only reads the *rank order*, so a monotone rescaling changes nothing, while
    the reported threshold stays in the feature's own units.

    The feature has to point the right way - higher meaning more likely a match - which
    the unit-interval similarity features all do. ``name_token_count_diff`` does not,
    and using it would produce a visibly inverted curve; the ``predict nothing`` and
    ``accept every candidate`` baselines printed beside it make that obvious rather
    than subtle.
    """
    if not write_matrix:
        raise ValueError(
            "--model threshold reads its score from the matrix; do not pass --no-matrix"
        )
    if score_feature not in feature_columns:
        raise ValueError(
            f"score feature {score_feature!r} is not one of the {len(feature_columns)} "
            f"features: {list(feature_columns)}"
        )
    column = list(feature_columns).index(score_feature)
    matrix = np.load(out_dir / "val_features.npy", mmap_mode="r")
    scores = np.empty(n_rows, dtype=np.float32)
    step = 10_000_000
    for start in range(0, n_rows, step):
        block = np.asarray(matrix[start : start + step, column], dtype=np.float32)
        # A missing value is not evidence of a match, so it scores at the bottom
        # rather than at the middle.
        scores[start : start + step] = np.nan_to_num(block, nan=0.0)
    del matrix
    return scores


def _fit_oof(
    lightgbm,
    out_dir: Path,
    feature_columns: Sequence[str],
    is_true: np.ndarray,
    row_folds: np.ndarray,
    folds: int,
    params: dict[str, Any],
    n_estimators: int,
    predict_chunk: int,
    probabilities: np.ndarray,
    per_fold: list[dict[str, Any]],
    log: logging.Logger,
):
    """Train one model per fold and fill ``probabilities`` out-of-fold.

    Returns ``(matrix, boosters)``: the memory-mapped matrix and the per-fold models.
    Each fold's training copy is released before the next fold starts, so peak RAM is
    one training split plus LightGBM's own binning, not five of them.
    """
    matrix = np.load(out_dir / "val_features.npy", mmap_mode="r")
    boosters = []
    label = is_true.astype(np.float32)

    for fold in range(folds):
        trained_on = row_folds != fold
        held_out = ~trained_on
        n_train = int(trained_on.sum())
        n_val = int(held_out.sum())
        positives_train = int(label[trained_on].sum())
        if n_train == 0 or n_val == 0 or positives_train == 0:
            raise ValueError(
                f"fold {fold}: {fmt_int(n_train)} train / {fmt_int(n_val)} held-out rows, "
                f"{fmt_int(positives_train)} positives in train - not trainable"
            )
        log.info(
            "fold %d/%d: %s train rows (%s positive, %.4f%%), %s held-out rows",
            fold + 1,
            folds,
            fmt_int(n_train),
            fmt_int(positives_train),
            100.0 * positives_train / n_train,
            fmt_int(n_val),
        )

        x_train = np.asarray(matrix[trained_on])
        dataset = lightgbm.Dataset(
            x_train, label=label[trained_on], feature_name=list(feature_columns), free_raw_data=True
        )
        started = time.time()
        booster = lightgbm.train(params, dataset, num_boost_round=int(n_estimators))
        train_seconds = time.time() - started
        del x_train, dataset

        started = time.time()
        held_out_rows = np.flatnonzero(held_out)
        for start in range(0, len(held_out_rows), predict_chunk):
            block = held_out_rows[start : start + predict_chunk]
            probabilities[block] = booster.predict(
                np.asarray(matrix[block]), num_threads=params.get("num_threads") or 0
            )
        predict_seconds = time.time() - started
        del held_out_rows

        log.info(
            "fold %d: trained in %.1f s, scored %s rows in %.1f s",
            fold + 1,
            train_seconds,
            fmt_int(n_val),
            predict_seconds,
        )
        per_fold.append(
            {
                "fold": fold,
                "train_rows": n_train,
                "held_out_rows": n_val,
                "train_positives": positives_train,
                "train_seconds": round(train_seconds, 1),
                "predict_seconds": round(predict_seconds, 1),
                "n_trees": int(booster.num_trees()),
            }
        )
        boosters.append(booster)
    return matrix, boosters


def _refine_around(
    sweep: dict[str, np.ndarray],
    probabilities: np.ndarray,
    is_true: np.ndarray,
    owners: np.ndarray,
    lengths: np.ndarray,
    entity_mask: np.ndarray,
    choice: dict[str, Any],
    refine_points: int,
) -> dict[str, np.ndarray]:
    """Re-sweep between the neighbours of the coarse optimum and merge the curves.

    The coarse grid is log-spaced, so near the optimum its spacing can be wider than
    the plateau itself. Refining there is what makes "the middle of the plateau" mean
    something, rather than the middle of two arbitrary grid points.

    Duplicate positions across the two grids are dropped rather than kept twice: a
    prefix gives the same counts however it was reached, so the merged curve would
    otherwise carry repeated rows that inflate ``plateau_points`` without adding
    resolution.
    """
    if refine_points <= 0 or len(sweep["positions"]) < 2:
        return sweep
    index = int(choice["best_index"])
    low = int(sweep["positions"][max(index - 1, 0)])
    high = int(sweep["positions"][min(index + 1, len(sweep["positions"]) - 1)])
    if high - low <= 1:
        return sweep
    extra = np.linspace(low, high, int(refine_points) + 2).astype(np.int64)
    refined = sweep_thresholds(
        probabilities, is_true, owners, lengths, entity_mask, positions=extra
    )
    merged = {name: np.concatenate([sweep[name], refined[name]]) for name in sweep}
    order = np.argsort(merged["positions"], kind="stable")
    merged = {name: values[order] for name, values in merged.items()}
    _, first = np.unique(merged["positions"], return_index=True)
    first.sort()
    return {name: values[first] for name, values in merged.items()}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(
    config: dict,
    bundle: ModelBundle,
    features: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """P(match) for a matrix of candidate features.

    Accepts a DataFrame of feature columns (the shape the extractor's chunks arrive in)
    or a pre-built ``float32`` matrix in ``bundle.feature_columns`` order. An ensemble
    of fold models is averaged - the fold models have seen the same data five different
    ways, and averaging them costs one extra forward pass each.
    """
    if not bundle.is_ready():
        raise ValueError("bundle has no fitted model - call train() first")
    matrix = _as_matrix(bundle, features)
    if bundle.model == MODEL_THRESHOLD:
        column = list(bundle.feature_columns).index(bundle.score_feature)
        return np.nan_to_num(matrix[:, column], nan=0.0).astype(np.float32)
    total = np.zeros(len(matrix), dtype=np.float64)
    for booster in bundle.boosters:
        total += booster.predict(matrix)
    if bundle.boosters:
        total /= len(bundle.boosters)
    return total.astype(np.float32)


def decide(config: dict, bundle: ModelBundle, probabilities: np.ndarray) -> np.ndarray:
    """Pair-level decisions at the bundle's threshold.

    One-to-many by design: there is no top-1 step anywhere in this module. Every pair
    scoring at or above the threshold is a predicted match, so an S1 with three
    surviving candidates keeps all three.
    """
    if not np.isfinite(bundle.threshold):
        raise ValueError("bundle has no threshold - run train() first")
    return np.asarray(probabilities) >= bundle.threshold


def aggregate_matches(
    s1_ids: Sequence[str],
    target_ids: Sequence[str],
    decisions: np.ndarray,
    s1_universe: Optional[Sequence[str]] = None,
) -> dict[str, list[str]]:
    """Group predicted pairs into the submission's shape: one row per S1 entity.

    Kept next to the decision rule because the graded artifact
    (``matching_results.tsv``, columns ``source1_entity_id`` and a comma-separated
    ``matched_entity_ids``) is exactly this aggregation. It is *not* wired into a
    submission script yet: the test features do not exist, and ``scripts/predict.py``
    documents why the format is still unconfirmed.

    Every S1 entity that appears in the rows gets an entry, including the ones whose
    candidates all fell below the threshold - they map to an empty list, which is a
    required output row ("empty list for a singleton"), not an omission. An entity that
    has no candidate rows at all cannot appear in ``s1_ids``; pass those in
    ``s1_universe`` so the submission has one row per test entity.

    Target ids are deduplicated and sorted so reruns are byte-identical.
    """
    decisions = np.asarray(decisions, dtype=bool)
    if not (len(s1_ids) == len(target_ids) == len(decisions)):
        raise ValueError("s1_ids, target_ids and decisions must be row-aligned")
    grouped: dict[str, set[str]] = {str(s1): set() for s1 in s1_ids}
    for s1, target, keep in zip(s1_ids, target_ids, decisions):
        if keep:
            grouped[str(s1)].add(str(target))
    if s1_universe is not None:
        for s1 in s1_universe:
            grouped.setdefault(str(s1), set())
    return {s1: sorted(targets) for s1, targets in grouped.items()}


def _as_matrix(bundle: ModelBundle, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Coerce either input shape into the ``float32`` matrix the models expect."""
    if isinstance(features, pd.DataFrame):
        missing = [c for c in bundle.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"feature frame is missing columns: {missing}")
        # Column ORDER comes from the bundle, never from the frame: a reordered frame
        # would otherwise feed every model the wrong feature.
        return features[list(bundle.feature_columns)].to_numpy(dtype=np.float32)
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[1] != len(bundle.feature_columns):
        raise ValueError(
            f"expected a matrix with {len(bundle.feature_columns)} columns, got shape {matrix.shape}"
        )
    return matrix.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_bundle(bundle: ModelBundle, out_dir: str | Path) -> Path:
    """Write the fitted models and their metadata next to the run's other artifacts."""
    out_dir = ensure_dir(out_dir)
    model_dir = ensure_dir(Path(out_dir) / "model")
    names = []
    for index, booster in enumerate(bundle.boosters):
        name = f"fold_{index}.txt"
        booster.save_model(str(model_dir / name))
        names.append(name)
    write_json(
        model_dir / "model_meta.json",
        {
            "model": bundle.model,
            "feature_columns": list(bundle.feature_columns),
            "params": _jsonable(bundle.params),
            "folds": bundle.folds,
            "fold_mode": bundle.fold_mode,
            "threshold": bundle.threshold,
            "score_feature": bundle.score_feature,
            "booster_files": names,
            "prediction_mode": "mean of the per-fold models" if len(names) > 1 else "single model",
        },
    )
    return model_dir


def load_bundle(out_dir: str | Path) -> ModelBundle:
    """Rebuild a bundle from a saved run, so scoring needs no retraining."""
    out_dir = Path(out_dir)
    model_dir = out_dir / "model"
    meta_path = model_dir / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no saved model at {meta_path} - run train() first")
    meta = read_json(meta_path)

    boosters = []
    if meta.get("model") == MODEL_LIGHTGBM and meta.get("booster_files"):
        lightgbm = _import_lightgbm()
        for name in meta["booster_files"]:
            boosters.append(lightgbm.Booster(model_file=str(model_dir / name)))

    metrics_path = out_dir / "v1_metrics.json"
    return ModelBundle(
        model=meta.get("model", MODEL_LIGHTGBM),
        feature_columns=tuple(meta.get("feature_columns") or FEATURE_COLUMNS),
        params=meta.get("params") or {},
        folds=int(meta.get("folds", DEFAULT_FOLDS)),
        fold_mode=str(meta.get("fold_mode", FOLD_MODE_HASH)),
        threshold=float(meta.get("threshold", float("nan"))),
        score_feature=str(meta.get("score_feature", "name_token_set_ratio")),
        boosters=boosters,
        metrics=read_json(metrics_path) if metrics_path.is_file() else {},
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _log_summary(log: logging.Logger, metrics: dict[str, Any]) -> None:
    """The numbers the run exists to produce, in one block."""
    final = metrics["val_metrics"]
    baselines = metrics["baselines"]
    choice = metrics["operating_point"]
    rule = "=" * 78
    log.info(rule)
    log.info(
        "V1 MATCHER - %s%s (smoke=%s)",
        metrics["model"],
        f" on {metrics['score_feature']}" if metrics.get("score_feature") else "",
        metrics["is_smoke"],
    )
    log.info(rule)
    log.info("  rows                     : %s", fmt_int(metrics["rows"]))
    log.info(
        "  labels                   : %s positive / %s negative (%.4f%% positive)",
        fmt_int(metrics["positive_labels"]),
        fmt_int(metrics["negative_labels"]),
        100.0 * metrics["positive_rate"],
    )
    log.info(
        "  S1 entities with rows    : %s of %s in the ground truth",
        fmt_int(metrics["n_s1_entities_with_rows"]),
        fmt_int((metrics.get("labels") or {}).get("n_s1_entities_in_ground_truth") or 0),
    )
    log.info("  folds                    : %s (%s)", metrics["n_folds"], metrics["fold_mode"])
    purity = metrics.get("fold_purity") or {}
    if purity:
        # Printed because it is the coverage of the leakage check, not a formality: the
        # rest of the ground truth has no candidate rows in this run to keep together.
        log.info(
            "  entity-fold purity       : %s entities checked, each in exactly one fold",
            fmt_int(purity.get("entities_checked", 0)),
        )
    log.info(
        "  threshold                : %.6g  (plateau %s points, %s)",
        metrics["threshold"],
        fmt_int(choice["plateau_points"]),
        "single point" if choice["plateau_is_single_point"] else "flat",
    )
    log.info("  predicted pairs          : %s", fmt_int(final["predicted_pairs"]))
    log.info("  -- challenge metric (macro F0.5, score_zero) " + "-" * 31)
    log.info("  V1 macro F0.5            : %.4f", final["macro_f05_score_zero"])
    log.info("  macro F0.5 (exclude policy) : %.4f", final["macro_f05_exclude"])
    log.info("  -- baselines " + "-" * 60)
    log.info(
        "  predict nothing          : %.4f   (empty-GT entities only)",
        baselines["predict_nothing_macro_f05_score_zero"],
    )
    log.info(
        "  accept every candidate   : %.4f   (score_zero) / %.4f (exclude)",
        baselines["accept_every_candidate_macro_f05_score_zero"],
        baselines["accept_every_candidate_macro_f05_exclude"],
    )
    log.info(
        "  candidate precision      : %.4f  (pair recall of the candidate set: %.4f)",
        baselines["candidate_precision_of_the_feature_file"],
        baselines["candidate_pair_recall_of_the_feature_file"],
    )
    log.info("  -- pair-level diagnostics (NOT the objective) " + "-" * 30)
    log.info(
        "  precision / recall / F0.5: %.4f / %.4f / %.4f",
        final["pair_precision"],
        final["pair_recall"],
        final["pair_f05"],
    )
    log.info(
        "  macro precision / recall : %.4f / %.4f",
        final["macro_precision_entity"],
        final["macro_recall_entity"],
    )
    log.info(
        "  empty-GT entities merged : %s of %s",
        fmt_int(final["n_empty_gt_entities_with_predictions"]),
        fmt_int(final["n_empty_gt_entities"]),
    )
    log.info(
        "  matched S1 with nothing predicted : %s",
        fmt_int(final["n_matched_entities_with_no_prediction"]),
    )
    log.info(
        "  predictions per S1       : mean %.2f, max %s",
        final["mean_predictions_per_s1"],
        fmt_int(final["max_predictions_per_s1"]),
    )
    for label, stats in final["per_source"].items():
        log.info(
            "  %s: %s predictions, %s true (%s true pairs), precision %.4f, macro F0.5 %.4f",
            label,
            fmt_int(stats["predictions"]),
            fmt_int(stats["true_positives"]),
            fmt_int(stats["n_true_pairs"]),
            stats["pair_precision"],
            stats["macro_f05_source_restricted"],
        )
    log.info("  artifacts                : %s", metrics.get("out_dir") or "")
    log.info(rule)


def describe_bundle(bundle: ModelBundle) -> str:
    """One-line description, handy in logs and tests."""
    return (
        f"ModelBundle(model={bundle.model}, features={len(bundle.feature_columns)}, "
        f"boosters={len(bundle.boosters)}, folds={bundle.folds}, "
        f"threshold={bundle.threshold:.6g}, ready={bundle.is_ready()})"
    )


def feature_matrix_bytes(n_rows: int, n_features: int = len(FEATURE_COLUMNS)) -> str:
    """Matrix size estimate for a run, so the storage decision is made with numbers."""
    return human_bytes(n_rows * n_features * 4)
