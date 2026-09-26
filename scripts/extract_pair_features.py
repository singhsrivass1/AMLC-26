#!/usr/bin/env python
"""Step 3 de-risk: materialize the first matcher features on a sampled candidate set.

This is a **measurement script, not a pipeline stage**. It exists to answer one
question before anyone writes the matcher: can the pair-feature stage run over the
production candidate set at all, and at what cost?

It deliberately does NOT:

* process all 336M pairs (it samples S1 entities and keeps every candidate of the
  sampled entities, so the per-entity structure the macro metric depends on is
  preserved);
* train anything, tune anything, or touch the blockers;
* read ground truth. The validation split is derived from the S1 id alone, with
  ``assign_splits`` - the same pure function ``src/evaluation.py`` uses - so no
  label can leak into a feature and the split cannot drift from the evaluator's.

Why a two-phase design
----------------------
``s1_candidate_count`` is one of the most valuable features in the first matcher
(it tells the model how crowded an entity's candidate list is), and it has to be
known before the sampled rows are featurized. Phase 1 streams the candidate file
once, keeps the sampled rows, and accumulates the per-S1 counts on the way past;
phase 2 reads only the small sampled file, joins it to the prepared text, and
computes features. That buys two things: the scan - the dominant cost - is
measured on its own rather than mixed into the feature timing, and all the
string work happens on a few million rows instead of 336M.

Note what the counts are and are not. Because the sample keeps **whole** entities,
an entity's rows in the sample are all of its rows in the file, so the counts are
also derivable from the sample alone - phase 1 computes them only because that
pass already touches every row. Phase 2 therefore re-counts them independently and
compares: any mismatch means an entity was split across the sample boundary, which
would silently corrupt the feature, and the run fails rather than reporting it.

    python scripts/extract_pair_features.py
    python scripts/extract_pair_features.py --sample-fraction 0.05
    python scripts/extract_pair_features.py --workers 8               # 8 shards
    python scripts/extract_pair_features.py --limit-rows 200000        # smoke test

Parallel execution (``--workers N``)
-----------------------------------
``--workers 1`` (the default) is the original single-process path, unchanged. For
``N > 1`` phase 1 still runs once, in the parent: the candidate file is streamed,
the validation sample is decided per S1 id exactly as before, and nothing about
which entities are selected changes. Only phase 2 is parallelized, and the parent
carries no prepared text while it happens:

1. The parent partitions the **selected S1 entities** - not the rows - into N
   contiguous, row-balanced groups (``partition_entities``). Entity to worker is a
   pure function of the entity's position in the scan and its candidate count, so
   the partition is reproducible to the row and each entity lands on exactly one
   worker.
2. The parent writes one shard file per worker under ``<output-dir>/workers/``,
   appending rows in file order, so a shard holds its entities' rows in the order
   they appeared in the sample.
3. Each worker loads *only the text its own shard can join to* - its S1 ids, and
   the target ids its rows reference - and streams its shard twice: once to learn
   those ids, once to featurize. The filter changes what is resident, never what a
   lookup returns, because an id is kept whenever it is needed and present in the
   prepared file, and an id absent from the prepared file is a join failure either
   way. W workers therefore hold roughly one copy of the prepared text between
   them instead of W copies.
4. The parent concatenates the per-worker feature files **in worker order**, not in
   completion order. Because the partition is contiguous in file order, and the
   generator emits a pair's rows under its S1 entity ("groups stay contiguous in
   the output"), this reproduces the sample file's row order exactly - so the
   merged output is byte-identical to the single-process output rather than merely
   equivalent. If a candidate file were ever not grouped that way, the merged file
   would hold the same rows in a different order: the same values either way, and
   the regression test compares both the multiset and the worker-order sequence.

Workers are started with the ``spawn`` start method on every platform. ``fork``
would share the parent's address space copy-on-write, which buys nothing here
(the parent holds no prepared text) and would make the per-worker RSS figures
include inherited pages - and RSS is one of the numbers this experiment exists to
report.

Feature definitions, column names, dtypes, blank-evidence rules and join-failure
rules are identical in both paths; they are one implementation (``build_features``
plus :class:`FeatureStats`), not two.

Outputs, all inside one experiment directory so nothing lands in the production
``outputs/candidates/`` tree::

    sample_candidates.tsv        the sampled candidate rows, verbatim
    features.tsv                 ids + features, one row per candidate pair
    feature_missingness.csv      per-feature missing rate
    step3_features_report.json   every measurement this experiment must report
    extract_pair_features.log    the run log
    workers/                     --workers N > 1 only: one shard, per-worker log and
                                 per-worker feature file per worker, kept after the
                                 run so a disagreement can be traced to a worker
                                 (--cleanup-shards removes them)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

# Make ``import src.*`` work when the script is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (  # noqa: E402
    BLOCKER_CHAR_NGRAM,
    BLOCKER_DENSE,
    BLOCKER_EXACT_NAME,
    BLOCKER_TOKEN,
    UNION_BLOCKERS,
    _trigram_jaccard,
    evidence_columns_for,
)
from src.data_loader import (  # noqa: E402
    ChunkWriter,
    assign_splits,
    candidates_path,
    describe_environment,
    iter_tsv,
    load_config,
    prepared_path,
)
from src.utils import (  # noqa: E402
    current_rss_bytes,
    ensure_dir,
    fmt_int,
    human_bytes,
    log_memory,
    peak_rss_bytes,
    set_seed,
    setup_logging,
    stable_hash64,
    write_json,
)

LOG_NAME = "extract_pair_features"

# Candidate schema (see scripts/generate_candidates.py). The evidence columns are
# present only when the blocker that produces them was enabled.
CANDIDATE_S1_COLUMN = "source1_entity_id"
CANDIDATE_TARGET_COLUMN = "matched_entity_id"
CANDIDATE_SOURCE_COLUMN = "source"
CANDIDATE_BLOCKERS_COLUMN = "blockers"

# Prepared schema (see scripts/prepare_data.py).
PREPARED_ID_COLUMN = "entity_id"
PREPARED_NAME_NORM = "name_norm"
PREPARED_NAME_KEY = "name_key"
PREPARED_ADDRESS_NORM = "address_norm"
PREPARED_COUNTRY_NORM = "country_norm"

PREPARED_COLUMNS = (
    PREPARED_ID_COLUMN,
    PREPARED_NAME_NORM,
    PREPARED_NAME_KEY,
    PREPARED_ADDRESS_NORM,
    PREPARED_COUNTRY_NORM,
)

# The full-corpus production acceptance target, used only for extrapolation. It is
# written down rather than re-measured: this experiment's job is to extrapolate,
# not to re-run generation.
FULL_CANDIDATE_PAIRS = 336_056_756

# The worker count the report used to project to on the strength of linear scaling
# alone. That projection is now a measured one at the requested ``--workers``, so
# this constant survives only to label the clearly-marked theoretical line.
THEORETICAL_HPC_WORKERS = 48

# Default worker count. 1 is the original single-process path, so an invocation
# that does not ask for parallelism gets byte-identical behaviour to before.
DEFAULT_WORKERS = 1

# Where the per-worker shard, log and feature files live, inside --output-dir.
WORKER_DIR_NAME = "workers"

# Evidence columns carried by the candidate file, and the dtypes to parse them as.
# A blank means "the blocker that measures this value did not propose this pair",
# which is the normal case in a union - so blank must become NaN, never 0.
EVIDENCE_FLOAT_COLUMNS = evidence_columns_for(UNION_BLOCKERS)

# Feature dtypes. Fixed up front so the matrix is dense, typed and small, and so
# the report can state exactly what the full-scale matrix would cost.
FEATURE_DTYPES: dict[str, str] = {
    # -- name --
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
    # -- address --
    "address_norm_equal": "uint8",
    "address_token_jaccard": "float32",
    "address_shared_token_count": "int16",
    "address_length_ratio": "float32",
    "s1_address_missing": "uint8",
    "target_address_missing": "uint8",
    "both_address_missing": "uint8",
    # -- blocker evidence + provenance --
    "token_df": "float32",
    "char_jaccard": "float32",
    "dense_cosine": "float32",
    "blocker_exact_name": "uint8",
    "blocker_token": "uint8",
    "blocker_char_ngram": "uint8",
    "blocker_dense": "uint8",
    "n_blockers": "uint8",
    "s1_candidate_count": "int32",
    # -- other --
    "source_is_s2": "uint8",
    "country_equal": "uint8",
    "country_missing": "uint8",
    # -- integrity, not a feature --
    "text_join_ok": "uint8",
}

# Columns present in the written matrix that are NOT features and must be dropped
# before training. They are written so a failed join stays diagnosable per row, but
# a trainer that reads the whole matrix would hand the model a column describing
# whether the feature pipeline worked - which is not a property of the pair.
NON_FEATURE_COLUMNS = ("text_join_ok",)

# Features that are defined on [0, 1]. Checked on every batch so a scale error
# (e.g. leaving rapidfuzz on its 0-100 scale) trips the check instead of silently
# feeding the model a 0-100 column. ``token_df`` is deliberately NOT in this list:
# it is a document frequency, not a similarity.
UNIT_INTERVAL_FEATURES = (
    "name_token_jaccard",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_char3_jaccard",
    "name_length_ratio",
    "address_token_jaccard",
    "address_length_ratio",
    "char_jaccard",
)

INTEGRITY_COLUMNS = (CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN)

# Features that come from joining to the prepared text. A failed join blanks these
# and only these - provenance, evidence and the S1 candidate count are properties of
# the candidate file itself and stay valid either way.
TEXT_DERIVED_FEATURES = tuple(
    column
    for column in FEATURE_DTYPES
    if column.startswith(("name_", "address_", "country_"))
    or column in ("s1_address_missing", "target_address_missing", "both_address_missing")
)


def _sample_rss(tracker: list[int]) -> None:
    """Record the current RSS, keeping the running maximum.

    ``peak_rss_bytes()`` returns ``None`` on Windows (``resource`` is unavailable),
    so the highest *observed* RSS is tracked as a portable lower bound on the true
    peak. On Linux the two agree closely, because the run logs at every phase
    boundary and every ten feature batches.
    """
    value = current_rss_bytes()
    if value:
        tracker[0] = max(tracker[0], value)


def _peak_description(exact: Optional[int], sampled: int) -> tuple[Optional[int], str]:
    """The best peak-RSS number available, and where it came from."""
    if exact and sampled:
        return max(exact, sampled), "resource.getrusage high-water mark"
    if exact:
        return exact, "resource.getrusage high-water mark"
    if sampled:
        return sampled, "max observed RSS (getrusage unavailable on this platform)"
    return None, "unavailable"


# ---------------------------------------------------------------------------
# pure per-pair helpers (the semantics live here and nowhere else)
# ---------------------------------------------------------------------------
def token_set(text: str) -> frozenset[str]:
    """Distinct whitespace tokens of an already-normalized field."""
    return frozenset(text.split()) if text else frozenset()


def set_jaccard(left: frozenset, right: frozenset) -> float:
    """Jaccard overlap of two token sets; 0.0 when either side is empty.

    An empty side means "no tokens to compare", which is not evidence of a match,
    so the intersection-based 1.0 that an empty-vs-empty comparison would produce
    is deliberately not returned.
    """
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared)


def length_ratio(left: str, right: str) -> float:
    """Shorter/longer character length; 0.0 when either side is empty."""
    a, b = len(left), len(right)
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def first_token(text: str) -> str:
    """First whitespace token, or "" for an empty field."""
    parts = text.split()
    return parts[0] if parts else ""


def parse_provenance(text: str) -> tuple[int, int, int, int, int, int]:
    """``(exact, token, char, dense, n_blockers, unknowns)`` from a provenance string.

    Provenance survives the union as comma-joined ``sourceN:blocker`` labels (see
    ``union_blockers``), so the blocker part is what identifies the blocker.
    """
    exact = token = char = dense = unknown = 0
    if text:
        for part in text.split(","):
            blocker = part.rsplit(":", 1)[-1]
            if blocker == BLOCKER_EXACT_NAME:
                exact = 1
            elif blocker == BLOCKER_TOKEN:
                token = 1
            elif blocker == BLOCKER_CHAR_NGRAM:
                char = 1
            elif blocker == BLOCKER_DENSE:
                dense = 1
            else:
                unknown += 1
    return exact, token, char, dense, exact + token + char + dense, unknown


# ---------------------------------------------------------------------------
# prepared-text lookup
# ---------------------------------------------------------------------------
class PreparedLookup:
    """``entity_id`` -> the normalized text columns, for one source.

    A dict of positions plus one object array per column: the join is then a
    Python ``get`` per sampled row, which is cheap because only sampled rows are
    ever looked up. Loading the *whole* prepared file (all 10.3M target records)
    is deliberate - it is the honest measure of what the join costs in RAM, and it
    is the reason the sampled rows can be featurized in a single streaming pass.
    """

    def __init__(self, source: str, ids: Sequence[str], columns: dict[str, np.ndarray]) -> None:
        self.source = source
        self.ids = ids
        self.columns = columns
        self.index = {entity_id: position for position, entity_id in enumerate(ids)}

    @property
    def n_entities(self) -> int:
        return len(self.ids)

    def take(self, entity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions and a found-mask for a batch of ids; -1 where absent."""
        index = self.index
        positions = np.fromiter(
            (index.get(entity_id, -1) for entity_id in entity_ids),
            dtype=np.int64,
            count=len(entity_ids),
        )
        return positions, positions >= 0

    def values(self, column: str, positions: np.ndarray, found: np.ndarray) -> np.ndarray:
        """Text for a column; "" where the id was not found."""
        values = self.columns[column]
        out = np.full(len(positions), "", dtype=object)
        if found.any():
            out[found] = values[positions[found]]
        return out

    TEXT_SAMPLE_ROWS = 2000

    def memory_bytes(self) -> int:
        """Estimated resident bytes of this lookup.

        The string payload of an object array is the dominant cost and is not in
        ``nbytes``, so it is estimated by sampling ``TEXT_SAMPLE_ROWS`` elements per
        column and extrapolating. This is an estimate, not a measurement: it exists
        to answer "how many GB per worker does the full prepared source cost?", and
        the process-wide RSS line in the log is the check on it.
        """
        total = sum(int(array.nbytes) for array in self.columns.values())
        total += len(self.ids) * 8 * (1 + len(self.columns)) + len(self.index) * 100
        for array in (self.ids, *self.columns.values()):
            n = len(array)
            if n == 0:
                continue
            if n <= self.TEXT_SAMPLE_ROWS:
                sample = array
            else:
                sample = array[:: n // self.TEXT_SAMPLE_ROWS]
            total += sum(sys.getsizeof(value) for value in sample) * (n / len(sample))
        return int(total)


def load_lookup(
    config: dict,
    split: str,
    source: str,
    log: logging.Logger,
    keep_ids: Optional[set[str]] = None,
) -> PreparedLookup:
    """Load one prepared source into a :class:`PreparedLookup`.

    ``keep_ids`` restricts the load to those entity ids. A worker passes exactly
    the ids its shard can join to, so W workers hold roughly one copy of the
    prepared text between them instead of W copies. The filter cannot change a
    join result: an id is kept whenever it is both needed and present in the
    prepared file, and an id absent from the prepared file is a join failure with
    or without the filter. Unfiltered (the default) is the single-process load.
    """
    path = prepared_path(config, split, source)
    if not path.is_file():
        raise FileNotFoundError(
            f"prepared file missing: {path}\n"
            f"  Run first: python scripts/prepare_data.py --splits {split}"
        )
    parts: list[list[str]] = []
    buffers: dict[str, list[np.ndarray]] = {column: [] for column in PREPARED_COLUMNS if column != PREPARED_ID_COLUMN}
    chunksize = max(1, int(config.get("io", {}).get("chunksize", 500_000)))
    # Hoisted out of the chunk loop: pandas rebuilds its lookup table on every
    # ``isin`` call, so passing the caller's set would re-read it once per chunk.
    wanted = (
        None
        if keep_ids is None
        else np.fromiter(keep_ids, dtype=object, count=len(keep_ids))
    )
    rows_read = 0
    for frame in iter_tsv(path, columns=list(PREPARED_COLUMNS), chunksize=chunksize):
        rows_read += len(frame)
        if wanted is not None:
            mask = frame[PREPARED_ID_COLUMN].isin(wanted)
            if not mask.any():
                continue
            frame = frame.loc[mask]
        parts.append(frame[PREPARED_ID_COLUMN].to_numpy(dtype=object))
        for column in buffers:
            buffers[column].append(frame[column].to_numpy(dtype=object))
    ids = np.concatenate(parts) if parts else np.empty(0, dtype=object)
    columns = {
        column: (np.concatenate(values) if values else np.empty(0, dtype=object))
        for column, values in buffers.items()
    }
    lookup = PreparedLookup(source, ids, columns)
    log.info(
        "  loaded %s prepared rows from %s (lookup ~%s, RSS now %s)%s",
        fmt_int(lookup.n_entities),
        path.name,
        human_bytes(lookup.memory_bytes()),
        human_bytes(current_rss_bytes() or 0),
        f" [filtered from {fmt_int(rows_read)} rows]" if keep_ids is not None else "",
    )
    return lookup


# ---------------------------------------------------------------------------
# phase 1: sample S1 entities out of the candidate file
# ---------------------------------------------------------------------------
POPULATION_VAL = "val"
POPULATION_TRAIN = "train"
POPULATION_ALL = "all"
POPULATIONS = (POPULATION_VAL, POPULATION_TRAIN, POPULATION_ALL)


def resolve_population(split: str, requested: Optional[str]) -> str:
    """Which S1 entities a run may sample from.

    * ``--split test``: always **every** entity. The test split has no train/val
      partition - the val hash is a device for holding out *labelled* entities - and
      filtering by it used to drop ~80% of the test S1 entities, which would then
      have been emitted as singletons.
    * ``--split train``: ``all`` by default, so the matcher trains on every labelled
      entity (its entity-grouped out-of-fold scores keep that leak-free, and the
      threshold is still tuned on the val entities). ``val`` reproduces the original
      val-only de-risk sample.
    """
    if split == "test":
        if requested not in (None, POPULATION_ALL):
            raise ValueError(
                f"--population {requested} is meaningless for --split test: every test "
                "S1 entity must be featurized, or it is silently emitted with no match"
            )
        return POPULATION_ALL
    return requested or POPULATION_ALL


def _sample_mask_for_ids(
    unique_ids: np.ndarray,
    cache: dict[str, int],
    config: dict,
    sub_threshold: int,
    log: logging.Logger,
    population: str = POPULATION_VAL,
) -> np.ndarray:
    """Which of ``unique_ids`` are in the sample. Pure function of the id.

    Two conditions, both deterministic functions of the S1 id and nothing else:
    the entity must be in ``population`` (``val``/``train`` by ``assign_splits``, the
    same function ``src/evaluation.py`` uses; ``all`` = no split filter), and it must
    fall below ``sub_threshold`` in a second bucket taken from the high bits of the
    same 64-bit hash. Because both are functions of the id alone, every candidate row
    of an entity makes the same decision, so whole entities are kept together
    without needing the file to be grouped by S1.
    """
    if population not in POPULATIONS:
        raise ValueError(f"unknown population {population!r}; expected one of {POPULATIONS}")
    section = config.get("evaluation", {}).get("split", {}) or {}
    val_fraction = section.get("val_fraction", 0.2)
    mode = section.get("mode", "hash")
    seed = config.get("project", {}).get("seed", 42)

    missing = [entity_id for entity_id in unique_ids if entity_id not in cache]
    if missing:
        series = pd.Series(missing, dtype=object)
        if population == POPULATION_ALL:
            in_population = np.ones(len(series), dtype=bool)
        else:
            labels = assign_splits(series, val_fraction=val_fraction, mode=mode, seed=seed)
            in_population = labels == population
        # A different slice of the same hash than assign_splits uses, so the
        # subsample is independent of the split decision.
        buckets = (stable_hash64(series) // np.uint64(1_000_000)) % np.uint64(1_000_000)
        keep = in_population & (buckets < np.uint64(sub_threshold))
        for entity_id, flag in zip(missing, keep):
            cache[entity_id] = 1 if flag else 0
    return np.fromiter(
        (cache[entity_id] for entity_id in unique_ids), dtype=bool, count=len(unique_ids)
    )


def scan_and_sample(
    config: dict,
    args: argparse.Namespace,
    output_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    """Phase 1: stream the candidate file, write the sampled rows, count per S1."""
    population = getattr(args, "population", None) or POPULATION_VAL
    source_path = candidates_path(config, args.candidates, split=args.split)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"candidate file not found: {source_path}\n"
            f"  Run first: python scripts/generate_candidates.py --split {args.split}"
        )

    available = _peek_columns(source_path)
    missing = [
        column
        for column in (CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN, CANDIDATE_SOURCE_COLUMN)
        if column not in available
    ]
    if missing:
        raise ValueError(f"candidate file {source_path} is missing column(s) {missing}")

    read_columns = [
        column
        for column in (
            CANDIDATE_S1_COLUMN,
            CANDIDATE_TARGET_COLUMN,
            CANDIDATE_SOURCE_COLUMN,
            CANDIDATE_BLOCKERS_COLUMN,
            *EVIDENCE_FLOAT_COLUMNS,
        )
        if column in available
    ]
    evidence_columns = [column for column in EVIDENCE_FLOAT_COLUMNS if column in available]

    sub_threshold = int(round(min(max(args.sample_fraction, 0.0), 1.0) * 1_000_000))
    chunksize = args.chunksize or int(config.get("io", {}).get("chunksize", 500_000))

    log.info("scanning %s", source_path)
    log.info("  columns read : %s", ", ".join(read_columns))
    log.info("  evidence cols: %s", ", ".join(evidence_columns) or "(none)")
    log.info(
        "  sample       : %.3f%% of %s S1 entities (population=%s)",
        args.sample_fraction * 100.0,
        {"val": "validation", "train": "train-split", "all": "all"}[population],
        population,
    )

    cache: dict[str, int] = {}
    counts: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    rows_scanned = 0
    rows_sampled = 0
    seen_entities: set[str] = set()
    rss_tracker = [0]
    sample_path = output_dir / "sample_candidates.tsv.partial"

    started = time.time()
    with ChunkWriter(sample_path) as writer:
        chunks = iter_tsv(source_path, columns=read_columns, chunksize=chunksize)
        for chunk in chunks:
            if args.limit_rows is not None and rows_scanned >= args.limit_rows:
                break
            if args.limit_rows is not None and rows_scanned + len(chunk) > args.limit_rows:
                chunk = chunk.iloc[: args.limit_rows - rows_scanned]
            rows_scanned += len(chunk)

            s1_array = chunk[CANDIDATE_S1_COLUMN].to_numpy(dtype=object)
            # Factorize first: a 500k-row chunk holds only a few thousand distinct
            # S1 entities, so the sample decision is made a few thousand times
            # instead of 500k times.
            codes, uniques = pd.factorize(s1_array, sort=False)
            # Exact global distinct count: union the per-chunk unique ids. This costs one
            # set insert per per-chunk-distinct id, and generate_candidates.py emits a
            # whole entity's candidate block contiguously, so a 500k-row chunk holds a few
            # thousand distinct ids rather than 500k - the insert count stays near the true
            # distinct count (2.2M) instead of near the row count (336M). It is NOT a sum
            # of per-chunk distinct counts, which would double-count ids straddling chunks.
            seen_entities.update(uniques)
            keep_by_entity = _sample_mask_for_ids(
                uniques, cache, config, sub_threshold, log, population=population
            )
            keep = keep_by_entity[codes] if len(keep_by_entity) else np.zeros(len(chunk), dtype=bool)
            if not keep.any():
                _sample_rss(rss_tracker)
                continue

            sampled = chunk.loc[keep]
            rows_sampled += len(sampled)
            writer.append(sampled)

            # Per-S1 candidate counts: the S1's size in the WHOLE file, not in the
            # sample. This is the feature the matcher needs to know how crowded an
            # entity's candidate list is.
            chunk_counts = np.bincount(codes[keep], minlength=len(uniques))
            for position in np.flatnonzero(chunk_counts):
                entity_id = uniques[position]
                counts[entity_id] = counts.get(entity_id, 0) + int(chunk_counts[position])

            # Exact duplicate detection within the sample. A duplicated pair in the
            # file puts both copies in the sample (same S1), so the sampled rate is
            # an unbiased estimate of the file-wide rate - and it needs a set of
            # only the sampled pairs rather than of all 336M.
            for entity_id, target_id in zip(
                sampled[CANDIDATE_S1_COLUMN].to_numpy(dtype=object),
                sampled[CANDIDATE_TARGET_COLUMN].to_numpy(dtype=object),
            ):
                key = (entity_id, target_id)
                if key in seen_pairs:
                    duplicate_pairs += 1
                else:
                    seen_pairs.add(key)
            _sample_rss(rss_tracker)

    scan_seconds = time.time() - started

    if rows_sampled == 0:
        raise RuntimeError(
            "no candidate row matched the sample; is the candidate file empty, or "
            "is --sample-fraction too small?"
        )
    sample_path.replace(output_dir / "sample_candidates.tsv")

    log.info(
        "  scanned %s rows in %.1f s (%.0f rows/s); sampled %s pairs over %s S1 entities",
        fmt_int(rows_scanned),
        scan_seconds,
        rows_scanned / max(scan_seconds, 1e-9),
        fmt_int(rows_sampled),
        fmt_int(len(counts)),
    )
    log_memory(log, "after scan")

    return {
        "source_path": str(source_path),
        "read_columns": read_columns,
        "evidence_columns": evidence_columns,
        "rows_scanned": rows_scanned,
        "rows_sampled": rows_sampled,
        "limited": args.limit_rows is not None,
        "n_s1_entities_sampled": len(counts),
        "n_s1_entities_seen": len(seen_entities),
        "n_s1_entity_id_set_bytes": sum(
            sys.getsizeof(entity_id) for entity_id in seen_entities
        ),
        "duplicate_sampled_pairs": duplicate_pairs,
        "per_s1_counts": counts,
        "scan_seconds": scan_seconds,
        "rss_sampled_peak": rss_tracker[0],
        "sample_path": output_dir / "sample_candidates.tsv",
        "sample_fraction": args.sample_fraction,
        "population": population,
    }


def _peek_columns(path: Path) -> list[str]:
    """Read just the header of a (possibly compressed) TSV."""
    import gzip

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    return header.rstrip("\n").split("\t")


# ---------------------------------------------------------------------------
# phase 2: join + features
# ---------------------------------------------------------------------------
def _bump(integrity: dict[str, int], key: str, amount: int) -> None:
    """Increment one integrity counter, creating it if the caller did not pre-seed it.

    ``build_features`` accumulates into a caller-owned dict, so a caller that passes
    ``{}`` must get a working counter rather than a ``KeyError`` deep in the join. The
    key is created even when ``amount`` is 0, so a counter that happened to stay at
    zero is still present (and readable) rather than missing.
    """
    integrity[key] = integrity.get(key, 0) + int(amount)


def _seed_integrity() -> dict[str, int]:
    """The integrity counters every feature pass starts from, at zero.

    One definition, used by the single-process pass and by every worker, so the
    two cannot report different sets of counters.
    """
    integrity: dict[str, int] = {
        "s1_join_failures": 0,
        "target_join_failures": 0,
        "unknown_source_labels": 0,
        "unknown_blocker_labels": 0,
        "missing_s1_counts": 0,
        "rapidfuzz_available": 0,
    }
    for column in EVIDENCE_FLOAT_COLUMNS:
        integrity[f"{column}_blank"] = 0
    return integrity


def build_features(
    frame: pd.DataFrame,
    lookups: dict[str, PreparedLookup],
    s1_lookup: PreparedLookup,
    counts: dict[str, int],
    integrity: dict[str, int],
) -> pd.DataFrame:
    """One row of features per candidate pair. Never drops a pair."""
    n = len(frame)
    s1_ids = frame[CANDIDATE_S1_COLUMN].to_numpy(dtype=object)
    target_ids = frame[CANDIDATE_TARGET_COLUMN].to_numpy(dtype=object)
    source_labels = frame[CANDIDATE_SOURCE_COLUMN].to_numpy(dtype=object)

    # -- join ---------------------------------------------------------------
    s1_positions, s1_found = s1_lookup.take(s1_ids)
    target_positions = np.full(n, -1, dtype=np.int64)
    target_found = np.zeros(n, dtype=bool)
    for label, lookup in lookups.items():
        selected = np.flatnonzero(source_labels == label)
        if not selected.size:
            continue
        positions, found = lookup.take(target_ids[selected])
        target_positions[selected] = positions
        target_found[selected] = found
    unknown_source = ~np.isin(source_labels, list(lookups))

    text_join_ok = s1_found & target_found & ~unknown_source
    _bump(integrity, "s1_join_failures", (~s1_found).sum())
    _bump(integrity, "target_join_failures", (~target_found & ~unknown_source).sum())
    _bump(integrity, "unknown_source_labels", unknown_source.sum())

    s1_name_norm = s1_lookup.values(PREPARED_NAME_NORM, s1_positions, s1_found)
    s1_name_key = s1_lookup.values(PREPARED_NAME_KEY, s1_positions, s1_found)
    s1_address = s1_lookup.values(PREPARED_ADDRESS_NORM, s1_positions, s1_found)
    s1_country = s1_lookup.values(PREPARED_COUNTRY_NORM, s1_positions, s1_found)

    target_name_norm = np.full(n, "", dtype=object)
    target_name_key = np.full(n, "", dtype=object)
    target_address = np.full(n, "", dtype=object)
    target_country = np.full(n, "", dtype=object)
    for label, lookup in lookups.items():
        selected = np.flatnonzero((source_labels == label) & target_found)
        if not selected.size:
            continue
        positions = target_positions[selected]
        target_name_norm[selected] = lookup.values(PREPARED_NAME_NORM, positions, target_found[selected])
        target_name_key[selected] = lookup.values(PREPARED_NAME_KEY, positions, target_found[selected])
        target_address[selected] = lookup.values(PREPARED_ADDRESS_NORM, positions, target_found[selected])
        target_country[selected] = lookup.values(PREPARED_COUNTRY_NORM, positions, target_found[selected])

    # -- name ---------------------------------------------------------------
    s1_tokens = [token_set(text) for text in s1_name_norm]
    target_tokens = [token_set(text) for text in target_name_norm]
    name_token_jaccard = np.fromiter(
        (set_jaccard(a, b) for a, b in zip(s1_tokens, target_tokens)),
        dtype=np.float64,
        count=n,
    )

    try:
        from rapidfuzz import fuzz

        token_set_ratio = np.fromiter(
            (fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_name_norm, target_name_norm)),
            dtype=np.float64,
            count=n,
        )
        token_sort_ratio = np.fromiter(
            (fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_name_norm, target_name_norm)),
            dtype=np.float64,
            count=n,
        )
        partial_ratio = np.fromiter(
            (fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_name_norm, target_name_norm)),
            dtype=np.float64,
            count=n,
        )
        integrity["rapidfuzz_available"] = 1
    except ImportError:  # pragma: no cover - depends on the environment
        token_set_ratio = np.full(n, np.nan)
        token_sort_ratio = np.full(n, np.nan)
        partial_ratio = np.full(n, np.nan)
        integrity["rapidfuzz_available"] = 0

    # The same trigram Jaccard the char blocker verifies with, on the same field
    # (name_key), so the feature and the blocker cannot disagree.
    name_char3_jaccard = np.fromiter(
        (_trigram_jaccard(a, b) for a, b in zip(s1_name_key, target_name_key)),
        dtype=np.float64,
        count=n,
    )
    name_length_ratio = np.fromiter(
        (length_ratio(a, b) for a, b in zip(s1_name_norm, target_name_norm)),
        dtype=np.float64,
        count=n,
    )
    name_token_count_diff = np.fromiter(
        (abs(len(a) - len(b)) for a, b in zip(s1_tokens, target_tokens)),
        dtype=np.int64,
        count=n,
    )
    name_first_token_equal = np.fromiter(
        (1 if first_token(a) and first_token(a) == first_token(b) else 0
         for a, b in zip(s1_name_norm, target_name_norm)),
        dtype=np.int64,
        count=n,
    )

    # -- address ------------------------------------------------------------
    # ``address_norm_equal`` deliberately requires a non-blank address. Two records
    # that both have no address compare equal as empty strings, and that "" == ""
    # carries no evidence of a match - it would hand the model a spurious positive.
    # The missingness flags below let it learn the absence explicitly instead.
    address_norm_equal = np.fromiter(
        (1 if a and a == b else 0 for a, b in zip(s1_address, target_address)),
        dtype=np.int64,
        count=n,
    )
    s1_address_missing = np.fromiter((1 if not a else 0 for a in s1_address), dtype=np.int64, count=n)
    target_address_missing = np.fromiter((1 if not a else 0 for a in target_address), dtype=np.int64, count=n)
    both_address_missing = (s1_address_missing & target_address_missing).astype(np.int64)

    s1_address_tokens = [token_set(text) for text in s1_address]
    target_address_tokens = [token_set(text) for text in target_address]
    address_token_jaccard = np.fromiter(
        (set_jaccard(a, b) for a, b in zip(s1_address_tokens, target_address_tokens)),
        dtype=np.float64,
        count=n,
    )
    address_shared_token_count = np.fromiter(
        (len(a & b) for a, b in zip(s1_address_tokens, target_address_tokens)),
        dtype=np.int64,
        count=n,
    )
    address_length_ratio = np.fromiter(
        (length_ratio(a, b) for a, b in zip(s1_address, target_address)),
        dtype=np.float64,
        count=n,
    )

    # -- blocker evidence + provenance --------------------------------------
    provenance = frame[CANDIDATE_BLOCKERS_COLUMN].to_numpy(dtype=object) if CANDIDATE_BLOCKERS_COLUMN in frame else np.full(n, "", dtype=object)
    parsed = [parse_provenance(text) for text in provenance]
    blocker_exact = np.fromiter((p[0] for p in parsed), dtype=np.int64, count=n)
    blocker_token = np.fromiter((p[1] for p in parsed), dtype=np.int64, count=n)
    blocker_char = np.fromiter((p[2] for p in parsed), dtype=np.int64, count=n)
    blocker_dense = np.fromiter((p[3] for p in parsed), dtype=np.int64, count=n)
    n_blockers = np.fromiter((p[4] for p in parsed), dtype=np.int64, count=n)
    _bump(integrity, "unknown_blocker_labels", sum(p[5] for p in parsed))

    evidence: dict[str, np.ndarray] = {}
    for column in EVIDENCE_FLOAT_COLUMNS:
        if column in frame:
            # Blank -> NaN, which is the whole point: a blank evidence cell means
            # the blocker that measures it did not propose this pair.
            parsed_column = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            _bump(integrity, f"{column}_blank", np.isnan(parsed_column).sum())
            evidence[column] = parsed_column
        else:
            evidence[column] = np.full(n, np.nan)

    s1_candidate_count = np.fromiter(
        (counts.get(entity_id, -1) for entity_id in s1_ids), dtype=np.int64, count=n
    )
    _bump(integrity, "missing_s1_counts", (s1_candidate_count < 0).sum())

    # -- other --------------------------------------------------------------
    source_is_s2 = (source_labels == "S2").astype(np.int64)
    country_equal = np.fromiter(
        (1 if a and a == b else 0 for a, b in zip(s1_country, target_country)),
        dtype=np.int64,
        count=n,
    )
    country_missing = np.fromiter(
        (1 if not a or not b else 0 for a, b in zip(s1_country, target_country)),
        dtype=np.int64,
        count=n,
    )

    features = pd.DataFrame(
        {
            # Like address_norm_equal, both equality flags require a non-blank left
            # side: "" == "" is not evidence that two records share a name, and
            # handing the model that 1 is a spurious positive - which F0.5 punishes
            # four times harder than it punishes a miss. A blank name therefore
            # scores 0, the same as two different names.
            "name_norm_equal": np.fromiter(
                (1 if a and a == b else 0 for a, b in zip(s1_name_norm, target_name_norm)),
                dtype=np.int64,
                count=n,
            ),
            "name_key_equal": np.fromiter(
                (1 if a and a == b else 0 for a, b in zip(s1_name_key, target_name_key)),
                dtype=np.int64,
                count=n,
            ),
            "name_token_jaccard": name_token_jaccard,
            "name_token_set_ratio": token_set_ratio,
            "name_token_sort_ratio": token_sort_ratio,
            "name_partial_ratio": partial_ratio,
            "name_char3_jaccard": name_char3_jaccard,
            "name_length_ratio": name_length_ratio,
            "name_token_count_diff": name_token_count_diff,
            "name_first_token_equal": name_first_token_equal,
            "address_norm_equal": address_norm_equal,
            "address_token_jaccard": address_token_jaccard,
            "address_shared_token_count": address_shared_token_count,
            "address_length_ratio": address_length_ratio,
            "s1_address_missing": s1_address_missing,
            "target_address_missing": target_address_missing,
            "both_address_missing": both_address_missing,
            "token_df": evidence["token_df"],
            "char_jaccard": evidence["char_jaccard"],
            "dense_cosine": evidence["dense_cosine"],
            "blocker_exact_name": blocker_exact,
            "blocker_token": blocker_token,
            "blocker_char_ngram": blocker_char,
            "blocker_dense": blocker_dense,
            "n_blockers": n_blockers,
            "s1_candidate_count": s1_candidate_count,
            "source_is_s2": source_is_s2,
            "country_equal": country_equal,
            "country_missing": country_missing,
            "text_join_ok": text_join_ok.astype(np.int64),
        },
        index=frame.index,
    )

    # A failed text join leaves every TEXT-DERIVED feature meaningless. Blank those
    # rather than let a "" comparison masquerade as a measurement, and keep the row:
    # one row per candidate pair, always. Provenance, evidence and the S1 candidate
    # count come from the candidate file, not from the join, so they stay valid.
    failed = ~text_join_ok
    if failed.any():
        for column in TEXT_DERIVED_FEATURES:
            if FEATURE_DTYPES[column] == "float32":
                features.loc[failed, column] = np.nan
            else:
                features.loc[failed, column] = 0

    for column, dtype in FEATURE_DTYPES.items():
        features[column] = features[column].astype(dtype)

    ids = pd.DataFrame(
        {
            CANDIDATE_S1_COLUMN: s1_ids,
            CANDIDATE_TARGET_COLUMN: target_ids,
            CANDIDATE_SOURCE_COLUMN: source_labels,
        },
        index=frame.index,
    )
    return pd.concat([ids, features], axis=1)


class FeatureStats:
    """Per-batch statistics over the written feature matrix.

    This is the one implementation of the accumulation loop, shared by the
    single-process pass and by each worker, so the two execution paths cannot
    drift apart in what they measure. Every quantity here is order-independent -
    sums, minima, maxima - which is what lets batches be processed in any order
    and worker results be merged without a sort.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.matrix_bytes = 0
        self.missing_counts: dict[str, int] = {column: 0 for column in FEATURE_DTYPES}
        self.blank_counts: dict[str, int] = {column: 0 for column in INTEGRITY_COLUMNS}
        self.value_min: dict[str, float] = {}
        self.value_max: dict[str, float] = {}
        self.out_of_range: dict[str, int] = {}
        self.dtypes_seen: dict[str, str] = {}
        self.dtype_conflicts: dict[str, list[str]] = {}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStats":
        """Rebuild a stats object from :meth:`as_dict`, as a parent does for a worker."""
        stats = cls()
        stats.rows = int(payload.get("rows_featurized", 0))
        stats.matrix_bytes = int(payload.get("matrix_bytes", 0))
        for key in ("missing_counts", "blank_counts", "value_min", "value_max",
                    "out_of_range", "dtypes_seen"):
            setattr(stats, key, dict(payload.get(key) or {}))
        return stats

    def add(self, features: pd.DataFrame) -> None:
        """Fold one batch of featurized rows into the running totals."""
        for column in FEATURE_DTYPES:
            if column not in self.dtypes_seen:
                self.dtypes_seen[column] = str(features[column].dtype)
        for column in INTEGRITY_COLUMNS:
            if column in features:
                self.blank_counts[column] += int((features[column] == "").sum())
        for column, dtype in FEATURE_DTYPES.items():
            values = features[column].to_numpy()
            if dtype == "float32":
                self.missing_counts[column] += int(np.isnan(values).sum())
                finite = values[np.isfinite(values)]
                if finite.size:
                    self.value_min[column] = min(self.value_min.get(column, np.inf), float(finite.min()))
                    self.value_max[column] = max(self.value_max.get(column, -np.inf), float(finite.max()))
                    if column in UNIT_INTERVAL_FEATURES:
                        bad = int(np.count_nonzero((finite < 0.0) | (finite > 1.0)))
                        if bad:
                            self.out_of_range[column] = self.out_of_range.get(column, 0) + bad
            elif column == "s1_candidate_count":
                self.value_min[column] = min(self.value_min.get(column, np.inf), float(values.min()))
                self.value_max[column] = max(self.value_max.get(column, -np.inf), float(values.max()))
        self.rows += len(features)
        self.matrix_bytes += int(
            sum(features[column].to_numpy().nbytes for column in FEATURE_DTYPES)
        )

    def merge(self, other: "FeatureStats") -> None:
        """Fold another stats object (another worker's) into this one."""
        self.rows += other.rows
        self.matrix_bytes += other.matrix_bytes
        for column, count in other.missing_counts.items():
            self.missing_counts[column] = self.missing_counts.get(column, 0) + int(count)
        for column, count in other.blank_counts.items():
            self.blank_counts[column] = self.blank_counts.get(column, 0) + int(count)
        for column, value in other.out_of_range.items():
            self.out_of_range[column] = self.out_of_range.get(column, 0) + int(value)
        for column, value in other.value_min.items():
            self.value_min[column] = min(self.value_min.get(column, np.inf), value)
        for column, value in other.value_max.items():
            self.value_max[column] = max(self.value_max.get(column, -np.inf), value)
        for column, dtype in other.dtypes_seen.items():
            known = self.dtypes_seen.setdefault(column, dtype)
            if known != dtype:
                # Two workers wrote the same column as different types. Not expected
                # (both run the same casts), but a matrix whose dtype depends on
                # which worker produced the row would be read differently by the
                # trainer, so it is recorded and fails the run.
                self.dtype_conflicts[column] = [known, dtype]

    def dtype_mismatches(self) -> dict[str, list[str]]:
        """Columns whose observed dtype is not the declared one, or differed between workers."""
        mismatches = {
            column: [observed, FEATURE_DTYPES[column]]
            for column, observed in self.dtypes_seen.items()
            if observed != FEATURE_DTYPES[column]
        }
        mismatches.update(self.dtype_conflicts)
        return mismatches

    def as_dict(self) -> dict[str, Any]:
        """The accumulation fields, named as the report expects them."""
        return {
            "rows_featurized": self.rows,
            "matrix_bytes": self.matrix_bytes,
            "dtypes_seen": self.dtypes_seen,
            "missing_counts": self.missing_counts,
            "blank_counts": self.blank_counts,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "out_of_range": self.out_of_range,
        }


def extract_features(
    config: dict,
    args: argparse.Namespace,
    scan: dict[str, Any],
    output_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    """Phase 2: join the sampled rows to prepared text and write the features.

    ``--workers 1`` runs the original single-process pass; anything above it runs
    the partitioned one. Both are the same feature code - this function only picks
    the execution layer, so the caller does not need to know which ran.
    """
    if int(getattr(args, "workers", DEFAULT_WORKERS)) <= 1:
        return _extract_features_single(config, args, scan, output_dir, log)
    return _extract_features_parallel(config, args, scan, output_dir, log)


def _extract_features_single(
    config: dict,
    args: argparse.Namespace,
    scan: dict[str, Any],
    output_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    """Phase 2, single process: the original path, unchanged."""
    log.info("loading prepared text for the join")
    split = args.split
    s1_lookup = load_lookup(config, split, "source1", log)
    lookups = {
        label: load_lookup(config, split, source, log)
        for source, label in (("source2", "S2"), ("source3", "S3"))
    }
    lookup_bytes = s1_lookup.memory_bytes() + sum(item.memory_bytes() for item in lookups.values())
    log_memory(log, "after prepared lookups")
    rss_after_lookup = current_rss_bytes()
    rss_tracker = [scan.get("rss_sampled_peak", 0)]
    _sample_rss(rss_tracker)

    feature_path = output_dir / "features.tsv.partial"
    integrity = _seed_integrity()
    stats = FeatureStats()

    counts = scan["per_s1_counts"]
    sample_counts: dict[str, int] = {}
    started = time.time()
    with ChunkWriter(feature_path) as writer:
        for batch_index, frame in enumerate(
            iter_tsv(scan["sample_path"], chunksize=args.feature_batch_size), start=1
        ):
            features = build_features(frame, lookups, s1_lookup, counts, integrity)
            # Independent re-count of the sampled rows, to prove the whole-entity
            # invariant held. Cheap: one factorize per batch.
            batch_codes, batch_ids = pd.factorize(
                frame[CANDIDATE_S1_COLUMN].to_numpy(dtype=object), sort=False
            )
            batch_counts = np.bincount(batch_codes, minlength=len(batch_ids))
            for position in np.flatnonzero(batch_counts):
                entity_id = batch_ids[position]
                sample_counts[entity_id] = sample_counts.get(entity_id, 0) + int(batch_counts[position])
            stats.add(features)
            if batch_index % 10 == 0:
                log_memory(log, f"features: {fmt_int(stats.rows)} rows")
            _sample_rss(rss_tracker)
            writer.append(features)

    feature_seconds = time.time() - started
    feature_path.replace(output_dir / "features.tsv")

    # The whole-entity invariant: an entity's rows in the sample must be exactly
    # its rows in the file. A mismatch means the sample split an entity, which
    # would make s1_candidate_count wrong for it - silently, and in a feature the
    # matcher leans on. Fail loudly instead.
    mismatches = {
        entity_id: (counts.get(entity_id, 0), sample_counts[entity_id])
        for entity_id in sample_counts
        if counts.get(entity_id, 0) != sample_counts[entity_id]
    }
    extra_in_scan = set(counts) - set(sample_counts)
    integrity["count_mismatches"] = len(mismatches) + len(extra_in_scan)
    integrity["count_mismatch_examples"] = {str(k): v for k, v in list(mismatches.items())[:5]}
    if integrity["count_mismatches"]:
        # Not fatal here: the report still has to be written so the failure is
        # visible on disk. ``main`` turns a non-zero count into a non-zero exit.
        log.error(
            "whole-entity invariant violated: %s entities disagree between the scan "
            "and the sample (examples: %s)",
            fmt_int(integrity["count_mismatches"]),
            integrity["count_mismatch_examples"],
        )

    log.info(
        "  featurized %s pairs in %.1f s (%.0f pairs/s)",
        fmt_int(stats.rows),
        feature_seconds,
        stats.rows / max(feature_seconds, 1e-9),
    )
    log_memory(log, "after features")

    return _finish_feature_result(
        stats=stats,
        integrity=integrity,
        output_dir=output_dir,
        feature_path=output_dir / "features.tsv",
        feature_seconds=feature_seconds,
        lookup_bytes=lookup_bytes,
        rss_after_lookup=rss_after_lookup,
        rss_tracker=rss_tracker,
        sample_path=scan["sample_path"],
        workers=1,
    )


def _finish_feature_result(
    stats: FeatureStats,
    integrity: dict[str, Any],
    output_dir: Path,
    feature_path: Path,
    feature_seconds: float,
    lookup_bytes: int,
    rss_after_lookup: Optional[int],
    rss_tracker: list[int],
    sample_path: Path,
    workers: int,
    shard_seconds: float = 0.0,
    merge_seconds: float = 0.0,
    parallel: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the phase-2 result dict, shared by both execution paths.

    Keeping this in one place is what makes ``--workers 1`` and ``--workers N``
    comparable: every key has the same meaning in both, and the report builder
    never has to ask which path produced it.
    """
    peak = peak_rss_bytes()
    peak_value, peak_source = _peak_description(peak, rss_tracker[0])
    dtype_mismatches = stats.dtype_mismatches()
    if dtype_mismatches:
        # Not expected - both paths run the same casts - but a worker whose dtype
        # silently differed would write a matrix the trainer reads differently, so
        # it is recorded rather than assumed away.
        integrity["dtype_mismatches"] = len(dtype_mismatches)
        integrity["dtype_mismatch_examples"] = dtype_mismatches

    return {
        **stats.as_dict(),
        "feature_seconds": feature_seconds,
        "shard_seconds": shard_seconds,
        "merge_seconds": merge_seconds,
        "workers": workers,
        "integrity": integrity,
        "lookup_bytes": lookup_bytes,
        "rss_after_lookup_bytes": rss_after_lookup,
        "peak_rss_bytes": peak,
        "sampled_peak_rss_bytes": rss_tracker[0],
        "peak_rss_value": peak_value,
        "peak_rss_source": peak_source,
        "feature_path": feature_path,
        "feature_bytes": feature_path.stat().st_size,
        "sample_bytes": sample_path.stat().st_size,
        "parallel": parallel,
    }


# ---------------------------------------------------------------------------
# phase 2, parallel: partition -> shards -> workers -> deterministic merge
# ---------------------------------------------------------------------------
def partition_entities(counts: dict[str, int], workers: int) -> list[list[str]]:
    """Assign every selected S1 entity to exactly one worker, deterministically.

    The rule is one integer expression: walking the entities in the order the scan
    first saw them (``counts`` is a dict, so that order is insertion order, which
    is file order), entity *i* goes to worker ``rows_before_i * workers // total_rows``,
    clamped to the last worker. That gives four properties at once, none of which
    needs a sort of the rows later:

    * **complete and disjoint** - the cumulative sum is non-decreasing, so the
      worker index is too; every entity lands once, and no worker's set interleaves
      with another's.
    * **balanced by rows, not by entity count.** Entities differ by an order of
      magnitude in candidate count, so splitting the entity *list* evenly would
      leave one worker holding most of the work; the cumulative-row rule balances
      the actual work.
    * **reproducible** - integer arithmetic on the scan's own output, so the same
      sample always yields the same partition, whatever the platform.
    * **contiguous in file order**, which is what lets the merge be a plain
      concatenation in worker order (see :func:`merge_features`).

    Fewer entities than workers leaves the extra workers with an empty list, which
    the worker path handles as an empty partition rather than an error.
    """
    entities = list(counts)
    assignments: list[list[str]] = [[] for _ in range(max(int(workers), 1))]
    if not entities:
        return assignments
    total = int(sum(int(value) for value in counts.values()))
    if len(assignments) == 1 or total <= 0:
        assignments[0].extend(entities)
        return assignments
    cumulative = 0
    for entity_id in entities:
        index = min(len(assignments) - 1, (cumulative * len(assignments)) // total)
        assignments[index].append(entity_id)
        cumulative += int(counts[entity_id])
    return assignments


def write_shards(
    sample_path: Path,
    assignments: list[list[str]],
    output_dir: Path,
    chunksize: int,
    log: logging.Logger,
) -> dict[str, Any]:
    """Split the sampled rows into one file per worker, in file order.

    Rows are appended as they are read, so a shard holds its entities' rows in the
    order they appeared in the sample - not grouped, not reordered. That is what
    makes the merged output byte-identical to the single-process output rather
    than merely the same set of rows.

    A row whose S1 is not in the partition (which cannot happen from this script's
    own scan, but can if the sample file was edited) goes to worker 0 and is
    counted in ``unassigned_rows``, so the whole-entity check downstream reports it
    instead of a ``KeyError`` here.
    """
    worker_dir = ensure_dir(output_dir / WORKER_DIR_NAME)
    paths = [worker_dir / f"shard_{index:02d}.tsv" for index in range(len(assignments))]
    worker_of = {
        entity_id: index for index, group in enumerate(assignments) for entity_id in group
    }
    rows_per_worker = [0] * len(assignments)
    unassigned_rows = 0

    started = time.time()
    with ExitStack() as stack:
        writers = [stack.enter_context(ChunkWriter(path)) for path in paths]
        for frame in iter_tsv(sample_path, chunksize=chunksize):
            s1_ids = frame[CANDIDATE_S1_COLUMN].to_numpy(dtype=object)
            codes = np.fromiter(
                (worker_of.get(entity_id, -1) for entity_id in s1_ids),
                dtype=np.int64,
                count=len(s1_ids),
            )
            missing = int((codes < 0).sum())
            if missing:
                unassigned_rows += missing
                codes[codes < 0] = 0
            if not codes.size:
                continue
            for index in np.unique(codes):
                mask = codes == index
                writers[int(index)].append(frame.loc[mask])
                rows_per_worker[int(index)] += int(mask.sum())
    seconds = time.time() - started

    if unassigned_rows:
        log.warning(
            "%s sampled rows reference an S1 entity the scan did not count; they were "
            "assigned to worker 0 and will be reported by the whole-entity check",
            fmt_int(unassigned_rows),
        )
    return {
        "shard_dir": worker_dir,
        "shard_paths": paths,
        "rows_per_worker": rows_per_worker,
        "unassigned_rows": unassigned_rows,
        "shard_seconds": seconds,
    }


def _collect_needed_ids(
    shard_path: Path, chunksize: int
) -> tuple[set[str], dict[str, set[str]]]:
    """The ids one shard can join to: its S1 entities, and its targets per source.

    Read *before* the prepared files are, so a worker can load only the text its
    own shard will look up. This is a second read of the shard - one extra pass
    over the sample, which is a fraction of the prepared corpus - and it is what
    keeps W workers from each holding a full copy of the prepared text.
    """
    s1_ids: set[str] = set()
    targets: dict[str, set[str]] = {}
    columns = [CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN, CANDIDATE_SOURCE_COLUMN]
    for frame in iter_tsv(shard_path, columns=columns, chunksize=chunksize):
        s1_ids.update(frame[CANDIDATE_S1_COLUMN].unique().tolist())
        grouped = frame.groupby(CANDIDATE_SOURCE_COLUMN, sort=True)[CANDIDATE_TARGET_COLUMN]
        for label, column in grouped:
            targets.setdefault(str(label), set()).update(column.unique().tolist())
    return s1_ids, targets


def _empty_worker_result(index: int, feature_path: Path) -> dict[str, Any]:
    """A worker that had no rows to featurize. Not an error, and not a special case
    downstream: it returns the same shape as a worker that did work."""
    return {
        "index": index,
        "rows_featurized": 0,
        "feature_seconds": 0.0,
        "lookup_bytes": 0,
        "rss_after_lookup_bytes": None,
        "peak_rss_value": None,
        "peak_rss_source": "unavailable (empty partition)",
        "feature_path": str(feature_path),
        "feature_bytes": 0,
        "stats": FeatureStats().as_dict(),
        "integrity": _seed_integrity(),
        "count_mismatches": 0,
        "count_mismatch_examples": {},
        "n_s1_entities": 0,
        "n_target_ids": {},
    }


def _feature_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Featurize one shard. Runs in a child process.

    The payload is plain data only - paths, the config dict, and this worker's own
    subset of the S1 counts - because workers are started with ``spawn``: nothing
    is inherited, so the RSS this worker reports is its own and the measurement
    means what it says.
    """
    index = int(payload["index"])
    shard_path = Path(payload["shard_path"])
    feature_path = Path(payload["feature_path"])
    log = setup_logging(
        f"{LOG_NAME}_w{index:02d}",
        log_dir=payload["log_dir"],
        level=getattr(logging, str(payload.get("log_level", "INFO")).upper(), logging.INFO),
    )
    log.info("worker %d: shard %s", index, shard_path.name)

    if not shard_path.is_file():
        # Fewer sampled entities than workers, so this partition is empty. There is
        # nothing to read and nothing to featurize; writing no feature file is the
        # correct outcome, and the merge skips it.
        log.info("worker %d: empty partition, nothing to featurize", index)
        return _empty_worker_result(index, feature_path)

    config = payload["config"]
    split = payload["split"]
    worker_counts: dict[str, int] = payload["counts"]
    chunksize = int(payload["chunksize"])
    batch_size = int(payload["feature_batch_size"])

    s1_ids, targets = _collect_needed_ids(shard_path, chunksize)
    log.info(
        "worker %d: %s S1 entities, %s target ids across %s",
        index,
        fmt_int(len(worker_counts)),
        fmt_int(sum(len(ids) for ids in targets.values())),
        ", ".join(f"{label}={fmt_int(len(ids))}" for label, ids in sorted(targets.items())) or "no sources",
    )
    s1_lookup = load_lookup(config, split, "source1", log, keep_ids=s1_ids)
    lookups = {
        label: load_lookup(config, split, source, log, keep_ids=targets.get(label, set()))
        for source, label in (("source2", "S2"), ("source3", "S3"))
    }
    lookup_bytes = s1_lookup.memory_bytes() + sum(item.memory_bytes() for item in lookups.values())
    log_memory(log, f"worker {index} after prepared lookups")
    rss_after_lookup = current_rss_bytes()
    rss_tracker = [0]
    _sample_rss(rss_tracker)

    integrity = _seed_integrity()
    stats = FeatureStats()
    sample_counts: dict[str, int] = {}
    started = time.time()
    with ChunkWriter(feature_path) as writer:
        for batch_index, frame in enumerate(iter_tsv(shard_path, chunksize=batch_size), start=1):
            features = build_features(frame, lookups, s1_lookup, worker_counts, integrity)
            batch_codes, batch_ids = pd.factorize(
                frame[CANDIDATE_S1_COLUMN].to_numpy(dtype=object), sort=False
            )
            batch_counts = np.bincount(batch_codes, minlength=len(batch_ids))
            for position in np.flatnonzero(batch_counts):
                entity_id = batch_ids[position]
                sample_counts[entity_id] = sample_counts.get(entity_id, 0) + int(batch_counts[position])
            stats.add(features)
            if batch_index % 10 == 0:
                log_memory(log, f"worker {index}: {fmt_int(stats.rows)} rows")
            _sample_rss(rss_tracker)
            writer.append(features)
    feature_seconds = time.time() - started

    # The whole-entity check, for this worker's own entities. Every entity in the
    # partition belongs to exactly one worker, so summing these over workers gives
    # the same total the single-process pass computes over the whole sample.
    mismatches = {
        entity_id: (worker_counts.get(entity_id, 0), written)
        for entity_id, written in sample_counts.items()
        if worker_counts.get(entity_id, 0) != written
    }
    extra_in_partition = sum(1 for entity_id in worker_counts if entity_id not in sample_counts)

    peak = peak_rss_bytes()
    peak_value, peak_source = _peak_description(peak, rss_tracker[0])
    log.info(
        "worker %d: featurized %s pairs in %.1f s (%.0f pairs/s)",
        index,
        fmt_int(stats.rows),
        feature_seconds,
        stats.rows / max(feature_seconds, 1e-9),
    )
    log_memory(log, f"worker {index} after features")

    return {
        "index": index,
        "rows_featurized": stats.rows,
        "feature_seconds": feature_seconds,
        "lookup_bytes": lookup_bytes,
        "rss_after_lookup_bytes": rss_after_lookup,
        "peak_rss_bytes": peak,
        "sampled_peak_rss_bytes": rss_tracker[0],
        "peak_rss_value": peak_value,
        "peak_rss_source": peak_source,
        "feature_path": str(feature_path),
        "feature_bytes": feature_path.stat().st_size if feature_path.is_file() else 0,
        "stats": stats.as_dict(),
        "integrity": integrity,
        "count_mismatches": len(mismatches) + extra_in_partition,
        "count_mismatch_examples": {str(k): v for k, v in list(mismatches.items())[:5]},
        "n_s1_entities": len(worker_counts),
        "n_target_ids": {label: len(ids) for label, ids in sorted(targets.items())},
    }


def merge_features(paths: Sequence[Path], target: Path) -> int:
    """Concatenate worker feature files in worker order; return the byte count.

    Worker order, never completion order: because the partition is contiguous in
    file order, concatenating shard 0..N-1 keeps every worker's rows in their
    original relative order, so nothing about the output depends on which worker
    finished first. Each file after the first contributes its rows but not its
    header.

    When the candidate file groups a pair's rows under its S1 entity - which the
    generator guarantees ("groups stay contiguous in the output") - the partition's
    contiguous blocks line up with those groups and the concatenation reproduces
    the sample file's row order exactly. If a candidate file were ever not grouped
    that way, the merged file would hold the same rows in a different order: the
    *values* are identical either way, which is what
    ``test_merged_workers2_equals_workers1`` sorts on to check.

    Copied as bytes, so the merge costs no memory regardless of how large the
    matrix is.
    """
    with open(target, "w", encoding="utf-8", newline="") as out:
        wrote_header = False
        for path in paths:
            path = Path(path)
            if not path.is_file() or path.stat().st_size == 0:
                continue
            with open(path, "r", encoding="utf-8", newline="") as handle:
                header = handle.readline()
                if not wrote_header and header:
                    out.write(header)
                    wrote_header = True
                shutil.copyfileobj(handle, out, length=1 << 20)
    return target.stat().st_size


def _extract_features_parallel(
    config: dict,
    args: argparse.Namespace,
    scan: dict[str, Any],
    output_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    """Phase 2 across ``--workers N`` processes, then a deterministic merge."""
    workers = int(args.workers)
    counts: dict[str, int] = scan["per_s1_counts"]
    chunksize = args.chunksize or int(config.get("io", {}).get("chunksize", 500_000))

    log.info("workers=%d", workers)
    log.info("selected S1 entities=%s", fmt_int(len(counts)))

    assignments = partition_entities(counts, workers)
    for index, group in enumerate(assignments):
        log.info(
            "worker %d: %s S1 entities (%s candidate pairs)",
            index,
            fmt_int(len(group)),
            fmt_int(sum(int(counts[entity_id]) for entity_id in group)),
        )

    shards = write_shards(scan["sample_path"], assignments, output_dir, chunksize, log)
    worker_dir: Path = shards["shard_dir"]
    log.info(
        "  wrote %d shards in %.1f s: %s",
        len(shards["shard_paths"]),
        shards["shard_seconds"],
        ", ".join(fmt_int(rows) for rows in shards["rows_per_worker"]),
    )

    payloads = [
        {
            "index": index,
            "shard_path": str(shards["shard_paths"][index]),
            "feature_path": str(worker_dir / f"worker_{index:02d}_features.tsv"),
            "config": config,
            "split": args.split,
            "chunksize": chunksize,
            "feature_batch_size": args.feature_batch_size,
            "counts": {entity_id: int(counts[entity_id]) for entity_id in assignments[index]},
            "log_dir": str(worker_dir),
            "log_level": args.log_level,
        }
        for index in range(workers)
    ]

    log.info("starting %d workers (spawn)", workers)
    results: dict[int, dict[str, Any]] = {}
    started = time.time()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(_feature_worker, payload): payload["index"] for payload in payloads}
        # Collected by worker index, not in completion order: the merge below reads
        # the results in worker order, so two runs of the same command produce the
        # same bytes however the workers happened to finish.
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                log.error("worker %d failed: %s", index, exc)
                for pending in futures:
                    pending.cancel()
                raise
    worker_wall_seconds = time.time() - started
    ordered = [results[index] for index in sorted(results)]

    for result in ordered:
        log.info(
            "worker %d: %s pairs, %s S1 entities, %.1f s, %.0f pairs/s, peak RSS %s, lookup ~%s",
            result["index"],
            fmt_int(result["rows_featurized"]),
            fmt_int(result["n_s1_entities"]),
            result["feature_seconds"],
            result["rows_featurized"] / max(result["feature_seconds"], 1e-9),
            human_bytes(result["peak_rss_value"] or 0),
            human_bytes(result["lookup_bytes"]),
        )

    merge_started = time.time()
    partial = output_dir / "features.tsv.partial"
    merge_features([Path(result["feature_path"]) for result in ordered], partial)
    merge_seconds = time.time() - merge_started
    partial.replace(output_dir / "features.tsv")

    stats = FeatureStats()
    integrity = _seed_integrity()
    for result in ordered:
        stats.merge(FeatureStats.from_dict(result["stats"]))
        for key, value in result["integrity"].items():
            if key == "rapidfuzz_available":
                # Environment, not work: any worker that had rapidfuzz had it.
                integrity[key] = max(int(integrity.get(key, 0)), int(value))
            else:
                _bump(integrity, key, value)

    integrity["count_mismatches"] = sum(int(r["count_mismatches"]) for r in ordered)
    examples: dict[str, Any] = {}
    for result in ordered:
        for key, value in result["count_mismatch_examples"].items():
            if len(examples) < 5:
                examples[key] = value
    integrity["count_mismatch_examples"] = examples
    if integrity["count_mismatches"]:
        log.error(
            "whole-entity invariant violated: %s entities disagree between the scan "
            "and the sample (examples: %s)",
            fmt_int(integrity["count_mismatches"]),
            integrity["count_mismatch_examples"],
        )

    log.info(
        "  featurized %s pairs in %.1f s across %d workers (%.0f pairs/s aggregate)",
        fmt_int(stats.rows),
        worker_wall_seconds,
        workers,
        stats.rows / max(worker_wall_seconds, 1e-9),
    )

    peaks = [int(r["peak_rss_value"]) for r in ordered if r["peak_rss_value"]]
    lookup_per_worker = [int(r["lookup_bytes"]) for r in ordered]
    worker_seconds = sum(float(r["feature_seconds"]) for r in ordered)
    parallel_phase_seconds = (
        float(shards["shard_seconds"]) + worker_wall_seconds + merge_seconds
    )
    parallel = {
        "workers": workers,
        "shard_dir": str(worker_dir),
        "shard_seconds": round(float(shards["shard_seconds"]), 2),
        "worker_wall_seconds": round(worker_wall_seconds, 2),
        "merge_seconds": round(merge_seconds, 2),
        "parallel_phase_seconds": round(parallel_phase_seconds, 2),
        "n_s1_entities_per_worker": [int(r["n_s1_entities"]) for r in ordered],
        "rows_per_worker": [int(r["rows_featurized"]) for r in ordered],
        "shard_rows_per_worker": [int(rows) for rows in shards["rows_per_worker"]],
        "feature_seconds_per_worker": [round(float(r["feature_seconds"]), 2) for r in ordered],
        "lookup_bytes_per_worker": lookup_per_worker,
        "peak_rss_bytes_per_worker": peaks,
        "peak_rss_source": (
            "max across worker processes (resource.getrusage high-water mark)"
        ),
        # The number that decides how many workers fit on the node: every worker
        # holds its own prepared text while it runs, so the node needs the SUM.
        "total_lookup_bytes": sum(lookup_per_worker),
        "total_lookup_size": human_bytes(sum(lookup_per_worker)),
        "total_peak_rss_bytes": sum(peaks) if peaks else None,
        "total_peak_rss": human_bytes(sum(peaks)) if peaks else None,
        "max_worker_lookup_bytes": max(lookup_per_worker) if lookup_per_worker else 0,
        "worker_cpu_seconds_total": round(worker_seconds, 2),
        # <= 1: how much of the requested parallelism the pool actually used.
        "worker_utilization": round(
            worker_seconds / max(worker_wall_seconds * workers, 1e-9), 3
        ),
        "unassigned_rows": int(shards["unassigned_rows"]),
        "s1_counts_entries": int(len(counts)),
        "merge_order": "worker index (not completion order)",
        "notes": [
            "each worker reads the whole prepared file to filter it, so the prepared "
            "corpus is read once per worker (payload resident: total_lookup_size); "
            "the repeated read is bounded by the page cache, the repeated payload is "
            "not, which is why the lookup is filtered to the ids each shard needs",
            "compare total_lookup_size here with memory.prepared_lookup_estimate from "
            "a --workers 1 run of the same command to see the per-node saving",
            "the per-worker S1 counts in the pool payload are the scan's own counts "
            "split by partition, so the parent transiently holds about two copies of "
            "them (s1_counts_entries entries) - the whole-entity check needs the "
            "file-wide count, not the shard's, which is why they are passed at all",
        ],
    }

    result = _finish_feature_result(
        stats=stats,
        integrity=integrity,
        output_dir=output_dir,
        feature_path=output_dir / "features.tsv",
        feature_seconds=worker_wall_seconds,
        lookup_bytes=sum(lookup_per_worker),
        rss_after_lookup=max(
            (int(r["rss_after_lookup_bytes"]) for r in ordered if r["rss_after_lookup_bytes"]),
            default=None,
        ),
        rss_tracker=[max(peaks) if peaks else 0],
        sample_path=scan["sample_path"],
        workers=workers,
        shard_seconds=float(shards["shard_seconds"]),
        merge_seconds=merge_seconds,
        parallel=parallel,
    )
    # ``peak_rss_value`` is the largest *single* process's high-water mark (the
    # parent or one worker) - what a per-process memory limit is checked against,
    # not what the node needs. The node total is parallel.total_peak_rss_bytes, and
    # the two differ by roughly the worker count, so which one this is says so.
    result["peak_rss_source"] = (
        f"max over the parent and {workers} worker processes "
        f"({result['peak_rss_source']}); this is one process, not the node total - "
        f"see parallel.total_peak_rss"
    )
    return result


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def summarize(
    features: dict[str, Any],
    scan: dict[str, Any],
    total_seconds: float,
    log: logging.Logger,
) -> dict[str, Any]:
    """Assemble the report, including the full-scale extrapolation."""
    rows = features["rows_featurized"]
    per_row_bytes = features["matrix_bytes"] / max(rows, 1)
    per_row_tsv = features["feature_bytes"] / max(rows, 1)
    workers = int(features.get("workers", 1))
    shard_seconds = float(features.get("shard_seconds", 0.0))
    merge_seconds = float(features.get("merge_seconds", 0.0))

    # The scan cost is measured on the whole file already - phase 1 reads every
    # row, whatever the worker count - so it needs no scaling. Only the feature
    # phase is extrapolated, and it is extrapolated from what the workers actually
    # did: feature_seconds is the wall clock of the worker pool, so multiplying it
    # by the corpus/sample ratio projects *this* worker count, measured, rather
    # than a theoretical one. The shard split and the merge scale with the sample
    # too, so they are carried into the same projection.
    # Unless --limit-rows stopped the scan early, in which case the scan is scaled
    # too, and the report says the scan timing is an extrapolation rather than a
    # measurement.
    parallel_phase_seconds = features["feature_seconds"] + shard_seconds + merge_seconds
    parallel_phase_full = parallel_phase_seconds * (FULL_CANDIDATE_PAIRS / max(rows, 1))
    # "The same phase done by one process." At workers=1 that is the measured figure
    # itself; above it, it is the parallel wall clock times the worker count, which
    # is the definition of linear scaling and is therefore flagged as extrapolated
    # rather than reported as a measurement.
    single_equivalent_full = parallel_phase_full * workers
    if scan["limited"]:
        scan_full = scan["scan_seconds"] * (FULL_CANDIDATE_PAIRS / max(scan["rows_scanned"], 1))
    else:
        scan_full = scan["scan_seconds"]
    total_single = scan_full + single_equivalent_full
    total_parallel = scan_full + parallel_phase_full
    # Extrapolating linearly from the measured worker count to 48 is exactly the
    # assumption this experiment exists to replace, so it is quarantined in its own
    # clearly-labelled block rather than mixed in with the measured numbers.
    theoretical_48 = scan_full + parallel_phase_full * (THEORETICAL_HPC_WORKERS / max(workers, 1))
    scan_was_measured = not scan["limited"]
    matrix_full = per_row_bytes * FULL_CANDIDATE_PAIRS
    tsv_full = per_row_tsv * FULL_CANDIDATE_PAIRS

    # 540 GB is the HPC node's RAM; the matrix has to fit with room for the
    # lookups, one batch and the model that follows it.
    fits_in_ram = matrix_full < 0.25 * 540 * 1024**3

    # A per-pair cost measured on a handful of rows is noise. Say so rather than
    # let a linear extrapolation from it be quoted as a projection.
    if rows >= 1_000_000:
        confidence = "high (>=1M sampled pairs)"
    elif rows >= 100_000:
        confidence = "moderate (100k-1M sampled pairs)"
    else:
        confidence = (
            f"low ({fmt_int(rows)} sampled pairs): the per-pair cost is not stable "
            f"at this size, so the projections below are indicative only"
        )

    report = {
        "sample": {
            "n_s1_entities_sampled": scan["n_s1_entities_sampled"],
            "n_s1_entities_seen": scan["n_s1_entities_seen"],
            "n_candidate_pairs_sampled": rows,
            "rows_scanned": scan["rows_scanned"],
            "sample_fraction": scan["sample_fraction"],
            "sample_candidates_file_rows": rows,
        },
        "timing": {
            "scan_seconds": round(scan["scan_seconds"], 2),
            "feature_seconds": round(features["feature_seconds"], 2),
            "shard_seconds": round(shard_seconds, 2),
            "merge_seconds": round(merge_seconds, 2),
            "parallel_phase_seconds": round(parallel_phase_seconds, 2),
            "total_seconds": round(total_seconds, 2),
            "scan_rows_per_sec": round(scan["rows_scanned"] / max(scan["scan_seconds"], 1e-9), 1),
            "feature_pairs_per_sec": round(rows / max(features["feature_seconds"], 1e-9), 1),
        },
        "memory": {
            "peak_rss_bytes": features["peak_rss_value"],
            "peak_rss": human_bytes(features["peak_rss_value"] or 0),
            "peak_rss_source": features["peak_rss_source"],
            "rss_after_prepared_lookups": human_bytes(features["rss_after_lookup_bytes"] or 0),
            "prepared_lookup_estimate": human_bytes(features["lookup_bytes"]),
            "n_s1_id_set_bytes": scan["n_s1_entity_id_set_bytes"],
        },
        "features": {
            "n_columns": len(FEATURE_DTYPES),
            "n_features": len(FEATURE_DTYPES) - len(NON_FEATURE_COLUMNS),
            "non_feature_columns": list(NON_FEATURE_COLUMNS),
            "dtypes": features["dtypes_seen"],
            "matrix_bytes": features["matrix_bytes"],
            "matrix_row_bytes": round(per_row_bytes, 2),
            "matrix_size": human_bytes(features["matrix_bytes"]),
            "missingness": {
                column: {
                    "rate": round(
                        features["missing_counts"].get(column, 0) / max(rows, 1), 6
                    ),
                    "count": int(features["missing_counts"].get(column, 0)),
                    "dtype": features["dtypes_seen"].get(column),
                    "min": features["value_min"].get(column),
                    "max": features["value_max"].get(column),
                    "out_of_unit_range": int(features["out_of_range"].get(column, 0)),
                }
                for column in FEATURE_DTYPES
            },
        },
        "integrity": {
            **features["integrity"],
            "duplicate_sampled_pairs": scan["duplicate_sampled_pairs"],
            "duplicate_rate": round(
                scan["duplicate_sampled_pairs"] / max(rows, 1), 8
            ),
            "blank_id_columns": features["blank_counts"],
            "matrix_columns": list(FEATURE_DTYPES),
        },
        "outputs": {
            "sample_candidates": {
                "path": str(scan["sample_path"]),
                "bytes": features["sample_bytes"],
                "size": human_bytes(features["sample_bytes"]),
            },
            "features": {
                "path": str(features["feature_path"]),
                "bytes": features["feature_bytes"],
                "size": human_bytes(features["feature_bytes"]),
            },
        },
        "extrapolation": {
            "full_candidate_pairs": FULL_CANDIDATE_PAIRS,
            "assumed_workers": workers,
            "confidence": confidence,
            "scan_seconds_measured_on_full_file": round(scan["scan_seconds"], 2),
            "scan_seconds_full_scale": round(scan_full, 2),
            "scan_timing_is_a_measurement": scan_was_measured,
            "parallel_phase_seconds_measured": round(parallel_phase_seconds, 2),
            "parallel_phase_seconds_full_scale": round(parallel_phase_full, 1),
            "single_process_equivalent": {
                "is_extrapolated": workers > 1,
                "note": (
                    "measured, not extrapolated: this run used one process"
                    if workers <= 1
                    else "EXTRAPOLATED: the measured parallel phase scaled by the worker "
                    "count, which assumes the phase scales linearly. Compare a --workers 1 "
                    "run of the same command to measure the real single-process figure."
                ),
                "parallel_phase_equivalent_seconds_full_scale": round(single_equivalent_full, 1),
                "total_seconds_full_scale": round(total_single, 1),
            },
            "total_hours_at_assumed_workers": round(total_parallel / 3600.0, 2),
            "workers_are_measured": True,
            "theoretical_48_workers": {
                "is_theoretical": True,
                "note": (
                    "UNMEASURED. This assumes the feature phase scales linearly from the "
                    f"{workers} worker(s) measured here to {THEORETICAL_HPC_WORKERS}. Run "
                    f"--workers {THEORETICAL_HPC_WORKERS} to measure it instead of "
                    "assuming it."
                ),
                "total_seconds": round(theoretical_48, 1),
                "total_hours": round(theoretical_48 / 3600.0, 2),
            },
            "matrix_bytes_full": int(matrix_full),
            "matrix_size_full": human_bytes(matrix_full),
            "features_tsv_bytes_full": int(tsv_full),
            "features_tsv_size_full": human_bytes(tsv_full),
            "matrix_fits_in_ram": bool(fits_in_ram),
            "assumptions": [
                f"the projection is measured at {workers} worker(s): total_seconds_at_"
                "assumed_workers extrapolates the worker phase's own wall clock, not a "
                "per-worker ideal",
                "the per-pair cost depends on name length, and this sample's names may be "
                "shorter (cheaper) than the real ones, so the throughput here is an upper "
                "bound on the real run",
                "the scan runs once, in the parent, whatever --workers is; only the "
                "feature phase is parallel",
                "the feature phase scales with workers only as far as memory bandwidth "
                "allows, and each worker holds its own prepared-text lookup - see "
                "parallel.total_lookup_size for what the node needs at this worker count",
            ] + (
                []
                if scan_was_measured
                else [
                    "the scan timing is NOT a measurement: --limit-rows stopped the scan "
                    "early, so scan_seconds_full_scale is a linear extrapolation from a "
                    "prefix of the file",
                ]
            ),
        },
    }
    if features.get("parallel"):
        report["parallel"] = features["parallel"]

    log.info("=" * 78)
    log.info("workers: %d", workers)
    log.info("sample : %s S1 entities, %s candidate pairs", fmt_int(report["sample"]["n_s1_entities_sampled"]), fmt_int(rows))
    log.info("scan   : %.1f s (%.0f rows/s) for %s rows", scan["scan_seconds"], report["timing"]["scan_rows_per_sec"], fmt_int(scan["rows_scanned"]))
    log.info("features: %.1f s (%.0f pairs/s)", features["feature_seconds"], report["timing"]["feature_pairs_per_sec"])
    log.info("peak RSS: %s", report["memory"]["peak_rss"])
    log.info(
        "full %s pairs: ~%.2f h at %d workers (measured phase), matrix %s",
        fmt_int(FULL_CANDIDATE_PAIRS),
        report["extrapolation"]["total_hours_at_assumed_workers"],
        workers,
        report["extrapolation"]["matrix_size_full"],
    )
    log.info("=" * 78)
    return report


def write_missingness_csv(report: dict[str, Any], path: Path) -> None:
    """Per-feature missingness as a CSV, for eyeballing."""
    rows = [
        {
            "feature": column,
            "dtype": entry["dtype"],
            "missing_rate": entry["rate"],
            "missing_count": entry["count"],
            "min": entry["min"],
            "max": entry["max"],
            "out_of_unit_range": entry["out_of_unit_range"],
        }
        for column, entry in report["features"]["missingness"].items()
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# worker count
# ---------------------------------------------------------------------------
def available_cpu_count() -> int:
    """CPUs this process may actually use, not the CPUs the node has.

    On a scheduler-allocated node ``os.cpu_count()`` reports the whole machine
    while the allocation is usually a subset of it, so the affinity mask is the
    honest upper bound for ``--workers`` - and getting it wrong would let a job
    oversubscribe its own allocation. Platforms without CPU affinity (Windows)
    fall back to the logical count.
    """
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            return max(1, len(getter(0)))
        except OSError:  # pragma: no cover - defensive
            pass
    return max(1, os.cpu_count() or 1)


def validate_workers(parser: argparse.ArgumentParser, workers: int) -> int:
    """Reject a worker count outside ``1 .. available_cpu_count()``.

    Rejected rather than clamped: silently running 4 workers when 10 were asked
    for would make a benchmark report a number nobody asked for. The upper bound
    is read here rather than from a constant, because the whole point is that the
    allocation differs between the login node and a job.
    """
    if workers < 1:
        parser.error(
            f"--workers must be >= 1 (got {workers}); use --workers 1 for the "
            f"original single-process path"
        )
    limit = available_cpu_count()
    if workers > limit:
        parser.error(
            f"--workers {workers} exceeds the {limit} CPU(s) available to this "
            f"process (an affinity/scheduler allocation, not the node's core count)"
        )
    return workers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3 de-risk: sampled pair features for the matcher."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--candidates", default="candidate_pairs", help="candidate file stem")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.03,
        help="fraction of the --population S1 entities to sample (whole entities). "
        "Must be 1.0 for --split test: every test entity needs its features",
    )
    parser.add_argument(
        "--population",
        choices=POPULATIONS,
        default=None,
        help="which S1 entities may be sampled. Default: all (train: every labelled "
        "entity, so the matcher trains on all of them; test: forced to all). "
        "val reproduces the original val-only de-risk sample",
    )
    parser.add_argument("--chunksize", type=int, default=None, help="candidate rows per chunk")
    parser.add_argument("--feature-batch-size", type=int, default=200_000, help="rows per feature batch")
    parser.add_argument("--limit-rows", type=int, default=None, help="max candidate rows to scan (smoke test)")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="feature-extraction worker processes (default 1 = the original "
        "single-process path, unchanged). Above 1, the selected S1 entities are "
        "partitioned into contiguous row-balanced shards, one per worker; the "
        "sample itself is chosen once, in the parent, so --workers never changes "
        "which entities are selected. Validated against the CPUs available to this "
        "process (the scheduler allocation, not the node's core count)",
    )
    parser.add_argument(
        "--cleanup-shards",
        action="store_true",
        help="delete <output-dir>/workers/ (shards, per-worker logs, per-worker "
        "feature files) after a successful merge. Off by default: the shards are "
        "what makes a worker disagreement traceable",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="experiment directory (default: <candidates_dir>/../experiments/"
        "step3_features for train, .../step3_features_test for test)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    validate_workers(parser, args.workers)
    try:
        args.population = resolve_population(args.split, args.population)
    except ValueError as error:
        parser.error(str(error))
    if args.split == "test" and args.sample_fraction < 1.0:
        parser.error(
            f"--split test needs --sample-fraction 1.0 (got {args.sample_fraction}): an "
            "unsampled test entity has no features and would be emitted with no match"
        )
    return args


def default_output_dir(config: dict, split: str) -> Path:
    """Per-split experiment directory, so a test run never overwrites train features."""
    name = "step3_features" if split == "train" else f"step3_features_{split}"
    return Path(config["resolved"]["candidates_dir"]).parent / "experiments" / name


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir}
    )
    # Keep the experiment in its own directory next to (never inside) the
    # production outputs, so nothing here can be mistaken for a pipeline artifact.
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = default_output_dir(config, args.split)
    ensure_dir(output_dir)

    log = setup_logging(
        LOG_NAME,
        log_dir=output_dir,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    set_seed(config.get("project", {}).get("seed", 42))

    log.info("=" * 78)
    log.info("extract_pair_features: Step 3 de-risk (sampled pair features)")
    log.info(describe_environment(config))
    log.info("output_dir=%s", output_dir)
    log.info("=" * 78)

    started = time.time()
    scan = scan_and_sample(config, args, output_dir, log)
    try:
        features = extract_features(config, args, scan, output_dir, log)
    except Exception:
        # A worker that died (OOM, a bad shard, a pickling error) must leave a
        # traceback in the log and a non-zero exit, not a half-written report that
        # looks like a completed run.
        log.exception("feature extraction failed; no report was written")
        return 1
    report = summarize(features, scan, time.time() - started, log)

    report["inputs"] = {
        "candidates": scan["source_path"],
        "split": args.split,
        "population": args.population,
        "sample_fraction": args.sample_fraction,
        "workers": int(args.workers),
        "chunksize": args.chunksize,
        "feature_batch_size": args.feature_batch_size,
        "limit_rows": args.limit_rows,
        "read_columns": scan["read_columns"],
        "evidence_columns": scan["evidence_columns"],
        "config_path": config.get("config_path"),
    }
    write_json(output_dir / "step3_features_report.json", report)
    write_missingness_csv(report, output_dir / "feature_missingness.csv")

    # The end-of-run block the benchmark is read from: workers, sample size, pairs,
    # rows, elapsed, memory and throughput in one place.
    log.info("-" * 78)
    log.info("workers          : %d", report["inputs"]["workers"])
    log.info("selected S1      : %s", fmt_int(report["sample"]["n_s1_entities_sampled"]))
    log.info(
        "candidate pairs  : %s sampled / %s scanned",
        fmt_int(report["sample"]["n_candidate_pairs_sampled"]),
        fmt_int(report["sample"]["rows_scanned"]),
    )
    log.info("feature rows     : %s", fmt_int(features["rows_featurized"]))
    log.info("elapsed          : %.1f s", report["timing"]["total_seconds"])
    log.info("peak RSS         : %s", report["memory"]["peak_rss"])
    log.info("throughput       : %.0f pairs/s", report["timing"]["feature_pairs_per_sec"])
    if report.get("parallel"):
        log.info(
            "node memory      : %s across %d workers (lookups %s)",
            report["parallel"]["total_peak_rss"] or "unavailable",
            report["parallel"]["workers"],
            report["parallel"]["total_lookup_size"],
        )
    log.info("-" * 78)
    log.info("report: %s", output_dir / "step3_features_report.json")

    if args.cleanup_shards and report.get("parallel"):
        shutil.rmtree(report["parallel"]["shard_dir"], ignore_errors=True)
        log.info("removed shard directory %s", report["parallel"]["shard_dir"])

    if features["out_of_range"]:
        log.error("unit-interval features outside [0, 1]: %s", features["out_of_range"])
        return 1
    if features["integrity"].get("count_mismatches"):
        return 1
    if features["integrity"].get("dtype_mismatches"):
        log.error("worker output dtypes differ from the declaration: %s",
                  features["integrity"].get("dtype_mismatch_examples"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
