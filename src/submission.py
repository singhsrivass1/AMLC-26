"""The graded artifact: ``matching_results.tsv``, written, validated and scored.

Format (mirrors the training ground truth, which is the only format evidence the
challenge provides)::

    source1_entity_id<TAB>matched_entity_ids
    S1-965667<TAB>S2-166376419,S3-2231
    S1-12<TAB>

Rules this module enforces, because each one is a way to lose the leaderboard:

* **every S1 entity exactly once**, in S1 file order - including every entity the
  blockers proposed nothing for, which never appears in a candidate or feature file;
* **a singleton is an exact empty string** - the line is ``S1-12\\t`` and nothing
  else: never ``nan``, ``None``, ``[]`` or a missing row;
* matched ids are valid ``S2-``/``S3-`` ids, deduplicated and sorted, so reruns are
  byte-identical.

The file is written by hand, not through ``DataFrame.to_csv``: the empty-string
rule is the one thing that must not depend on a library's NA handling, and a
hand-written line cannot acquire quotes, an index column or a ``nan``.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .blocking import PAIR_MULTIPLIER
from .data_loader import (
    GroundTruth,
    count_data_lines,
    iter_tsv,
    prepared_path,
    raw_path,
    read_tsv,
)
from .evaluation import CandidateEvaluation, _contains_sorted, build_true_pair_codes

logger = logging.getLogger(__name__)

SUBMISSION_FILENAME = "matching_results.tsv"
S1_ID_PATTERN = re.compile(r"^S1-\d+$")
TARGET_ID_PATTERN = re.compile(r"^S[23]-\d+$")
# Strings a library writes for a missing value. None of them is a valid id, so any
# of them in the matches column means an NA leaked into the file.
NA_SPELLINGS = frozenset({"nan", "NaN", "None", "NULL", "null", "NA", "<NA>", "[]"})


class SubmissionError(ValueError):
    """The submission file violates the format rules."""


def _columns(config: Optional[dict]) -> tuple[str, str]:
    columns = (config or {}).get("columns", {}) or {}
    return (
        columns.get("gt_source1_id", "source1_entity_id"),
        columns.get("gt_matched_ids", "matched_entity_ids"),
    )


# ---------------------------------------------------------------------------
# The S1 universe
# ---------------------------------------------------------------------------
def load_s1_universe(config: dict, split: str, log: Optional[logging.Logger] = None) -> list[str]:
    """Every S1 entity id of ``split``, in file order - the rows the submission must have.

    Read from the **raw** S1 file when it is present (it is the definition of the
    entity list; a prepared table could be a smoke-test prefix), else from the
    prepared table. The id count is cross-checked against the raw file's physical
    line count, so a parse that merged or dropped a row fails here instead of
    producing a submission with missing entities.

    Raises:
        SubmissionError: on duplicate, blank or malformed ids, or a row-count mismatch.
    """
    log = log or logger
    id_column = (config.get("columns", {}) or {}).get("entity_id", "entity_id")
    source_path = raw_path(config, split, "source1")
    from_raw = source_path.is_file()
    if not from_raw:
        source_path = prepared_path(config, split, "source1")
        if not source_path.is_file():
            raise FileNotFoundError(
                f"no S1 table for split={split!r}: neither {raw_path(config, split, 'source1')} "
                f"nor {source_path} exists"
            )
        log.warning("raw S1 file missing; reading the S1 universe from %s", source_path)

    parts = [chunk[id_column].to_numpy(dtype=object) for chunk in iter_tsv(source_path, columns=[id_column])]
    ids = np.concatenate(parts).tolist() if parts else []

    problems: list[str] = []
    blank = [i for i, entity_id in enumerate(ids) if not entity_id or not entity_id.strip()]
    if blank:
        problems.append(f"{len(blank)} blank S1 ids (first at data row {blank[0] + 1})")
    malformed = [entity_id for entity_id in ids if entity_id and not S1_ID_PATTERN.match(entity_id)]
    if malformed:
        problems.append(f"{len(malformed)} malformed S1 ids, e.g. {malformed[:3]}")
    duplicated = pd.Series(ids, dtype=object)
    duplicated = duplicated[duplicated.duplicated()].unique().tolist()
    if duplicated:
        problems.append(f"{len(duplicated)} duplicated S1 ids, e.g. {duplicated[:3]}")
    if from_raw:
        lines = count_data_lines(source_path)
        if lines != len(ids):
            problems.append(
                f"{source_path.name} has {lines:,} data lines but {len(ids):,} ids were parsed"
            )
    if problems:
        raise SubmissionError(f"S1 universe from {source_path} is unusable: " + "; ".join(problems))
    log.info("S1 universe: %s entities from %s", f"{len(ids):,}", source_path)
    return ids


# ---------------------------------------------------------------------------
# Target conflicts: one S2/S3 record, several S1 claimants
# ---------------------------------------------------------------------------
def best_claim_mask(
    target_codes: np.ndarray,
    scores: np.ndarray,
    tiebreak: np.ndarray,
    return_order: bool = False,
):
    """For each target, keep the one row with the highest score (ties: lowest ``tiebreak``).

    This IS the greedy global assignment - sort every prediction by score descending
    and give each target to its first claimant, dropping later claims - because only
    the target side is capacity-limited (an S1 may keep many targets). The greedy
    walk therefore never has a reason to skip a target's best claim, and "first
    claimant in descending order" is exactly "argmax per target". Computed with one
    ``lexsort`` over integer arrays, so it scales to hundreds of millions of rows.

    Threshold order is irrelevant: a target's best claim passes a cut-off iff any of
    its claims does, so thresholding before or after this mask selects the same rows.

    Returns:
        the boolean keep mask (and, with ``return_order``, the sort order and the
        first-of-target flags in that order, for callers that need tie statistics).
    """
    target_codes = np.asarray(target_codes)
    scores = np.asarray(scores, dtype=np.float64)
    keep = np.zeros(len(target_codes), dtype=bool)
    if len(target_codes) == 0:
        return (keep, np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)) if return_order else keep
    # Primary key target, then score descending, then the tie-break ascending.
    order = np.lexsort((np.asarray(tiebreak), -scores, target_codes))
    sorted_codes = target_codes[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_codes[1:] != sorted_codes[:-1]
    keep[order[first]] = True
    return (keep, order, first) if return_order else keep


def resolve_target_conflicts(
    s1_ids: Sequence[str],
    target_ids: Sequence[str],
    scores: Sequence[float],
    s1_order: Mapping[str, int],
) -> tuple[np.ndarray, dict]:
    """Keep every target for its single highest-scoring S1; drop its other claims.

    S1 is the *deduplicated* reference list, so a noisy S2/S3 record describes at
    most one S1 business. When the thresholded predictions give the same target to
    several S1 entities, all but one of those claims is a false positive - and under
    F0.5 a false positive costs four times a false negative. The claim kept is the
    one with the highest score; an exact tie goes to the S1 that comes first in S1
    file order, so the result is deterministic.

    The one-S1-per-target premise is an assumption about the data. Check it with
    :func:`ground_truth_target_multiplicity` before relying on this, and compare the
    dry-run score with and without resolution (``predict.py --split train`` does both).

    Args:
        s1_ids, target_ids, scores: the predicted pairs (row-aligned). A duplicated
            ``(s1, target)`` pair counts once, at its highest score.
        s1_order: S1 id -> position in S1 file order (the tie-break).

    Returns:
        ``(keep, stats)``: a boolean mask over the input rows, and counts of what
        the resolution did.
    """
    s1 = np.asarray(s1_ids, dtype=object)
    targets = np.asarray(target_ids, dtype=object)
    score = np.asarray(scores, dtype=np.float64)
    n = len(targets)
    if not (len(s1) == n == len(score)):
        raise ValueError("s1_ids, target_ids and scores must be row-aligned")
    stats = {
        "pairs_in": n,
        "targets": 0,
        "targets_with_conflicts": 0,
        "pairs_dropped": 0,
        "tied_best_scores": 0,
        "s1_left_without_matches": 0,
    }
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep, stats
    if np.isnan(score).any():
        raise ValueError("scores must not be NaN")

    target_codes, target_uniques = pd.factorize(targets, sort=False)
    s1_rank = np.fromiter((s1_order[value] for value in s1), dtype=np.int64, count=n)
    keep, order, first = best_claim_mask(target_codes, score, s1_rank, return_order=True)

    # A pair duplicated in the input would count as a conflict with itself; it is
    # not one, so claimants are counted as distinct S1 ids per target.
    distinct = pd.DataFrame({"t": target_codes, "s": s1}).drop_duplicates()
    claimants = np.bincount(distinct["t"].to_numpy(), minlength=len(target_uniques))
    runner_up = np.zeros(n, dtype=bool)
    runner_up[1:] = first[:-1] & ~first[1:]  # second row of a target's block
    sorted_scores, sorted_s1 = score[order], s1[order]
    tied = runner_up.copy()
    tied[runner_up] = (sorted_scores[np.flatnonzero(runner_up) - 1] == sorted_scores[runner_up]) & (
        sorted_s1[np.flatnonzero(runner_up) - 1] != sorted_s1[runner_up]
    )
    stats.update(
        {
            "targets": int(len(target_uniques)),
            "targets_with_conflicts": int((claimants > 1).sum()),
            "pairs_dropped": int((~keep).sum()),
            "tied_best_scores": int(tied.sum()),
            "s1_left_without_matches": int(len(set(s1) - set(s1[keep]))),
        }
    )
    return keep, stats


def group_matches(s1_ids: Sequence[str], target_ids: Sequence[str]) -> dict[str, set[str]]:
    """``{s1: {targets}}`` from row-aligned predicted pairs (the writer's input)."""
    grouped: dict[str, set[str]] = {}
    for s1, target in zip(s1_ids, target_ids):
        grouped.setdefault(s1, set()).add(target)
    return grouped


def ground_truth_target_multiplicity(ground_truth: GroundTruth) -> dict:
    """How often one target id is a true match of several S1 entities.

    The premise of :func:`resolve_target_conflicts` is that this is (almost) never
    the case. ``targets_with_multiple_s1 == 0`` means resolution can only remove
    false positives; anything above 0 is the number of true pairs it may delete.
    """
    codes, counts = np.unique(ground_truth.codes, return_counts=True)
    multi = counts > 1
    return {
        "distinct_targets": int(len(codes)),
        "targets_with_multiple_s1": int(multi.sum()),
        "true_pairs_on_shared_targets": int(counts[multi].sum()),
        "max_s1_per_target": int(counts.max()) if len(counts) else 0,
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_submission(
    path: str | os.PathLike,
    s1_ids: Sequence[str],
    matches: Mapping[str, Iterable[str]],
    config: Optional[dict] = None,
) -> dict:
    """Write one line per S1 in ``s1_ids`` order; a singleton gets an exact ``""``.

    ``matches`` may omit entities (they are singletons) but must not name an S1 that
    is not in ``s1_ids`` - that would mean the predictions came from another split.
    Written to ``.partial`` and renamed, so a crash never leaves a plausible-looking
    truncated submission.

    Returns:
        ``{"rows", "singletons", "matched_entities", "pairs"}``.
    """
    path = Path(path)
    id_column, match_column = _columns(config)
    universe = set(s1_ids)
    unknown = [s1 for s1 in matches if s1 not in universe]
    if unknown:
        raise SubmissionError(
            f"{len(unknown)} predicted S1 ids are not in the S1 universe, e.g. {unknown[:3]} - "
            "were the features built for another split?"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    singletons = pairs = 0
    with open(partial, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{id_column}\t{match_column}\n")
        for s1 in s1_ids:
            targets = sorted(set(matches.get(s1, ())))
            bad = [t for t in targets if not TARGET_ID_PATTERN.match(t)]
            if bad:
                raise SubmissionError(f"S1 {s1}: invalid matched ids {bad[:3]}")
            if targets:
                pairs += len(targets)
            else:
                singletons += 1
            handle.write(f"{s1}\t{','.join(targets)}\n")
    partial.replace(path)
    return {
        "rows": len(s1_ids),
        "singletons": singletons,
        "matched_entities": len(s1_ids) - singletons,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_submission(
    path: str | os.PathLike,
    expected_s1_ids: Optional[Sequence[str]] = None,
    config: Optional[dict] = None,
) -> dict:
    """Check a submission file byte by byte, then check it reads back the same way.

    Two independent readings: the raw lines (what the grader's bytes are), and a
    strict pandas read (``keep_default_na=False``), which must yield ``""`` - never
    NaN - for every singleton.

    Raises:
        SubmissionError: listing every violated rule (with examples).
    """
    path = Path(path)
    id_column, match_column = _columns(config)
    problems: list[str] = []
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("file starts with a UTF-8 BOM")
    if b"\r" in raw:
        problems.append("file contains CR characters (Windows line endings)")
    text = raw.decode("utf-8")
    if not text.endswith("\n"):
        problems.append("file does not end with a newline")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[0] != f"{id_column}\t{match_column}":
        problems.append(f"header must be exactly {id_column!r}<TAB>{match_column!r}, got {lines[:1]!r}")
    body = lines[1:]

    seen: list[str] = []
    singletons = pairs = 0
    examples: dict[str, list] = {}

    def note(kind: str, value) -> None:
        examples.setdefault(kind, [])
        if len(examples[kind]) < 3:
            examples[kind].append(value)

    for number, line in enumerate(body, start=2):
        fields = line.split("\t")
        if len(fields) != 2:
            note("lines without exactly one tab", number)
            continue
        s1, joined = fields
        seen.append(s1)
        if not S1_ID_PATTERN.match(s1):
            note("malformed S1 ids", s1)
        if joined == "":
            singletons += 1
            continue
        if joined.strip() in NA_SPELLINGS:
            note("NA spelled as text instead of an empty string", f"line {number}: {joined!r}")
            continue
        targets = joined.split(",")
        if len(set(targets)) != len(targets):
            note("duplicated matched ids within a row", s1)
        for target in targets:
            if not TARGET_ID_PATTERN.match(target):
                note("invalid matched ids", f"line {number}: {target!r}")
        pairs += len(targets)
    for kind, values in examples.items():
        problems.append(f"{kind}: {values}")

    duplicated = pd.Series(seen, dtype=object)
    duplicated = duplicated[duplicated.duplicated()].unique().tolist()
    if duplicated:
        problems.append(f"{len(duplicated)} S1 ids appear more than once, e.g. {duplicated[:3]}")
    if expected_s1_ids is not None:
        expected = list(expected_s1_ids)
        missing = set(expected) - set(seen)
        extra = set(seen) - set(expected)
        if missing:
            problems.append(f"{len(missing)} S1 entities missing, e.g. {sorted(missing)[:3]}")
        if extra:
            problems.append(f"{len(extra)} unexpected S1 ids, e.g. {sorted(extra)[:3]}")
        if not missing and not extra and not duplicated and seen != expected:
            problems.append("rows are not in S1 file order")

    # The second, library-level reading.
    frame = read_tsv(path)
    if list(frame.columns) != [id_column, match_column]:
        problems.append(f"pandas reads columns {list(frame.columns)}")
    elif frame[match_column].isna().any():
        problems.append("pandas reads NaN in the matches column")
    elif int((frame[match_column] == "").sum()) != singletons:
        problems.append("pandas and the raw lines disagree on the singleton count")

    if problems:
        raise SubmissionError(f"{path} is not a valid submission:\n  - " + "\n  - ".join(problems))
    return {"rows": len(body), "singletons": singletons, "pairs": pairs}


# ---------------------------------------------------------------------------
# Scoring (train split only - the test ground truth is the grader's)
# ---------------------------------------------------------------------------
def score_submission(
    path: str | os.PathLike,
    ground_truth: GroundTruth,
    s1_mask: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
) -> dict:
    """Macro F0.5 of a submission file against the ground truth.

    Uses :meth:`CandidateEvaluation._macro_f05` - the one implementation of the
    metric in this repository - under both zero-match policies. ``score_zero`` is the
    challenge's rule (an empty-GT entity scores 1 for ``""`` and 0 for anything else).
    An entity missing from the file is scored as predicting nothing.
    """
    id_column, match_column = _columns(config)
    predicted = GroundTruth.from_tsv(path, id_column=id_column, match_column=match_column)
    owners_of_rows = ground_truth.positions_of(pd.Series(predicted.entity_ids, dtype=object))
    if (owners_of_rows < 0).any():
        unknown = [e for e, o in zip(predicted.entity_ids, owners_of_rows) if o < 0][:3]
        raise SubmissionError(f"submission names S1 ids absent from the ground truth, e.g. {unknown}")

    owners = np.repeat(owners_of_rows, predicted.lengths())
    packed = np.unique(owners * PAIR_MULTIPLIER + predicted.codes)
    pair_owner = packed // PAIR_MULTIPLIER
    is_true = _contains_sorted(build_true_pair_codes(ground_truth), packed)

    n = ground_truth.n_entities
    counts = np.bincount(pair_owner, minlength=n)
    hits = np.bincount(pair_owner[is_true], minlength=n)
    lengths = ground_truth.lengths()
    mask = np.ones(n, dtype=bool) if s1_mask is None else np.asarray(s1_mask, dtype=bool)
    return {
        "n_s1_entities": int(mask.sum()),
        "predicted_pairs": int(counts[mask].sum()),
        "true_positives": int(hits[mask].sum()),
        "macro_f05_score_zero": CandidateEvaluation._macro_f05(
            lengths[mask], counts[mask], hits[mask], policy="score_zero"
        ),
        "macro_f05_exclude": CandidateEvaluation._macro_f05(
            lengths[mask], counts[mask], hits[mask], policy="exclude"
        ),
    }
