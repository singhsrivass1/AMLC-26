#!/usr/bin/env python
"""Offline blocking error analysis: which true pairs blocking loses, and what recovering them costs.

This is a **measurement** stage. It reads the production candidate file and the ground
truth; it writes nothing but its own reports under ``--output-dir``. No blocker, no
candidate file, no feature definition and no config value is touched.

What it answers
---------------
1. **Blocker contribution.** For every ground-truth pair, which blocker retrieved it -
   exact name, token, char n-gram, or a combination? Reported per blocker and over all
   seven non-empty combinations, from the ``blockers`` provenance column the union
   already writes. Per-blocker candidate volume is reported alongside, because a recall
   figure without the candidate rows it cost is not a decision.
2. **Missed pairs.** The ground-truth pairs no blocker retrieved, characterised by the
   same pair features the matcher sees - recomputed here from the prepared tables,
   because by definition these pairs are absent from the feature file.
3. **Per source.** S2 and S3 separately, since the failure modes may differ.
4. **Multilingual / script / transliteration.** Measured, not assumed: Unicode script of
   each side, cross-script pairs, reordering, insertion/deletion, abbreviation, legal
   suffix variation, address formatting and missingness. The repository contains no
   transliteration step and NFKC cannot make two scripts equivalent, so a cross-script
   pair is unreachable by *any* lexical blocker - this pass measures how big that floor
   is rather than guessing.
5. **Entity impact.** Macro F0.5 is averaged per S1 entity, so pair recall alone is not
   the currency. Entities fully / partially / zero covered, zero-candidate entities, the
   missed-pair distribution, and how concentrated the loss is (Lorenz / top-decile share).
6. **Proposed blockers.** Each proposal is a key function; its recall on the ground truth
   and the candidate rows it would create are measured, and the downstream effect is
   reported through the chain in :func:`metric_chain` - never as "recall up by X,
   therefore F0.5 up by X".

Why two passes over the candidate file
--------------------------------------
Pass 1 is the repository's own :class:`~src.evaluation.CandidateEvaluation`, unmodified,
so the headline recall / precision / accept-all macro F0.5 cannot drift from the graded
metric. Pass 2 is analysis-specific (provenance, per-blocker recovery) and reuses that
class's helpers, so both passes define a true pair the same way. Two passes cost roughly
twice the read; one pass would mean reimplementing the evaluator's accumulation and
risking exactly the divergence the first pass exists to prevent. ``--skip-official``
drops pass 1 for a faster, less authoritative run.

Usage
-----
    python scripts/analyze_blocking_errors.py --config configs/config.yaml --split val

The candidate file defaults to ``<candidates_dir>/candidate_pairs.tsv`` - the full
production set. ``--split val`` restricts scoring to the 20% of S1 entities the V1
matcher was evaluated on; ``--split all`` scores every entity.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (  # noqa: E402
    BLOCKER_CHAR_NGRAM,
    BLOCKER_EXACT_NAME,
    BLOCKER_TOKEN,
    LEXICAL_BLOCKERS,
    PAIR_MULTIPLIER,
    _trigram_jaccard,
    resolve_blocker_settings,
)

# This analysis is defined over the three lexical generators (its combination
# bitmask and the "all_three" label are). The dense blocker is not part of it.
UNION_BLOCKERS = LEXICAL_BLOCKERS
from src.data_loader import (  # noqa: E402
    GroundTruth,
    candidates_path,
    describe_environment,
    iter_prepared,
    iter_tsv,
    load_config,
    load_ground_truth,
    prepared_path,
)  # noqa: E402
from src.evaluation import (  # noqa: E402
    CANDIDATE_S1_COLUMN,
    CANDIDATE_TARGET_COLUMN,
    CandidateEvaluation,
    _contains_sorted,
    _encode_target_codes,
    _f05_ceiling,
    _macro_entity_recall,
    _peek_columns,
    build_true_pair_codes,
    format_report,
    save_metrics,
    split_mask_for,
)  # noqa: E402
from src.normalization import (  # noqa: E402
    ADDRESS_NORM,
    COUNTRY_NORM,
    NAME_KEY,
    NAME_NORM,
)
from src.utils import (  # noqa: E402
    decode_entity_id,
    decode_entity_ids,
    encode_entity_ids,
    fmt_int,
    log_memory,
    read_json,
    set_seed,
    setup_logging,
    write_json,
)

LOG_NAME = "analyze_blocking_errors"

# The union writes provenance as ``<source>:<blocker>`` (``source2:exact_name``); the
# colon anchor is what keeps ``token`` from one day matching inside a longer label.
PROVENANCE_COLUMN = "blockers"
EVIDENCE_COLUMN = "char_jaccard"

MISSED_COLUMNS = (
    NAME_NORM,
    NAME_KEY,
    ADDRESS_NORM,
    COUNTRY_NORM,
)

# The seven non-empty subsets of the three production blockers, as a bitmask so the
# combination histogram is one ``bincount``. Order matches UNION_BLOCKERS.
_BLOCKER_BITS = {
    BLOCKER_EXACT_NAME: 1,
    BLOCKER_TOKEN: 2,
    BLOCKER_CHAR_NGRAM: 4,
}


def combination_label(mask: int) -> str:
    """Render a provenance bitmask as ``exact+token`` / ``char_ngram`` / ``union``."""
    names = [b for b, bit in _BLOCKER_BITS.items() if mask & bit]
    if not names:
        return "none"
    if len(names) == len(UNION_BLOCKERS):
        return "all_three"
    return "+".join(names)


COMBINATION_ORDER = tuple(combination_label(mask) for mask in range(1, 8))


# ---------------------------------------------------------------------------
# Pass 2 - provenance and per-blocker recovery
# ---------------------------------------------------------------------------
def provenance_flags(provenance: pd.Series) -> dict[str, np.ndarray]:
    """Per-blocker boolean flags parsed from the ``blockers`` provenance column.

    Vectorized on purpose: the column is 336M strings in the full run, so a python
    ``split(",")`` per row is not affordable. The labels are a closed set written by
    :func:`src.blocking.union_blockers`, always with the source prefix, so an anchored
    substring test is exact rather than approximate.
    """
    text = provenance.fillna("").astype(str)
    return {label: text.str.contains(f":{label}", regex=False).to_numpy() for label in UNION_BLOCKERS}


def scan_candidates(
    path: Path,
    ground_truth: GroundTruth,
    true_pair_codes: np.ndarray,
    chunksize: int,
    max_rows: Optional[int],
    log: logging.Logger,
) -> dict[str, Any]:
    """Stream the candidate file and record what each blocker retrieved.

    Holds O(n_entities + n_true_pairs) state, never the candidate table: 336M rows
    stream through in ``chunksize`` blocks. Returns per-blocker candidate volumes,
    per-blocker recovered ground-truth pairs (as boolean arrays over the sorted
    true-pair array), the provenance combination histogram, and per-entity candidate
    counts including the ones spent on entities with no true match at all.
    """
    n_entities = ground_truth.n_entities
    lengths = ground_truth.lengths()

    flags_total = {label: 0 for label in UNION_BLOCKERS}
    flags_by_source: dict[str, dict[int, int]] = {label: {2: 0, 3: 0} for label in UNION_BLOCKERS}
    rows_by_entity = np.zeros(n_entities, dtype=np.int64)
    rows_by_entity_blocker = {label: np.zeros(n_entities, dtype=np.int64) for label in UNION_BLOCKERS}
    recovered = np.zeros(len(true_pair_codes), dtype=bool)
    recovered_by_blocker = {label: np.zeros(len(true_pair_codes), dtype=bool) for label in UNION_BLOCKERS}
    combination_counts = np.zeros(8, dtype=np.int64)
    combination_counts_by_source = {2: np.zeros(8, dtype=np.int64), 3: np.zeros(8, dtype=np.int64)}
    row_combination_counts = np.zeros(8, dtype=np.int64)
    # The two histograms above are global. These two are the same facts kept per key, so
    # a table restricted to a split can mask its numerator *and* its denominator. Without
    # them a `--split val` table would divide val recall by all-split volume, which reads
    # as a recall above 1.0 - the kind of mixed-scope number that misdirects a decision.
    pair_combination = np.zeros(len(true_pair_codes), dtype=np.uint8)
    row_combination_by_entity = np.zeros((8, n_entities), dtype=np.int64)

    rows_seen = 0
    unknown_s1 = 0
    unlabelled_rows = 0
    evidence_present = 0
    started = time.time()

    columns = [CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN, PROVENANCE_COLUMN]
    available = set(_peek_columns(path))
    missing = [name for name in columns if name not in available]
    if missing:
        raise ValueError(f"{path} is missing required column(s): " + ", ".join(missing))
    # char_jaccard is optional: it is the only evidence column, and its emptiness is
    # itself a fact worth reporting (a union row that only exact_name produced has none).
    has_evidence = EVIDENCE_COLUMN in available
    if has_evidence:
        columns.append(EVIDENCE_COLUMN)

    for chunk in iter_tsv(path, columns=columns, chunksize=chunksize):
        if max_rows is not None and rows_seen >= max_rows:
            break
        if max_rows is not None and rows_seen + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - rows_seen]
        rows_seen += len(chunk)

        owners = ground_truth.positions_of(chunk[CANDIDATE_S1_COLUMN])
        target_codes = _encode_target_codes(chunk[CANDIDATE_TARGET_COLUMN])
        flags = provenance_flags(chunk[PROVENANCE_COLUMN])

        keep = owners >= 0
        n_unknown = int((~keep).sum())
        if n_unknown:
            unknown_s1 += n_unknown
            owners = owners[keep]
            target_codes = target_codes[keep]
            flags = {label: values[keep] for label, values in flags.items()}
        if not len(owners):
            continue

        packed = owners * PAIR_MULTIPLIER + target_codes
        is_true = _contains_sorted(true_pair_codes, packed)
        position = np.searchsorted(true_pair_codes, packed)
        np.clip(position, 0, max(len(true_pair_codes) - 1, 0), out=position)

        source_codes = (target_codes // 10**10).astype(np.int64)
        bitmask = np.zeros(len(owners), dtype=np.int64)
        any_flag = np.zeros(len(owners), dtype=bool)
        for label, bit in _BLOCKER_BITS.items():
            values = flags[label]
            bitmask |= np.where(values, bit, 0)
            any_flag |= values
            if values.any():
                flags_total[label] += int(values.sum())
                np.add.at(rows_by_entity_blocker[label], owners[values], 1)
                for source_code in (2, 3):
                    flags_by_source[label][source_code] += int((values & (source_codes == source_code)).sum())
                recovered_by_blocker[label][position[is_true & values]] = True

        unlabelled_rows += int((~any_flag).sum())
        np.add.at(rows_by_entity, owners, 1)
        # Row-level combinations answer "how much volume does each blocker propose
        # alone" - the question a removal or a retune actually turns on. Pair-level
        # combinations (below) answer "what did each blocker contribute to recall".
        row_combination_counts += np.bincount(bitmask, minlength=8)
        np.add.at(row_combination_by_entity, (bitmask, owners), 1)

        if is_true.any():
            recovered[position[is_true]] = True
            combination_counts += np.bincount(bitmask[is_true], minlength=8)
            pair_combination[position[is_true]] = bitmask[is_true].astype(np.uint8)
            for source_code in (2, 3):
                in_source = is_true & (source_codes == source_code)
                if in_source.any():
                    combination_counts_by_source[source_code] += np.bincount(bitmask[in_source], minlength=8)

        if has_evidence:
            # na_filter=False throughout iter_tsv, so absence is the empty string, not NaN.
            evidence_present += int((chunk[EVIDENCE_COLUMN] != "").sum())

        if rows_seen and rows_seen % (chunksize * 20) < chunksize:
            log.info(
                "  scanned %s candidate rows (%.1f min, %s recovered true pairs)",
                fmt_int(rows_seen),
                (time.time() - started) / 60.0,
                fmt_int(int(recovered.sum())),
            )

    log.info("candidate scan: %s rows in %.1f min", fmt_int(rows_seen), (time.time() - started) / 60.0)
    if unknown_s1:
        log.warning("%s candidate rows name an S1 entity absent from the ground truth", fmt_int(unknown_s1))
    if unlabelled_rows:
        log.warning(
            "%s candidate rows carry no recognised blocker label; the per-blocker table under-counts by that much",
            fmt_int(unlabelled_rows),
        )

    return {
        "rows_seen": rows_seen,
        "unknown_s1_rows": unknown_s1,
        "unlabelled_rows": unlabelled_rows,
        "evidence_rows": evidence_present,
        "flags_total": flags_total,
        "flags_by_source": flags_by_source,
        "rows_by_entity": rows_by_entity,
        "rows_by_entity_blocker": rows_by_entity_blocker,
        "recovered": recovered,
        "recovered_by_blocker": recovered_by_blocker,
        "combination_counts": combination_counts,
        "combination_counts_by_source": combination_counts_by_source,
        "row_combination_counts": row_combination_counts,
        "pair_combination": pair_combination,
        "row_combination_by_entity": row_combination_by_entity,
        "lengths": lengths,
    }


def _pair_mask(scan: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    """Expand a per-entity mask to the per-pair arrays.

    The scan's recovery arrays are one entry per ground-truth pair while the mask is one
    entry per S1 entity, and the two counts differ (7.6M pairs over 2.2M entities), so
    indexing one with the other is an IndexError rather than a silent wrong answer. The
    expansion is the same ``repeat`` ``build_true_pair_codes`` packs with, which is why
    pair ``i`` belongs to ``owner[i]``.
    """
    owner = np.repeat(np.arange(len(scan["lengths"]), dtype=np.int64), scan["lengths"])
    return mask[owner]


def blocker_table(scan: dict[str, Any], mask: np.ndarray, n_rows_total: int) -> list[dict[str, Any]]:
    """Per-blocker contribution, with the incremental view over the union.

    Two numbers matter and they are not the same. ``true_pairs_recovered`` is what the
    blocker finds on its own; ``pairs_union_would_lose_without_it`` is what the *union*
    would lose if it were removed - the only quantity that changes a decision, because a
    blocker can carry a large share of recall while adding nothing its peers would not.
    Per-blocker candidate rows sit next to both, since recall bought with volume is not
    free at 1.3% candidate precision.

    Every column is restricted to the masked entities, numerator and denominator alike:
    ``candidate_rows`` counts rows whose S1 is in scope, and shares are taken against the
    in-scope row total. The unrestricted totals are kept as ``*_all`` so a split run can
    still be read against the whole file.
    """
    pair_mask = _pair_mask(scan, mask)
    recovered = scan["recovered"][pair_mask]
    n_true = int(recovered.size)
    if n_true == 0:
        return []

    singles = {label: scan["recovered_by_blocker"][label][pair_mask] for label in UNION_BLOCKERS}
    in_scope_rows = int(scan["rows_by_entity"][mask].sum())
    rows = []
    for label in UNION_BLOCKERS:
        alone = singles[label]
        others = np.zeros_like(alone)
        for other in UNION_BLOCKERS:
            if other != label:
                others |= singles[other]
        bit = _BLOCKER_BITS[label]
        scoped_rows = int(scan["rows_by_entity_blocker"][label][mask].sum())
        rows.append(
            {
                "blocker": label,
                "true_pairs_recovered": int(alone.sum()),
                "true_pair_recall": _ratio(int(alone.sum()), n_true),
                "candidate_rows": scoped_rows,
                "candidate_rows_all": int(scan["flags_total"][label]),
                "share_of_candidate_rows": _ratio(scoped_rows, in_scope_rows),
                "share_of_all_candidate_rows": _ratio(scan["flags_total"][label], n_rows_total),
                "candidate_precision": _ratio(int(alone.sum()), scoped_rows),
                "pairs_only_this_blocker_finds": int((alone & ~others).sum()),
                "pairs_union_would_lose_without_it": int((recovered & ~others).sum()),
                "rows_where_only_this_blocker_fires": int(scan["row_combination_by_entity"][bit][mask].sum()),
            }
        )
    return rows


def combination_table(scan: dict[str, Any], mask: np.ndarray) -> list[dict[str, Any]]:
    """Recall and volume for each of the seven non-empty provenance combinations.

    Reads as: ``char_ngram`` alone carries most of the recall, ``exact_name`` is nearly
    subsumed by it, and ``all_three`` is the redundant middle. ``rows`` here is the
    candidate-row count, so the last column is the precision that combination buys.
    Masked on both sides, like :func:`blocker_table`.
    """
    pair_mask = _pair_mask(scan, mask)
    n_true = int(pair_mask.sum())
    pairs_in_scope = scan["pair_combination"][pair_mask]
    row_counts = scan["row_combination_by_entity"][:, mask].sum(axis=1)
    rows = []
    for mask_value in range(1, 8):
        n_pairs = int((pairs_in_scope == mask_value).sum())
        n_rows = int(row_counts[mask_value])
        rows.append(
            {
                "combination": combination_label(mask_value),
                "true_pairs_recovered": n_pairs,
                "true_pair_recall": _ratio(n_pairs, n_true),
                "candidate_rows": n_rows,
                "candidate_precision": _ratio(n_pairs, n_rows),
            }
        )
    return rows


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


# ---------------------------------------------------------------------------
# Names for the pairs blocking lost
# ---------------------------------------------------------------------------
def load_prepared_names(
    config: dict,
    prepared_split: str,
    wanted_by_source: dict[str, np.ndarray],
    columns: Sequence[str],
    log: logging.Logger,
) -> dict[int, tuple[str, ...]]:
    """Fetch normalized fields for the entities a caller actually needs.

    The missed pairs, not the corpus, decide how much is loaded: prepared tables hold
    12.6M rows and there is no reason to hold them all to describe ~1M pairs. Each
    source streams once, encoding ids vectorized and keeping only wanted rows, so the
    resident cost is O(needed), not O(corpus).

    Args:
        wanted_by_source: ``{"source1": codes, "source2": codes, ...}`` - entity codes
            (from :func:`src.utils.encode_entity_ids`) to keep. Empty array = skip.

    Returns:
        ``{entity_code: (column values in ``columns`` order)}``.
    """
    lookup: dict[int, tuple[str, ...]] = {}
    for source, wanted in wanted_by_source.items():
        if not len(wanted):
            continue
        wanted_sorted = np.unique(wanted)
        keep_total = 0
        for chunk in iter_prepared(config, prepared_split, source, columns=["entity_id", *columns]):
            codes = encode_entity_ids(chunk["entity_id"])
            position = np.searchsorted(wanted_sorted, codes)
            np.clip(position, 0, len(wanted_sorted) - 1, out=position)
            keep = wanted_sorted[position] == codes
            if not keep.any():
                continue
            keep_total += int(keep.sum())
            for code, values in zip(codes[keep], chunk.loc[keep, list(columns)].itertuples(index=False, name=None)):
                lookup[int(code)] = tuple(str(value) for value in values)
        log.info("loaded %s prepared rows (%s wanted) for %s", fmt_int(keep_total), fmt_int(len(wanted_sorted)), source)
    return lookup


# ---------------------------------------------------------------------------
# Pair diagnostics - what is different about the pairs blocking missed
# ---------------------------------------------------------------------------
# Unicode script detection by name, cached per character: the cache is what makes this
# affordable over ~1M pairs, since a few hundred characters cover most of the corpus.
_SCRIPTS = (
    "LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "DEVANAGARI", "BENGALI",
    "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM",
    "SINHALA", "THAI", "LAO", "TIBETAN", "MYANMAR", "GEORGIAN", "ARMENIAN",
    "HANGUL", "HIRAGANA", "KATAKANA", "CJK", "THAANA", "ETHIOPIC", "CHEROKEE",
    "MONGOLIAN", "KHMER", "BOPOMOFO", "YI ",
)
_SCRIPT_CACHE: dict[str, str] = {}


def _char_script(char: str) -> str:
    cached = _SCRIPT_CACHE.get(char)
    if cached is not None:
        return cached
    name = unicodedata.name(char, "") if char.isalpha() else ""
    found = ""
    for script in _SCRIPTS:
        if script in name:
            found = script.strip()
            break
    _SCRIPT_CACHE[char] = found
    return found


def scripts_of(text: str) -> frozenset[str]:
    """Scripts present in ``text``; empty for digits/punctuation only."""
    return frozenset(filter(None, (_char_script(char) for char in text)))


# Legal-form and generic tokens that carry no discriminating signal. This list is a
# HYPOTHESIS under test by ``suffix_stripped`` in the proposal table - it is not a
# production rule and nothing in the pipeline reads it.
LEGAL_SUFFIXES = frozenset(
    {
        "ltd", "limited", "pvt", "private", "inc", "incorporated", "llp", "llc", "plc",
        "co", "company", "corp", "corporation", "enterprises", "enterprise", "industries",
        "industry", "services", "service", "solutions", "solution", "technologies",
        "technology", "tech", "systems", "system", "group", "holdings", "international",
        "intl", "and", "the", "of", "india", "traders", "trading", "agency", "agencies",
        "stores", "store", "shop", "shoppe", "centre", "center", "works", "products",
        "exports", "imports", "associates", "consultants", "consultancy", "ventures",
        "labs", "laboratories", "pharma", "pharmaceuticals", "motors", "auto", "automobiles",
    }
)


def strip_legal_suffixes(tokens: Sequence[str]) -> tuple[str, ...]:
    """Drop legal-form tokens, but never all of them.

    ``"Pvt Ltd"`` would otherwise reduce to nothing and collide with every other
    stripped-to-empty name, which is worse than not stripping it.
    """
    stripped = tuple(token for token in tokens if token not in LEGAL_SUFFIXES)
    return stripped or tuple(tokens)


def _token_set(text: str) -> frozenset[str]:
    return frozenset(text.split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def diagnose_pair(
    name_a: str,
    name_b: str,
    key_a: str,
    key_b: str,
    address_a: str,
    address_b: str,
    country_a: str,
    country_b: str,
) -> dict[str, Any]:
    """The diagnostic vector for one pair, plus a single primary-cause label.

    ``primary_cause`` is a decision tree over the same facts, ordered so that a
    *defect* (identical normalized names that exact-name blocking still missed) is
    reported as such rather than absorbed into a benign bucket like "shared token".
    """
    tokens_a = tuple(name_a.split())
    tokens_b = tuple(name_b.split())
    set_a, set_b = frozenset(tokens_a), frozenset(tokens_b)
    scripts_a = scripts_of(name_a)
    scripts_b = scripts_of(name_b)
    shared = set_a & set_b

    addr_a, addr_b = _token_set(address_a), _token_set(address_b)
    trigram = _trigram_jaccard(key_a, key_b)
    sorted_equal = tuple(sorted(tokens_a)) == tuple(sorted(tokens_b)) and bool(tokens_a)
    subset = bool(set_a) and bool(set_b) and (set_a <= set_b or set_b <= set_a)
    suffix_a = strip_legal_suffixes(tokens_a)
    suffix_b = strip_legal_suffixes(tokens_b)
    initials_a = "".join(token[0] for token in tokens_a if token)
    initials_b = "".join(token[0] for token in tokens_b if token)
    shorter, longer = (key_a, key_b) if len(key_a) <= len(key_b) else (key_b, key_a)

    if not name_a or not name_b:
        cause = "empty_normalized_name"
    elif name_a == name_b:
        cause = "DEFECT_identical_name_not_retrieved"
    elif scripts_a and scripts_b and not (scripts_a & scripts_b):
        cause = "cross_script_no_transliteration"
    elif sorted_equal:
        cause = "token_reorder_only"
    elif subset:
        cause = "token_insertion_or_deletion"
    elif suffix_a == suffix_b:
        cause = "legal_suffix_variation"
    elif initials_a and (initials_a == name_b.replace(" ", "") or initials_b == name_a.replace(" ", "")):
        cause = "abbreviation_initials"
    elif shorter and longer.startswith(shorter):
        cause = "prefix_truncation"
    elif shared:
        cause = "shares_token_but_not_retrieved"
    elif trigram >= 0.3:
        cause = "char_overlap_above_threshold_but_not_retrieved"
    elif trigram > 0.0:
        cause = "char_overlap_below_threshold"
    else:
        cause = "no_lexical_overlap"

    return {
        "name_a_empty": not name_a,
        "name_b_empty": not name_b,
        "name_equal": name_a == name_b,
        "script_a": ",".join(sorted(scripts_a)) or "NONE",
        "script_b": ",".join(sorted(scripts_b)) or "NONE",
        "cross_script": bool(scripts_a and scripts_b and not (scripts_a & scripts_b)),
        "non_latin_a": bool(scripts_a and "LATIN" not in scripts_a),
        "non_latin_b": bool(scripts_b and "LATIN" not in scripts_b),
        "token_count_a": len(tokens_a),
        "token_count_b": len(tokens_b),
        "shared_tokens": len(shared),
        "token_jaccard": _jaccard(set_a, set_b),
        "first_token_equal": bool(tokens_a and tokens_b and tokens_a[0] == tokens_b[0]),
        "sorted_tokens_equal": sorted_equal,
        "token_subset": subset,
        "initials_match": bool(
            initials_a and (initials_a == "".join(tokens_b) or initials_b == "".join(tokens_a))
        ),
        "suffix_stripped_equal": suffix_a == suffix_b,
        "prefix_relation": bool(shorter) and longer.startswith(shorter),
        "char_len_ratio": _ratio(min(len(key_a), len(key_b)), max(len(key_a), len(key_b))),
        "trigram_jaccard": trigram,
        "digits_only_diff": bool(name_a) and _strip_digits(name_a) == _strip_digits(name_b) and name_a != name_b,
        "address_jaccard": _jaccard(addr_a, addr_b),
        "address_a_blank": not address_a,
        "address_b_blank": not address_b,
        "address_both_blank": not address_a and not address_b,
        "country_equal": bool(country_a) and country_a == country_b,
        "country_a_blank": not country_a,
        "country_b_blank": not country_b,
        "country_both_blank": not country_a and not country_b,
        "primary_cause": cause,
    }


def _strip_digits(text: str) -> str:
    return "".join(char for char in text if not char.isdigit())


DIAGNOSTIC_NUMERIC = (
    "token_count_a", "token_count_b", "shared_tokens", "token_jaccard", "char_len_ratio",
    "trigram_jaccard", "address_jaccard",
)
DIAGNOSTIC_BOOLEAN = (
    "name_a_empty", "name_b_empty", "name_equal", "cross_script", "non_latin_a", "non_latin_b",
    "first_token_equal", "sorted_tokens_equal", "token_subset", "initials_match",
    "suffix_stripped_equal", "prefix_relation", "digits_only_diff", "address_a_blank",
    "address_b_blank", "address_both_blank", "country_equal", "country_a_blank",
    "country_b_blank", "country_both_blank",
)


def _iter_pair_diagnostics(
    pair_positions: np.ndarray,
    names: dict[int, tuple[str, ...]],
    true_pair_codes: np.ndarray,
    s1_codes: np.ndarray,
    chunk_reporter: Optional[Callable[[int, int], None]] = None,
) -> Iterable[dict[str, Any]]:
    """Diagnose pairs one at a time; the caller aggregates so nothing is held in full.

    ``true_pair_codes`` packs a ground-truth *row position* on the S1 side and a real
    entity code on the target side, so both sides are recovered from the same packed
    value - ``s1_codes[packed // PAIR_MULTIPLIER]`` and ``packed % PAIR_MULTIPLIER``.
    Indexing either entity array with the pair position itself is wrong as soon as pairs
    outnumber entities, and it fails silently for a dict lookup: an out-of-range code
    yields the empty tuple and the pair is reported as ``empty_normalized_name``.
    """
    empty = ("", "", "", "")
    for index in range(len(pair_positions)):
        packed = int(true_pair_codes[pair_positions[index]])
        values_a = names.get(int(s1_codes[packed // PAIR_MULTIPLIER]), empty)
        values_b = names.get(packed % PAIR_MULTIPLIER, empty)
        yield diagnose_pair(
            values_a[0], values_b[0], values_a[1], values_b[1],
            values_a[2], values_b[2], values_a[3], values_b[3],
        )
        if chunk_reporter is not None and index and index % 100_000 == 0:
            chunk_reporter(index, len(pair_positions))


def summarise_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Distributional summary of a diagnostic set: means, booleans, and cause shares."""
    n = len(rows)
    if not n:
        return {"n_pairs": 0}
    summary: dict[str, Any] = {"n_pairs": n}
    for name in DIAGNOSTIC_NUMERIC:
        values = np.array([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "share_nonzero": float((values > 0).mean()),
        }
    for name in DIAGNOSTIC_BOOLEAN:
        summary[name] = float(np.mean([bool(row[name]) for row in rows]))
    causes: dict[str, int] = {}
    for row in rows:
        causes[row["primary_cause"]] = causes.get(row["primary_cause"], 0) + 1
    summary["primary_cause"] = dict(sorted(causes.items(), key=lambda item: -item[1]))
    summary["primary_cause_share"] = {key: value / n for key, value in summary["primary_cause"].items()}
    return summary


# ---------------------------------------------------------------------------
# Entity accounting - per-entity, not per-pair, because that is what is graded
# ---------------------------------------------------------------------------
def entity_accounting(scan: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    """Per-entity coverage: what fraction of entities a blocker reaches at all.

    The graded score averages per S1 entity, so an entity whose 8 true matches lose
    1 of them costs nearly as much as one that loses all 8. Pair recall cannot show
    that; these buckets can. ``candidate_rows_on_no_match_entities`` is the other half
    of the story - volume spent on entities that have no true match to find, which is
    a pure precision cost at 1.3% candidate precision.
    """
    lengths = scan["lengths"]
    n_entities = len(lengths)
    owner = np.repeat(np.arange(n_entities, dtype=np.int64), lengths)
    per_entity_recovered = np.bincount(
        owner, weights=scan["recovered"].astype(np.float64), minlength=n_entities
    ).astype(np.int64)
    per_entity_missed = lengths - per_entity_recovered
    rows_by_entity = scan["rows_by_entity"]

    masked_lengths = lengths[mask]
    masked_recovered = per_entity_recovered[mask]
    masked_rows = rows_by_entity[mask]

    has_true = masked_lengths > 0
    full = has_true & (masked_recovered == masked_lengths)
    none = has_true & (masked_recovered == 0)
    partial = has_true & (masked_recovered > 0) & (masked_recovered < masked_lengths)

    macro_recall_candidates = _macro_entity_recall(masked_lengths, masked_recovered)
    missed = per_entity_missed[mask]
    positive = missed[missed > 0]
    top_decile_share = 0.0
    if len(positive):
        ordered = np.sort(positive)[::-1]
        cut = max(1, len(ordered) // 10)
        top_decile_share = float(ordered[:cut].sum()) / float(ordered.sum())

    return {
        "entities_scored": int(mask.sum()),
        "entities_with_true_matches": int(has_true.sum()),
        "entities_fully_covered": int(full.sum()),
        "entities_partially_covered": int(partial.sum()),
        "entities_with_zero_recovered": int(none.sum()),
        "entities_with_no_true_match": int((~has_true).sum()),
        "entities_with_zero_candidates": int((masked_rows == 0).sum()),
        "entities_with_zero_candidates_and_true_matches": int(((masked_rows == 0) & has_true).sum()),
        "candidate_rows_on_no_match_entities": int(masked_rows[~has_true].sum()),
        "candidate_rows_per_entity_p50": float(np.percentile(masked_rows, 50)) if len(masked_rows) else 0.0,
        "candidate_rows_per_entity_p99": float(np.percentile(masked_rows, 99)) if len(masked_rows) else 0.0,
        "candidate_macro_entity_recall": macro_recall_candidates,
        "candidate_macro_f05_ceiling": _f05_ceiling(macro_recall_candidates),
        "missed_pairs_per_entity_p50": float(np.percentile(missed, 50)) if len(missed) else 0.0,
        "missed_pairs_per_entity_p99": float(np.percentile(missed, 99)) if len(missed) else 0.0,
        "missed_pairs_per_entity_max": int(missed.max()) if len(missed) else 0,
        "missed_pairs_worst_decile_share": top_decile_share,
        "per_entity_recovered": per_entity_recovered,
        "per_entity_missed": per_entity_missed,
    }


# ---------------------------------------------------------------------------
# Proposed blockers - exact recall and candidate cost, measured not guessed
# ---------------------------------------------------------------------------
# Why these are df-free
# --------------------
# A faithful re-run of production token eligibility needs a target-corpus document
# frequency for every token (~30M distinct keys) and a trigram df for char n-grams
# (~100M) - several GB of a python dict, rebuilt per experiment, to answer a question
# that does not need it. Instead each proposal is a pure key function, and the df cap
# is measured *by difference*: the gap between a df-free key over all tokens/trigrams
# and the production blocker's own recall is exactly what the cap and ``rarest_k``
# cost. That keeps every number here exact while holding resident memory to O(n_s1).
def _tokens(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(values[0].split())


def _sorted_tokens(values: Sequence[str]) -> tuple[str, ...]:
    tokens = values[0].split()
    return (" ".join(sorted(set(tokens))),) if tokens else ()


def _initials(values: Sequence[str]) -> tuple[str, ...]:
    letters = "".join(token[0] for token in values[0].split() if token)
    return (letters,) if letters else ()


def _prefix(width: int) -> Callable[[Sequence[str]], tuple[str, ...]]:
    def keys(values: Sequence[str]) -> tuple[str, ...]:
        key = values[1][:width]
        return (key,) if key else ()

    return keys


def _suffix_stripped(values: Sequence[str]) -> tuple[str, ...]:
    stripped = strip_legal_suffixes(tuple(values[0].split()))
    joined = " ".join(sorted(set(stripped)))
    return (joined,) if joined else ()


def _suffix_stripped_prefix(values: Sequence[str]) -> tuple[str, ...]:
    joined = _suffix_stripped(values)
    key = joined[0].replace(" ", "")[:6] if joined else ""
    return (key,) if key else ()


def _digits_stripped(values: Sequence[str]) -> tuple[str, ...]:
    key = _strip_digits(values[1])
    return (key,) if key else ()


def _address_tokens(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(values[2].split())


def _country_prefix(values: Sequence[str]) -> tuple[str, ...]:
    if not values[3] or not values[1]:
        return ()
    return (values[3] + "#" + values[1][:6],)


def _ngrams(width: int) -> Callable[[Sequence[str]], tuple[str, ...]]:
    def keys(values: Sequence[str]) -> tuple[str, ...]:
        key = values[1]
        if len(key) < width:
            return (key,) if key else ()
        return tuple({key[start : start + width] for start in range(len(key) - width + 1)})

    return keys


# (label, key function, is_upper_bound, what the row means)
# ``is_upper_bound`` marks rows where production would still apply a verification
# filter this simulation does not reapply, so the recall and row counts are ceilings.
PROPOSALS: tuple[tuple[str, Callable[[Sequence[str]], tuple[str, ...]], bool, str], ...] = (
    ("token_any", _tokens, True, "every name token, no df cap and no rarest_k - the ceiling of token blocking"),
    ("token_sorted", _sorted_tokens, False, "token multiset order removed"),
    ("initials", _initials, False, "first letter of every token"),
    ("prefix4", _prefix(4), False, "first 4 characters of name_key"),
    ("prefix6", _prefix(6), False, "first 6 characters of name_key"),
    ("prefix8", _prefix(8), False, "first 8 characters of name_key"),
    ("suffix_stripped", _suffix_stripped, False, "legal-form tokens removed, remaining tokens sorted"),
    ("suffix_stripped_prefix6", _suffix_stripped_prefix, False, "legal-form tokens removed, first 6 characters"),
    ("digits_stripped", _digits_stripped, False, "digits removed from name_key"),
    ("address_tokens", _address_tokens, False, "any shared address token"),
    ("country_prefix6", _country_prefix, False, "same country and the same first 6 characters"),
    ("char_2gram_any", _ngrams(2), True, "all character bigrams, no df cap - granularity sweep"),
    ("char_trigram_any", _ngrams(3), True, "all character trigrams, no df cap - the df-free char ceiling"),
    ("char_4gram_any", _ngrams(4), True, "all character 4-grams, no df cap - granularity sweep"),
)


def collect_entity_keys(
    entity_names: dict[int, tuple[str, ...]],
    key_fn: Callable[[Sequence[str]], tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-entity key hashes as ``(codes_sorted, offsets, hashes)``.

    Only 64-bit hashes are kept, never the key strings: a trigram proposal over the
    ground-truth entities is ~250M incidences, which is 2GB as int64 and hopeless as
    python strings. Consistency within one process is all the hashes must have, since
    both sides of every comparison are hashed here - which is why the built-in
    ``hash`` is enough and no cryptographic digest is needed.

    The buffer grows geometrically and is sliced once at the end, so keys are computed
    exactly once per entity rather than once to size the buffer and once to fill it.
    """
    codes = np.fromiter(entity_names.keys(), dtype=np.int64, count=len(entity_names))
    codes.sort()
    offsets = np.zeros(len(codes) + 1, dtype=np.int64)
    hashes = np.empty(max(len(codes) * 8, 1), dtype=np.int64)
    cursor = 0
    for index, code in enumerate(codes):
        for key in key_fn(entity_names[int(code)]):
            if cursor == len(hashes):
                hashes = np.resize(hashes, len(hashes) * 2)
            hashes[cursor] = hash(key)
            cursor += 1
        offsets[index + 1] = cursor
    return codes, offsets, hashes[:cursor]


def _chunk_key_hashes(
    frame: pd.DataFrame,
    key_fn: Callable[[Sequence[str]], tuple[str, ...]],
) -> np.ndarray:
    """Hash every key of every row in one chunk, as a single int64 array."""
    rows = frame.itertuples(index=False, name=None)
    keys = [hash(key) for row in rows for key in key_fn(row)]
    return np.fromiter(keys, dtype=np.int64, count=len(keys))


def s1_key_space(
    config: dict,
    prepared_split: str,
    proposals: Sequence[tuple[str, Callable[[Sequence[str]], tuple[str, ...]], bool, str]],
    log: logging.Logger,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Distinct S1 keys and their entity counts, per proposal, for candidate growth.

    Growth is ``sum_k n_s1(k) * n_target(k)`` over keys the two sides share. Holding the
    S1 side as sorted unique hashes plus counts is what makes that a searchsorted join
    during a single streaming pass over the targets, instead of materialising the
    proposal's pairs - which for a 4-character prefix would be billions of rows.
    """
    space: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    columns = [NAME_NORM, NAME_KEY, ADDRESS_NORM, COUNTRY_NORM]
    for label, key_fn, _, _ in proposals:
        started = time.time()
        blocks: list[np.ndarray] = []
        for chunk in iter_prepared(config, prepared_split, "source1", columns=columns):
            block = _chunk_key_hashes(chunk, key_fn)
            if len(block):
                blocks.append(block)
        flat = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
        del blocks
        unique, counts = np.unique(flat, return_counts=True)
        del flat
        space[label] = (unique, counts)
        log.info(
            "  %-24s S1 key space: %s distinct keys (%.1f min)",
            label,
            fmt_int(len(unique)),
            (time.time() - started) / 60.0,
        )
    return space


def estimate_growth(
    config: dict,
    prepared_split: str,
    space: dict[str, tuple[np.ndarray, np.ndarray]],
    proposals: Sequence[tuple[str, Callable[[Sequence[str]], tuple[str, ...]], bool, str]],
    log: logging.Logger,
) -> dict[str, int]:
    """Candidate rows each proposal would create, over the whole target corpus.

    One pass over S2 and S3 computes every proposal's count, because the per-row key
    derivation - the only python-level cost here - is shared. A key that no S1 entity
    carries contributes nothing, so the S1 key space is the join filter, and the join
    itself is one searchsorted per chunk rather than one per key.
    """
    totals = {label: 0 for label, _, _, _ in proposals}
    started = time.time()
    columns = [NAME_NORM, NAME_KEY, ADDRESS_NORM, COUNTRY_NORM]
    for source in ("source2", "source3"):
        for chunk in iter_prepared(config, prepared_split, source, columns=columns):
            for label, key_fn, _, _ in proposals:
                keys = _chunk_key_hashes(chunk, key_fn)
                if not len(keys):
                    continue
                unique, counts = space[label]
                position = np.searchsorted(unique, keys)
                np.clip(position, 0, max(len(unique) - 1, 0), out=position)
                matched = unique[position] == keys
                totals[label] += int(counts[position[matched]].sum())
        log.info("  growth pass finished %s (%.1f min)", source, (time.time() - started) / 60.0)
    return totals


def build_s1_membership(
    s1_offsets: np.ndarray,
    s1_hashes: np.ndarray,
    s1_gt_positions: np.ndarray,
    n_s1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(unique_keys, sorted (key_id, s1_position) membership)`` for pair-level lookups.

    Collapsing ``(key, entity)`` into one int64 turns "do these two entities share any
    key" into a searchsorted probe - no per-pair python, no dict of postings lists, and
    entities absent from the ground truth are dropped rather than silently matching.
    """
    positions = np.repeat(s1_gt_positions, np.diff(s1_offsets))
    unique, key_ids = np.unique(s1_hashes, return_inverse=True)
    keep = positions >= 0
    combined = key_ids[keep].astype(np.int64) * np.int64(n_s1) + positions[keep]
    combined.sort()
    return unique, combined


def pairs_with_shared_key(
    pair_s1_positions: np.ndarray,
    pair_target_codes: np.ndarray,
    target_codes: np.ndarray,
    target_offsets: np.ndarray,
    target_hashes: np.ndarray,
    s1_unique: np.ndarray,
    s1_combined: np.ndarray,
    n_s1: int,
    chunk: int = 500_000,
) -> np.ndarray:
    """Which ground-truth pairs share at least one key - the proposal's exact recall.

    Vectorized per chunk of pairs: build the ragged (pair, target-key) incidence
    arrays, probe the S1 membership set, and reduce hits back to pairs. Nothing is
    held beyond one chunk, and no pair is enumerated individually.
    """
    recovered = np.zeros(len(pair_s1_positions), dtype=bool)
    if not len(s1_combined):
        return recovered
    for start in range(0, len(pair_s1_positions), chunk):
        stop = min(start + chunk, len(pair_s1_positions))
        wanted = pair_target_codes[start:stop]
        target_position = np.searchsorted(target_codes, wanted)
        np.clip(target_position, 0, max(len(target_codes) - 1, 0), out=target_position)
        valid = (len(target_codes) > 0) & (target_codes[target_position] == wanted) if len(target_codes) else np.zeros(len(wanted), dtype=bool)
        low = target_offsets[target_position]
        high = np.where(valid, target_offsets[target_position + 1], low)
        counts = high - low
        total = int(counts.sum())
        if total == 0:
            continue
        pair_index = np.repeat(np.arange(stop - start), counts)
        starts = np.repeat(low, counts)
        inner = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        keys = target_hashes[starts + inner]
        key_id = np.searchsorted(s1_unique, keys)
        np.clip(key_id, 0, max(len(s1_unique) - 1, 0), out=key_id)
        found = s1_unique[key_id] == keys
        if not found.any():
            continue
        pair_index = pair_index[found]
        combined = key_id[found].astype(np.int64) * np.int64(n_s1) + pair_s1_positions[start:stop][pair_index]
        hit = np.searchsorted(s1_combined, combined)
        np.clip(hit, 0, max(len(s1_combined) - 1, 0), out=hit)
        matched = s1_combined[hit] == combined
        if matched.any():
            recovered[start:stop][np.unique(pair_index[matched])] = True
    return recovered


def metric_chain(
    candidate_pair_recall: float,
    candidate_macro_recall: float,
    proposal_macro_recall: Optional[float],
    matcher_retention: float,
    v1_pair_recall: Optional[float],
    v1_macro_f05: Optional[float],
) -> dict[str, Any]:
    """The A/B/C distinction, kept as three separate numbers.

    A (candidate recall) is measured. B (matcher recall) multiplies it by the retention
    V1 already demonstrated - a *constant* assumption, and an optimistic one: retention
    was measured on the pairs the blockers already find, and the pairs a new blocker
    adds are the long tail, which a lexical matcher is likely to score worse. C is the
    F0.5 ceiling at that recall, not a prediction of F0.5: closing the gap to the
    ceiling needs precision, which no recall change supplies.

    Returns the three stages plus the explicit reason they are not interchangeable.
    """
    chain: dict[str, Any] = {
        "A_candidate_pair_recall": candidate_pair_recall,
        "A_candidate_macro_entity_recall": candidate_macro_recall,
        "A_candidate_f05_ceiling": _f05_ceiling(candidate_macro_recall),
        "matcher_retention_applied": matcher_retention,
        "matcher_retention_source": "V1 measured: 781,785 TP / (0.570374 * 1,530,245 blocked-true pairs)",
        "A_to_C_is_not_an_equality": (
            "candidate recall is an upper bound on matcher recall, and F0.5 is capped by the "
            "ceiling at that recall; an increase in A cannot be read as an equal increase in C, "
            "because C also needs precision and the matcher is held fixed"
        ),
    }
    if proposal_macro_recall is not None:
        chain["B_proposed_macro_entity_recall"] = proposal_macro_recall
        chain["B_matcher_macro_recall_at_fixed_retention"] = proposal_macro_recall * matcher_retention
        chain["C_f05_ceiling_at_fixed_retention"] = _f05_ceiling(proposal_macro_recall * matcher_retention)
        chain["C_f05_ceiling_uplift"] = chain["C_f05_ceiling_at_fixed_retention"] - _f05_ceiling(
            candidate_macro_recall * matcher_retention
        )
    if v1_pair_recall is not None:
        chain["V1_pair_recall_reference"] = v1_pair_recall
    if v1_macro_f05 is not None:
        chain["V1_macro_f05_reference"] = v1_macro_f05
        chain["V1_share_of_its_own_ceiling"] = _ratio(v1_macro_f05, _f05_ceiling(candidate_macro_recall * matcher_retention))
    chain["caveat"] = (
        "A reliable estimate of the FINAL macro F0.5 requires regenerating features for the "
        "new candidates and re-scoring with the frozen matcher; the numbers above bound it."
    )
    return chain


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def _write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(list(rows)).to_csv(path, sep="\t", index=False)


def _pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def _num(value: Any, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def write_markdown_report(
    path: Path,
    context: dict[str, Any],
    official: dict[str, Any],
    blockers: Sequence[dict[str, Any]],
    combinations: Sequence[dict[str, Any]],
    entity: dict[str, Any],
    missed_summary: dict[str, Any],
    recovered_summary: dict[str, Any],
    growth_rows: Sequence[dict[str, Any]],
    chain: dict[str, Any],
) -> None:
    """Render the seven-part engineering report from the measured numbers.

    Every figure carries its provenance inline - ``measured``, ``upper bound``, or
    ``input`` - because the three read very differently in a decision and a table that
    blurs them is worse than no table.
    """
    lines: list[str] = []
    add = lines.append
    add("# Blocking error analysis")
    add("")
    add(f"- candidate file: `{context['candidates']}`")
    add(f"- split scored: `{context['split']}`  ({fmt_int(entity['entities_scored'])} S1 entities)")
    add(f"- ground truth: {fmt_int(context['n_true_pairs_all'])} pairs over {fmt_int(context['n_entities'])} S1 entities")
    add(f"- matcher retention input: {context['matcher_retention']} (from the V1 report, not measured here)")
    add("")

    add("## 1. Current baseline")
    add("")
    add("Pass 1 numbers come from `CandidateEvaluation` unmodified, so they are the graded")
    add("definition, not a re-derivation.")
    add("")
    add("| metric | value | source |")
    add("|---|---|---|")
    add(f"| candidate rows read | {fmt_int(official['candidate_rows_read'])} | measured |")
    add(f"| candidate true pairs | {fmt_int(official['true_pairs_retrieved'])} | measured |")
    add(f"| true pairs in scope | {fmt_int(official['n_true_pairs'])} | measured |")
    add(f"| candidate pair recall | {_pct(official['blocking_recall_pair'])} | measured |")
    add(f"| candidate pair precision | {_pct(official['candidate_precision'])} | measured |")
    add(f"| macro entity recall | {_num(official['macro_recall_entity'])} | measured |")
    add(f"| F0.5 ceiling at that recall | {_num(official['f05_ceiling_from_macro_recall'])} | bound |")
    add(f"| accept-all macro F0.5 (exclude) | {_num(official['f05_accept_all_macro'])} | measured |")
    add(f"| accept-all macro F0.5 (score_zero) | {_num(official['f05_accept_all_macro_score_zero'])} | measured |")
    add(f"| S1 fully retrieved | {_pct(official['s1_full_recall_rate'])} | measured |")
    add(f"| S1 with zero candidates | {fmt_int(official['n_s1_with_zero_candidates'])} | measured |")
    add(f"| V1 matcher macro F0.5 | {context['v1_macro_f05']} | input |")
    add("")

    add("## 2. Where true matches are lost by blocker")
    add("")
    add("| blocker | true pairs recovered | recall | candidate rows | row precision | union loses if removed | rows where only it fires |")
    add("|---|---|---|---|---|---|---|")
    for row in blockers:
        add(
            f"| {row['blocker']} | {fmt_int(row['true_pairs_recovered'])} | {_pct(row['true_pair_recall'])} | "
            f"{fmt_int(row['candidate_rows'])} | {_pct(row['candidate_precision'])} | "
            f"{fmt_int(row['pairs_union_would_lose_without_it'])} | {fmt_int(row['rows_where_only_this_blocker_fires'])} |"
        )
    add("")
    add("| combination | true pairs | recall | candidate rows | row precision |")
    add("|---|---|---|---|---|")
    for row in combinations:
        add(
            f"| {row['combination']} | {fmt_int(row['true_pairs_recovered'])} | {_pct(row['true_pair_recall'])} | "
            f"{fmt_int(row['candidate_rows'])} | {_pct(row['candidate_precision'])} |"
        )
    add("")

    add("## 3. Ranked failure modes among missed pairs")
    add("")
    add(f"Missed pairs analysed: {fmt_int(missed_summary.get('n_pairs', 0))}; contrast sample of recovered pairs: "
        f"{fmt_int(recovered_summary.get('n_pairs', 0))}.")
    add("")
    add("| primary cause | missed share | recovered share | lift |")
    add("|---|---|---|---|")
    missed_causes = missed_summary.get("primary_cause_share", {})
    recovered_causes = recovered_summary.get("primary_cause_share", {})
    for cause, share in sorted(missed_causes.items(), key=lambda item: -item[1]):
        other = recovered_causes.get(cause, 0.0)
        lift = _ratio(share, other) if other else float("inf")
        add(f"| {cause} | {_pct(share)} | {_pct(other)} | {'inf' if lift == float('inf') else f'{lift:.2f}x'} |")
    add("")
    add("| diagnostic | missed | recovered |")
    add("|---|---|---|")
    for name in DIAGNOSTIC_NUMERIC:
        add(
            f"| {name} (mean) | {_num(missed_summary.get(name, {}).get('mean', 0.0), 4)} | "
            f"{_num(recovered_summary.get(name, {}).get('mean', 0.0), 4)} |"
        )
    for name in DIAGNOSTIC_BOOLEAN:
        add(f"| {name} (share) | {_pct(missed_summary.get(name, 0.0))} | {_pct(recovered_summary.get(name, 0.0))} |")
    add("")

    add("## 4. Entity-level impact")
    add("")
    add(f"- entities with true matches: {fmt_int(entity['entities_with_true_matches'])}")
    add(f"- fully covered: {fmt_int(entity['entities_fully_covered'])}")
    add(f"- partially covered: {fmt_int(entity['entities_partially_covered'])}")
    add(f"- zero recovered: {fmt_int(entity['entities_with_zero_recovered'])}")
    add(f"- zero candidates: {fmt_int(entity['entities_with_zero_candidates'])}")
    add(f"- candidate rows on entities with no true match: {fmt_int(entity['candidate_rows_on_no_match_entities'])}")
    add(f"- worst decile holds {_pct(entity['missed_pairs_worst_decile_share'])} of all missed pairs")
    add("")

    add("## 5. Proposed blockers - measured recall and cost")
    add("")
    add("| proposal | true pairs recovered | recall | candidate rows | growth | row precision | bound |")
    add("|---|---|---|---|---|---|---|")
    for row in growth_rows:
        add(
            f"| {row['proposal']} | {fmt_int(row['true_pairs_recovered'])} | {_pct(row['true_pair_recall'])} | "
            f"{fmt_int(row['candidate_rows'])} | {row['growth_factor']:.3f}x | {_pct(row['candidate_precision'])} | "
            f"{'upper bound' if row['upper_bound'] else 'exact'} |"
        )
    add("")

    add("## 6. The recall chain - A, B and C are different numbers")
    add("")
    add(f"- **A** candidate pair recall (current): {_pct(chain['A_candidate_pair_recall'])}")
    add(f"- **A** candidate macro entity recall (current): {_num(chain['A_candidate_macro_entity_recall'])}")
    add(f"- **A** F0.5 ceiling at that recall: {_num(chain['A_candidate_f05_ceiling'])}")
    add(f"- **B** matcher retention applied: {chain['matcher_retention_applied']} ({chain['matcher_retention_source']})")
    add(f"- **C** F0.5 ceiling at fixed retention: {_num(_f05_ceiling(chain['A_candidate_macro_entity_recall'] * chain['matcher_retention_applied']))}")
    add(f"- {chain['A_to_C_is_not_an_equality']}")
    add(f"- {chain['caveat']}")
    add("")
    add("## 7. Recommendation")
    add("")
    add("Driven by the table in section 5 once the HPC run fills it in: rank proposals by")
    add("incremental true pairs recovered per unit of candidate growth, reject any whose growth")
    add("factor makes the downstream matcher workload infeasible, and confirm the winner by")
    add("regenerating features and re-scoring with the frozen V1 matcher before touching")
    add("production blocking.")
    add("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
# The V1 matcher's own numbers, used only where a measured quantity is impossible
# offline. Both are inputs from the V1 report, not measurements of this script.
V1_PAIR_RECALL = 0.570374
V1_MACRO_F05 = 0.67111
V1_MATCHER_RETENTION = 0.8957


def _mask_pair_array(owner_of_pair: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restrict a per-pair array to pairs whose S1 entity is in ``mask``."""
    return values[mask[owner_of_pair]]


def run_analysis(args: argparse.Namespace, config: dict, log: logging.Logger) -> int:
    """Produce every artifact under ``--output-dir``. Returns a process exit code."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(config, log=log)
    true_pair_codes = build_true_pair_codes(ground_truth)
    lengths = ground_truth.lengths()
    n_entities = ground_truth.n_entities
    owner_of_pair = np.repeat(np.arange(n_entities, dtype=np.int64), lengths)
    log.info("ground truth: %s pairs over %s S1 entities", fmt_int(len(true_pair_codes)), fmt_int(n_entities))

    mask = (
        np.ones(n_entities, dtype=bool)
        if args.split == "all"
        else split_mask_for(ground_truth, config, split=args.split)
    )
    log.info("scoring %s of %s S1 entities (split=%s)", fmt_int(int(mask.sum())), fmt_int(n_entities), args.split)

    candidate_file = Path(args.candidates) if args.candidates else candidates_path(config)
    log.info("candidate file: %s", candidate_file)
    log.info(
        "scored split=%s (mask) reading prepared run=%s", args.split, args.split_candidates
    )

    # -- pass 1: the repository's own evaluator, unmodified --------------------
    official: dict[str, Any] = {}
    if not args.skip_official:
        n_targets = target_record_count(config, log)
        evaluator = CandidateEvaluation(
            ground_truth,
            n_target_records=n_targets,
            k_values=config.get("evaluation", {}).get("k_values", [10, 25, 50, 100, 200]),
            log=log,
        )
        started = time.time()
        evaluator.evaluate_file(candidate_file, chunksize=args.chunksize, max_rows=args.limit_rows)
        log.info("pass 1 finished in %.1f min", (time.time() - started) / 60.0)
        official = evaluator.compute_metrics(s1_mask=mask, split_label=args.split)
        print()
        print(format_report(official))

    # -- pass 2: provenance and per-blocker recovery --------------------------
    started = time.time()
    scan = scan_candidates(
        candidate_file,
        ground_truth,
        true_pair_codes,
        chunksize=args.chunksize,
        max_rows=args.limit_rows,
        log=log,
    )
    log.info("pass 2 finished in %.1f min", (time.time() - started) / 60.0)

    blockers = blocker_table(scan, mask, scan["rows_seen"])
    combinations = combination_table(scan, mask)
    entity = entity_accounting(scan, mask)
    log.info("candidate macro entity recall: %.6f", entity["candidate_macro_entity_recall"])

    recovered_masked = _mask_pair_array(owner_of_pair, scan["recovered"], mask)
    missed_positions = np.nonzero(_mask_pair_array(owner_of_pair, ~scan["recovered"], mask))[0]
    recovered_positions = np.nonzero(recovered_masked)[0]
    log.info("in scope: %s true pairs, %s recovered, %s missed", fmt_int(int(mask[owner_of_pair].sum())), fmt_int(len(recovered_positions)), fmt_int(len(missed_positions)))

    # -- names for diagnostics and proposals ----------------------------------
    s1_codes = encode_entity_ids(pd.Series(ground_truth.entity_ids))
    missed_target_codes = true_pair_codes[missed_positions] % PAIR_MULTIPLIER
    gt_target_codes = true_pair_codes % PAIR_MULTIPLIER
    wanted = {
        "source1": s1_codes[mask] if args.split != "all" else s1_codes,
        "source2": np.concatenate([missed_target_codes, gt_target_codes]) if not args.skip_proposals else missed_target_codes,
        "source3": np.concatenate([missed_target_codes, gt_target_codes]) if not args.skip_proposals else missed_target_codes,
    }
    for source in ("source2", "source3"):
        wanted[source] = wanted[source][(wanted[source] // 10**10) == int(source[-1])]
    names = load_prepared_names(config, args.split_candidates, wanted, list(MISSED_COLUMNS), log)
    del wanted
    log_memory(log, "after prepared name load")

    # -- diagnostics ----------------------------------------------------------
    missed_rows: list[dict[str, Any]] = []
    if not args.skip_diagnostics:
        started = time.time()
        for row in _iter_pair_diagnostics(missed_positions, names, true_pair_codes, s1_codes):
            missed_rows.append(row)
        log.info("diagnosed %s missed pairs in %.1f min", fmt_int(len(missed_rows)), (time.time() - started) / 60.0)

    recovered_rows: list[dict[str, Any]] = []
    if args.contrast_sample and len(recovered_positions):
        step = max(1, len(recovered_positions) // args.contrast_sample)
        sample = recovered_positions[::step][: args.contrast_sample]
        for row in _iter_pair_diagnostics(sample, names, true_pair_codes, s1_codes):
            recovered_rows.append(row)
        log.info("diagnosed %s recovered pairs as the contrast sample", fmt_int(len(recovered_rows)))

    missed_summary = summarise_diagnostics(missed_rows) if missed_rows else {}
    recovered_summary = summarise_diagnostics(recovered_rows) if recovered_rows else {}

    # -- proposals ------------------------------------------------------------
    selected = [entry for entry in PROPOSALS if not args.proposals or entry[0] in set(args.proposals.split(","))]
    growth_rows: list[dict[str, Any]] = []
    if selected and not args.skip_proposals:
        growth = measure_proposals(
            args, config, ground_truth, scan, owner_of_pair, names, s1_codes, true_pair_codes, mask, selected, log
        )
        growth_rows = growth["rows"]

    # -- metric chain ---------------------------------------------------------
    chain = metric_chain(
        candidate_pair_recall=official.get("blocking_recall_pair", 0.0),
        candidate_macro_recall=entity["candidate_macro_entity_recall"],
        proposal_macro_recall=None,
        matcher_retention=args.matcher_retention,
        v1_pair_recall=V1_PAIR_RECALL,
        v1_macro_f05=args.v1_macro_f05,
    )

    # -- artifacts ------------------------------------------------------------
    context = {
        "candidates": str(candidate_file),
        "split": args.split,
        "split_candidates": args.split_candidates,
        "n_entities": n_entities,
        "n_true_pairs_all": int(len(true_pair_codes)),
        "candidate_rows": scan["rows_seen"],
        "matcher_retention": args.matcher_retention,
        "v1_macro_f05": args.v1_macro_f05,
    }
    report = {
        "context": context,
        "official_metrics": official,
        "scan": {
            "rows_seen": scan["rows_seen"],
            "unknown_s1_rows": scan["unknown_s1_rows"],
            "unlabelled_rows": scan["unlabelled_rows"],
            "evidence_rows": scan["evidence_rows"],
            "flags_total": scan["flags_total"],
            "flags_by_source": scan["flags_by_source"],
            "combination_counts": scan["combination_counts"].tolist(),
            "combination_labels": list(COMBINATION_ORDER) + ["all_three", "none"],
        },
        "blockers": list(blockers),
        "combinations": list(combinations),
        "entity": {key: value for key, value in entity.items() if not key.startswith("per_entity")},
        "metric_chain": chain,
        "proposals": list(growth_rows),
    }
    write_json(output_dir / "blocking_recall_report.json", report)
    if official:
        # Same writer the production evaluator uses, so the baseline numbers can be
        # diffed against a plain evaluate_blocking run without translation.
        save_metrics(official, output_dir / f"candidate_metrics_{args.split}.json")
    _write_tsv(
        output_dir / "blocking_recall_by_blocker.tsv",
        [{"kind": "blocker", **row} for row in blockers] + [{"kind": "combination", **row} for row in combinations],
    )
    _write_tsv(output_dir / "missed_pairs_analysis.tsv", _diagnostic_rows(missed_summary, recovered_summary))
    write_json(
        output_dir / "missed_pairs_summary.json",
        {"missed": missed_summary, "recovered_contrast": recovered_summary, "n_missed_in_scope": int(len(missed_positions))},
    )
    _write_tsv(output_dir / "candidate_growth_estimates.tsv", list(growth_rows))
    _write_tsv(
        output_dir / "missed_pairs_examples.tsv",
        _example_rows(missed_rows, names, true_pair_codes, missed_positions, s1_codes, args.examples),
    )
    write_markdown_report(
        output_dir / "blocking_error_report.md",
        context,
        official,
        blockers,
        combinations,
        entity,
        missed_summary,
        recovered_summary,
        growth_rows,
        chain,
    )
    log.info("artifacts written under %s", output_dir)
    log_memory(log, "final")
    return 0


def measure_proposals(
    args: argparse.Namespace,
    config: dict,
    ground_truth: GroundTruth,
    scan: dict[str, Any],
    owner_of_pair: np.ndarray,
    names: dict[int, tuple[str, ...]],
    s1_codes: np.ndarray,
    true_pair_codes: np.ndarray,
    mask: np.ndarray,
    selected: Sequence[tuple[str, Callable[[Sequence[str]], tuple[str, ...]], bool, str]],
    log: logging.Logger,
) -> dict[str, Any]:
    """Measure every proposal: exact recall on the ground truth, and candidate cost.

    Recall and macro entity recall come from one key-intersection pass over the
    ground-truth pairs; candidate volume comes from the ``sum_k n_s1(k) * n_target(k)``
    identity over the full corpora, which needs no enumeration of the proposed pairs.
    """
    log.info("simulating %d proposals", len(selected))
    # S1 side: only the entities in scope, because recall is measured on those pairs.
    in_scope = {int(code): names[int(code)] for code in s1_codes[mask] if int(code) in names}
    all_gt_names = {int(code): names[int(code)] for code in np.unique(true_pair_codes % PAIR_MULTIPLIER) if int(code) in names}
    log.info("proposal inputs: %s S1 entities, %s target entities", fmt_int(len(in_scope)), fmt_int(len(all_gt_names)))

    growth = estimate_growth(
        config, args.split_candidates, s1_key_space(config, args.split_candidates, selected, log), selected, log
    )

    rows: list[dict[str, Any]] = []
    baseline_rows = float(scan["rows_seen"])
    for label, key_fn, is_bound, note in selected:
        started = time.time()
        s1_codes_sorted, s1_offsets, s1_hashes = collect_entity_keys(in_scope, key_fn)
        s1_gt_positions = ground_truth.positions_of(pd.Series(decode_entity_ids(s1_codes_sorted)))
        s1_unique, s1_combined = build_s1_membership(s1_offsets, s1_hashes, s1_gt_positions, ground_truth.n_entities)
        del s1_hashes

        target_codes, target_offsets, target_hashes = collect_entity_keys(all_gt_names, key_fn)
        recovered = pairs_with_shared_key(
            owner_of_pair,
            true_pair_codes % PAIR_MULTIPLIER,
            target_codes,
            target_offsets,
            target_hashes,
            s1_unique,
            s1_combined,
            ground_truth.n_entities,
        )
        del target_hashes, s1_combined, s1_unique

        in_mask = mask[owner_of_pair]
        proposal_recovered = recovered & in_mask
        n_true_pairs = int(in_mask.sum())
        true_pairs = int(proposal_recovered.sum())
        macro_recall = _macro_entity_recall(
            scan["lengths"][mask],
            np.bincount(owner_of_pair[proposal_recovered], minlength=ground_truth.n_entities)[mask],
        )
        new_pairs = int((proposal_recovered & ~scan["recovered"]).sum())
        candidate_rows = growth.get(label, 0)
        rows.append(
            {
                "proposal": label,
                "true_pairs_recovered": true_pairs,
                "true_pair_recall": _ratio(true_pairs, n_true_pairs),
                "incremental_true_pairs": new_pairs,
                "incremental_recall": _ratio(new_pairs, n_true_pairs),
                "candidate_rows": candidate_rows,
                "growth_factor": _ratio(candidate_rows, baseline_rows),
                "candidate_precision": _ratio(true_pairs, candidate_rows),
                "macro_entity_recall": macro_recall,
                "f05_ceiling_at_fixed_retention": _f05_ceiling(macro_recall * args.matcher_retention),
                "upper_bound": bool(is_bound),
                "note": note,
            }
        )
        log.info(
            "  %-24s recall %s (incremental %s), rows %s, growth %.3fx (%.1f min)",
            label,
            _pct(rows[-1]["true_pair_recall"]),
            _pct(rows[-1]["incremental_recall"]),
            fmt_int(candidate_rows),
            rows[-1]["growth_factor"],
            (time.time() - started) / 60.0,
        )
    rows.sort(key=lambda row: -row["incremental_true_pairs"])
    return {"rows": rows}


def _diagnostic_rows(missed: dict[str, Any], recovered: dict[str, Any]) -> list[dict[str, Any]]:
    """Long-form missed-vs-recovered contrast, one row per diagnostic."""
    rows: list[dict[str, Any]] = []
    for name in DIAGNOSTIC_NUMERIC:
        missed_stats = missed.get(name, {})
        recovered_stats = recovered.get(name, {})
        rows.append(
            {
                "diagnostic": name,
                "kind": "numeric",
                "missed_mean": missed_stats.get("mean", 0.0),
                "missed_p50": missed_stats.get("p50", 0.0),
                "missed_p90": missed_stats.get("p90", 0.0),
                "missed_share_nonzero": missed_stats.get("share_nonzero", 0.0),
                "recovered_mean": recovered_stats.get("mean", 0.0),
                "recovered_share_nonzero": recovered_stats.get("share_nonzero", 0.0),
            }
        )
    for name in DIAGNOSTIC_BOOLEAN:
        rows.append(
            {
                "diagnostic": name,
                "kind": "boolean",
                "missed_share": missed.get(name, 0.0),
                "recovered_share": recovered.get(name, 0.0),
            }
        )
    for cause, share in (missed.get("primary_cause_share") or {}).items():
        rows.append(
            {
                "diagnostic": f"cause:{cause}",
                "kind": "cause",
                "missed_share": share,
                "recovered_share": (recovered.get("primary_cause_share") or {}).get(cause, 0.0),
            }
        )
    return rows


def _example_rows(
    missed_rows: Sequence[dict[str, Any]],
    names: dict[int, tuple[str, ...]],
    true_pair_codes: np.ndarray,
    missed_positions: np.ndarray,
    s1_codes: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    """Representative missed pairs, stratified by primary cause and bounded by ``limit``.

    The user asked for representative examples rather than a dump, so the sample is
    stratified by cause: a few pairs showing each failure mode is what makes the
    categories checkable by eye, and 600k rows of the same would not.

    ``true_pair_codes`` packs a ground-truth *row position*, not an entity code, on the
    S1 side - the target side is a real code. Conflating the two would silently look up
    the wrong name for roughly every entity, so the position is mapped through
    ``s1_codes`` first.
    """
    if not missed_rows or limit <= 0:
        return []
    seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    per_cause = max(1, limit // max(1, len({row["primary_cause"] for row in missed_rows})))
    for index, row in enumerate(missed_rows):
        cause = row["primary_cause"]
        if seen.get(cause, 0) >= per_cause:
            continue
        seen[cause] = seen.get(cause, 0) + 1
        packed = int(true_pair_codes[missed_positions[index]])
        s1_position = packed // PAIR_MULTIPLIER
        target_code = packed % PAIR_MULTIPLIER
        s1_code = int(s1_codes[s1_position])
        values_a = names.get(s1_code, ("", "", "", ""))
        values_b = names.get(target_code, ("", "", "", ""))
        rows.append(
            {
                "primary_cause": cause,
                "s1_id": decode_entity_id(s1_code),
                "target_id": decode_entity_id(target_code),
                "s1_name": values_a[0],
                "target_name": values_b[0],
                "s1_country": values_a[3],
                "target_country": values_b[3],
                "trigram_jaccard": row["trigram_jaccard"],
                "token_jaccard": row["token_jaccard"],
            }
        )
        if len(rows) >= limit:
            break
    return rows


def target_record_count(config: dict, log: logging.Logger) -> Optional[int]:
    """S2+S3 record count from the prepare manifest, or ``None`` if unavailable.

    Mirrors ``scripts/evaluate_blocking.py``: reading the manifest beats rescanning
    1GB of TSVs just to fill a denominator, and a missing manifest omits the ratio
    instead of guessing it.
    """
    manifest_path = Path(config["resolved"]["prepared_dir"]) / "prepare_manifest.json"
    if not manifest_path.is_file():
        log.warning("no prepare manifest at %s; reduction ratio omitted", manifest_path)
        return None
    try:
        manifest = read_json(manifest_path)
    except Exception:
        log.warning("could not parse %s; reduction ratio omitted", manifest_path)
        return None
    total = sum(int(entry.get("rows", 0)) for entry in manifest.get("sources", []) if entry.get("source") in ("source2", "source3"))
    return total or None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline blocking error analysis: where true pairs are lost, and what recovering them costs."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--candidates", default=None, help="candidate TSV path (default: <candidates_dir>/candidate_pairs.tsv)")
    parser.add_argument("--output-dir", default="outputs/experiments/blocking_error_analysis")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "train", "all"],
        help="which S1 entities to score (val matches the V1 matcher's evaluation)",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--split-candidates",
        default="train",
        help=(
            "which prepared/candidate run to read (the data split); independent of "
            "--split, which selects the S1 entities scored. Mirrors evaluate_blocking.py"
        ),
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="max candidate rows to read (smoke tests)")
    parser.add_argument("--skip-official", action="store_true", help="skip the CandidateEvaluation pass")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-proposals", action="store_true", help="skip the proposed-blocker simulation")
    parser.add_argument("--proposals", default=None, help="comma-separated subset of proposal labels")
    parser.add_argument("--examples", type=int, default=200, help="max representative missed pairs to write")
    parser.add_argument("--contrast-sample", type=int, default=200_000, help="recovered pairs to sample for contrast")
    parser.add_argument("--matcher-retention", type=float, default=V1_MATCHER_RETENTION)
    parser.add_argument("--v1-macro-f05", type=float, default=V1_MACRO_F05)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir})
    log = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    set_seed(config.get("project", {}).get("seed", 42))
    log.info("=" * 78)
    log.info("analyze_blocking_errors: offline blocking error analysis")
    log.info(describe_environment(config))
    log.info("=" * 78)
    if args.limit_rows:
        log.warning("--limit-rows is set: every number describes a prefix of the candidate file")
    return run_analysis(args, config, log)


if __name__ == "__main__":
    raise SystemExit(main())
