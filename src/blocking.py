"""Blocking / candidate generation: inverted indexes and the candidate union.

Blocking proposes candidate pairs. It never decides whether a pair is a match -
that is the ranking model's job. So the only thing that matters here is
**recall at a tolerable candidate volume**.

The naive alternative is 2,206,821 x 10,320,219 = 22.8 trillion comparisons,
which is why we index instead: look up each S1 key and read back only the rows
that share it.

The blocker registry
--------------------
Three blockers are implemented, and the candidate set is their **union** - a pair
proposed by any one of them is a candidate, and the union is never an
intersection:

======================  ===========  ===============================================
blocker                 S1 key       retrieval rule
======================  ===========  ===============================================
``exact_name``          name_norm    exact key equality
``token``               name_norm    share an eligible token (``str.split()``)
``char_ngram``          name_key    share a character trigram, then Jaccard >= 0.3
======================  ===========  ===============================================

The two multi-key blockers implement the rule measured by the calibration
scripts, and the semantics live here in one place:

* **token.** ``str.split()`` on ``name_norm``; distinct tokens; the document
  frequency is counted over the *target* corpus, so the target source decides
  which tokens are rare; a token is eligible when ``0 < df <= df_cap``; the
  entity keeps its ``rarest_k`` rarest eligible tokens; sharing one is the whole
  decision, so there is **no** verification stage.
* **char_ngram.** character trigrams of ``name_key``; df cap and rarest-K the
  same way; retrieval through the inverted postings; then verified with the
  Phase 0.1 trigram Jaccard (``scripts/analyze_name_differences._trigram_jaccard``,
  reproduced bit-for-bit by :func:`_trigram_jaccard`) against a threshold.

Both codes are **injective** - dense token codes are a bijection by construction
and a trigram code packs its three code points into 63 bits - so a key hit is a
key match and neither blocker re-verifies strings after a lookup. That is the one
structural difference from :class:`ExactNameIndex`, which hashes and therefore
must compare.

``rarest_k`` is applied to the keys an entity keeps, not to the postings a key
returns: an entity spends its K slots on its rarest eligible keys, and the
blocker retrieves through whichever of those keys the target shares. Building a
cell directly is equivalent to building the loosest cell and filtering it down,
because eligibility is a prefix-preserving filter over the ``(df, code)`` order
(see :func:`_rarest_keep_mask`).

Calibrated vs provisional
-------------------------
The default cell in :data:`BLOCKER_SETTING_DEFAULTS` is the configuration that
measured 57.01% pair recall / 57.03% macro recall at 336,056,756 candidates
(exact u token DF<=1000/K=1 u char DF<=1000/K=5/J>=0.3). It is the **provisional
production configuration**. DF=5000 for the char arm is a separate background
experiment and is deliberately not the default here.

Implementation notes
--------------------
The textbook implementation is a ``dict[str, list[int]]``. For S2 that dict
would hold ~4.0M keys with string keys, costing roughly 700MB-1GB of RSS before
postings. Since the challenge dataset is HPC-scale and repeated over two target
sources, this module stores the same information in flat numpy arrays:

============  ======================  ===================  ================
array         dtype                   size (S2, ~5M rows)  purpose
============  ======================  ===================  ================
key_hashes    uint64, sorted          ~32MB (4M uniques)   binary-search lookup
key_offsets   int64                   ~32MB                slice into keys_blob
keys_blob     bytes (concatenated)    ~100MB               exact verification
postings      int64                   ~40MB                entity ids, key-grouped
post_offsets  int64                   ~32MB                slice into postings
============  ======================  ===================  ================

Peak build RSS is roughly 250MB for S2 instead of ~1GB, and lookup is a
``searchsorted`` instead of a dict probe.

**Hashes are verified against the real string on every lookup.** A hash match
that fails string comparison is rejected, so the index can never emit a
spurious pair. The reverse (two distinct keys sharing a 64-bit hash, causing a
missed pair) has probability ~4e-7 at this key count and is accepted: blocking
should fail toward "miss" rather than "wrong candidate".

Union semantics
---------------
Multiple blockers each return postings for the same S1. The candidate set is
their **union**, computed with one ``np.unique`` over packed
``(s1_position, entity_code)`` integers, which also deduplicates for free.

Each blocker may also carry per-pair *evidence* (the char Jaccard, the rarest
shared token's document frequency), which the union keeps aligned to the pairs it
emits. Evidence is descriptive only: it does not change which pairs are proposed,
and nothing here consumes it.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from .data_loader import (
    TARGET_SOURCES,
    SOURCE_PREFIX,
    iter_prepared,
    prepared_path,
    require_file,
)
from .normalization import NAME_KEY, NAME_NORM
from .utils import (
    ID_NUMERIC_MODULUS,
    ID_SOURCE_NAMES,
    decode_entity_ids,
    encode_entity_ids,
    ensure_dir,
    human_bytes,
    read_json,
    stable_hash64,
    write_json,
)

logger = logging.getLogger(__name__)

# On-disk format version. Bump when the layout changes so stale indexes are
# detected instead of silently misread.
INDEX_VERSION = 1
HASH_NAME = "blake2b-64"
META_FILE = "meta.json"
KEYS_FILE = "keys.bin"
KEY_HASHES_FILE = "key_hashes.npy"
KEY_OFFSETS_FILE = "key_offsets.npy"
POSTINGS_FILE = "postings.npy"
POSTINGS_OFFSETS_FILE = "postings_offsets.npy"

# Multi-key index files. ``keys``/``key_df``/``postings*`` are shared by the token
# and char blockers; the rest is blocker-specific.
MKEYS_FILE = "mkeys.npy"
KEY_DF_FILE = "key_df.npy"
POSTING_ROWS_FILE = "posting_rows.npy"
DF_CODES_FILE = "df_codes.npy"
DF_VALUES_FILE = "df_values.npy"
VOCAB_FILE = "vocab.bin"
VOCAB_META_FILE = "vocab_meta.json"
NAME_KEY_BLOB_FILE = "name_key_blob.bin"
NAME_KEY_OFFSETS_FILE = "name_key_offsets.npy"

# Packing factor for (s1_position, entity_code) -> single int64.
# Entity codes are <= 3*10**10 + 10**9 < 3.2e10, so 10**11 separates them
# cleanly; s1_position <= 2.2e6 gives a max packed value of ~2.2e17, well inside
# int64 range. Packing lets np.unique do union + dedupe + sort in one call.
PAIR_MULTIPLIER = 10**11

# Blocker registry. Token / n-gram / dense blockers plug in here as they are
# implemented; the CLI and the index loader are already generic.
BLOCKER_EXACT_NAME = "exact_name"
BLOCKER_TOKEN = "token"
BLOCKER_CHAR_NGRAM = "char_ngram"
BLOCKER_DENSE = "dense"
KNOWN_BLOCKERS = (BLOCKER_EXACT_NAME, BLOCKER_TOKEN, BLOCKER_CHAR_NGRAM, BLOCKER_DENSE)

# The three lexical generators - the calibrated provisional production set.
LEXICAL_BLOCKERS = (BLOCKER_EXACT_NAME, BLOCKER_TOKEN, BLOCKER_CHAR_NGRAM)

# The blockers a union may combine, in the order provenance is built. Fixed order
# so the candidate output does not depend on CLI argument order.
UNION_BLOCKERS = LEXICAL_BLOCKERS + (BLOCKER_DENSE,)

# Which normalized column each blocker keys on when the config does not say. The
# char blocker keys on ``name_key`` (separators removed) because that is the
# column its trigram signal is defined on - defaulting it to ``name_norm`` would
# silently produce a different blocker.
DEFAULT_KEY_FIELDS = {
    BLOCKER_EXACT_NAME: NAME_NORM,
    BLOCKER_TOKEN: NAME_NORM,
    BLOCKER_CHAR_NGRAM: NAME_KEY,
    BLOCKER_DENSE: NAME_NORM,
}

# The provisional production cell, per blocker. These are the values the
# calibration measured; changing them changes the blocker, so they are defaults
# for a fresh config rather than tuning knobs.
BLOCKER_SETTING_DEFAULTS = {
    BLOCKER_TOKEN: {"df_cap": 1000, "rarest_k": 1},
    BLOCKER_CHAR_NGRAM: {"df_cap": 1000, "rarest_k": 5, "jaccard": 0.3},
    # Dense multilingual retrieval. top_k / min_score are NOT calibrated yet - they
    # are the knobs the HPC recall x volume run has to choose, exactly as df_cap /
    # rarest_k / jaccard were chosen for the lexical blockers.
    BLOCKER_DENSE: {
        # A local directory in production (no network at runtime); a hub id only
        # together with local_files_only=false, i.e. on a machine allowed to download.
        "model_name_or_path": "BAAI/bge-m3",
        "backend": "sentence_transformers",
        "local_files_only": True,
        # Business names are short; 64 tokens covers them and is ~8x cheaper than
        # bge-m3's 8192 default.
        "max_length": 64,
        "batch_size": 256,
        # auto -> compute.device (CUDA when present, else CPU).
        "device": "auto",
        # FAISS index_factory string over inner product (= cosine on normalized
        # vectors). "Flat" is exact; at 5M x 1024 prefer e.g. "IVF16384,Flat" or
        # "HNSW32" once the recall cost of approximation has been measured.
        "faiss_factory": "Flat",
        "nprobe": 32,
        "train_sample": 500_000,
        "top_k": 20,
        # Retrieval is recall-first: the matcher does the precision work. bge-m3 puts
        # true cross-script pairs at ~0.67-0.84 cosine (synthetic smoke test; "राम
        # मार्केटिंग प्राइवेट लिमिटेड" vs "ram marketing private limited" = 0.758), so a
        # 0.75 floor would drop most of the pairs this blocker exists for.
        "min_score": 0.60,
    },
}

# Settings that change what an index *contains*. A persisted index whose recorded
# value differs from the config's is a different index and must be rebuilt. The
# remaining dense settings (top_k, min_score, nprobe, batch_size, device,
# local_files_only) are applied at query time, so changing them needs no rebuild.
BUILD_IDENTITY_SETTINGS = {
    BLOCKER_TOKEN: ("df_cap", "rarest_k"),
    BLOCKER_CHAR_NGRAM: ("df_cap", "rarest_k", "jaccard"),
    BLOCKER_DENSE: ("model_name_or_path", "backend", "max_length", "faiss_factory"),
}

# Per-pair evidence a blocker can carry into the candidate file, as
# ``blocker -> (column_name, format)``. The column is written **only when that
# blocker is enabled**, so an exact-name-only run keeps the original four-column
# candidate schema. Values are empty for pairs the blocker did not propose.
EVIDENCE_COLUMNS = {
    BLOCKER_CHAR_NGRAM: ("char_jaccard", "%.4f"),
    BLOCKER_TOKEN: ("token_df", "%.0f"),
    BLOCKER_DENSE: ("dense_cosine", "%.4f"),
}


def evidence_columns_for(blockers: Sequence[str]) -> list[str]:
    """Candidate-file columns the enabled blockers add, in registry order."""
    return [EVIDENCE_COLUMNS[b][0] for b in UNION_BLOCKERS if b in blockers and b in EVIDENCE_COLUMNS]



def index_dir_for(config: dict, split: str, source: str, blocker: str) -> Path:
    """Directory holding one persisted index."""
    return Path(config["resolved"]["index_dir"]) / f"{split}_{source}_{blocker}"


def _key_field_for(config: dict, blocker: str) -> str:
    """Which normalized column a blocker keys on, from config.

    The default is per blocker (:data:`DEFAULT_KEY_FIELDS`) rather than a single
    ``name_norm`` fallback, because the char blocker's signal is defined on
    ``name_key``: a config that enables it without naming a column must get
    ``name_key``, not a silently different blocker.
    """
    section = config.get("blocking", {}).get(blocker, {}) or {}
    return str(section.get("key") or DEFAULT_KEY_FIELDS.get(blocker, NAME_NORM))


def resolve_blocker_settings(config: dict, blocker: str) -> dict:
    """The (df_cap, rarest_k, jaccard) cell a blocker should be built at.

    Merges ``blocking.<blocker>`` over :data:`BLOCKER_SETTING_DEFAULTS` and
    validates, so a typo in the config fails loudly here instead of quietly
    building a different blocker than the one that was calibrated.

    Args:
        config: loaded config.
        blocker: blocker name from :data:`KNOWN_BLOCKERS`.

    Returns:
        A settings dict, empty for blockers that take no parameters.

    Raises:
        ValueError: on an unknown setting or an out-of-range value.
    """
    defaults = dict(BLOCKER_SETTING_DEFAULTS.get(blocker, {}))
    if not defaults:
        return {}
    section = config.get("blocking", {}).get(blocker, {}) or {}
    unknown = sorted(set(section) - set(defaults) - {"enabled", "key"})
    if unknown:
        raise ValueError(
            f"blocking.{blocker} has unknown setting(s) {unknown}; expected "
            f"{sorted(defaults)}"
        )
    settings = {name: section.get(name, default) for name, default in defaults.items()}
    if blocker == BLOCKER_DENSE:
        return _validate_dense_settings(config, settings)

    for name in ("df_cap", "rarest_k"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"blocking.{blocker}.{name} must be a positive integer, got {value!r}")
        settings[name] = int(value)
    if "jaccard" in settings:
        value = settings["jaccard"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
            raise ValueError(
                f"blocking.{blocker}.jaccard must be in (0, 1], got {value!r}"
            )
        settings["jaccard"] = float(value)
    return settings


def _validate_dense_settings(config: dict, settings: dict) -> dict:
    """Type-check the dense cell and resolve ``device: auto`` through ``compute.device``."""
    for name in ("max_length", "batch_size", "top_k", "nprobe", "train_sample"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"blocking.dense.{name} must be a positive integer, got {value!r}")
    value = settings["min_score"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1.0 <= float(value) <= 1.0:
        raise ValueError(f"blocking.dense.min_score must be a cosine in [-1, 1], got {value!r}")
    settings["min_score"] = float(value)
    if settings["backend"] not in DENSE_BACKENDS:
        raise ValueError(
            f"blocking.dense.backend must be one of {DENSE_BACKENDS}, got {settings['backend']!r}"
        )
    for name in ("model_name_or_path", "faiss_factory"):
        if not isinstance(settings[name], str) or not settings[name]:
            raise ValueError(f"blocking.dense.{name} must be a non-empty string")
    settings["local_files_only"] = bool(settings["local_files_only"])
    if str(settings["device"]) == "auto":
        from .utils import resolve_device_from_config

        settings["device"] = resolve_device_from_config(config)
    return settings


def identity_settings(blocker: str, settings: dict) -> dict:
    """The subset of a resolved cell that defines the index's content."""
    return {name: settings[name] for name in BUILD_IDENTITY_SETTINGS.get(blocker, ()) if name in settings}


def enabled_blockers(config: dict, requested: Optional[str] = None) -> list[str]:
    """Blockers to run: an explicit comma list, else those enabled in config.

    Shared by ``build_indexes.py`` and ``generate_candidates.py`` so the two can
    never disagree about which indexes a run needs. Always returned in
    :data:`UNION_BLOCKERS` order, so the per-pair provenance string does not depend
    on how the list was spelled. An explicit list wins even for blockers the config
    leaves disabled - naming them is the more specific instruction.

    Raises:
        ValueError: on an unknown blocker name.
    """
    if requested:
        named = [b.strip() for b in requested.split(",") if b.strip()]
        unknown = [b for b in named if b not in KNOWN_BLOCKERS]
        if unknown:
            raise ValueError(f"unknown blocker(s) {unknown}; expected from {KNOWN_BLOCKERS}")
        return [b for b in UNION_BLOCKERS if b in set(named)]

    blocking = config.get("blocking", {}) or {}
    active = [b for b in UNION_BLOCKERS if (blocking.get(b, {}) or {}).get("enabled", False)]
    if active:
        return active
    logger.warning("no blocker is enabled in config; falling back to %s", BLOCKER_EXACT_NAME)
    return [BLOCKER_EXACT_NAME]



# ---------------------------------------------------------------------------
# Exact-name index
# ---------------------------------------------------------------------------
class ExactNameIndex:
    """Inverted index from an exact normalized key to entity ids.

    Thread- and process-safe for reads; building is single-process.

    Attributes:
        source: logical source name, e.g. ``"source2"``.
        prefix: id prefix, e.g. ``"S2"``.
        key_field: the normalized column that was indexed.
        key_hashes: sorted uint64 hashes of the unique keys.
        postings: int64 entity id codes, grouped by key.
    """

    __slots__ = (
        "source",
        "prefix",
        "key_field",
        "key_hashes",
        "key_offsets",
        "keys_blob",
        "postings_offsets",
        "postings",
        "n_entities_indexed",
        "n_skipped_empty",
    )

    def __init__(
        self,
        source: str,
        prefix: str,
        key_field: str,
        key_hashes: np.ndarray,
        key_offsets: np.ndarray,
        keys_blob: bytes,
        postings_offsets: np.ndarray,
        postings: np.ndarray,
        n_entities_indexed: int = 0,
        n_skipped_empty: int = 0,
    ) -> None:
        self.source = source
        self.prefix = prefix
        self.key_field = key_field
        self.key_hashes = key_hashes
        self.key_offsets = key_offsets
        self.keys_blob = keys_blob
        self.postings_offsets = postings_offsets
        self.postings = postings
        self.n_entities_indexed = n_entities_indexed
        self.n_skipped_empty = n_skipped_empty

    # -- introspection ------------------------------------------------------
    @property
    def n_unique_keys(self) -> int:
        return len(self.key_hashes)

    @property
    def n_postings(self) -> int:
        return len(self.postings)

    @property
    def average_postings_per_key(self) -> float:
        return self.n_postings / self.n_unique_keys if self.n_unique_keys else 0.0

    def memory_bytes(self) -> int:
        return (
            self.key_hashes.nbytes
            + self.key_offsets.nbytes
            + len(self.keys_blob)
            + self.postings_offsets.nbytes
            + self.postings.nbytes
        )

    def describe(self) -> dict:
        return {
            "source": self.source,
            "key_field": self.key_field,
            "n_entities_indexed": int(self.n_entities_indexed),
            "n_skipped_empty_keys": int(self.n_skipped_empty),
            "n_unique_keys": int(self.n_unique_keys),
            "n_postings": int(self.n_postings),
            "avg_postings_per_key": round(self.average_postings_per_key, 3),
            "max_postings_per_key": int(np.diff(self.postings_offsets).max()) if self.n_unique_keys else 0,
            "index_memory": human_bytes(self.memory_bytes()),
        }

    def key_at(self, position: int) -> str:
        """Decode the unique key stored at ``position``."""
        start = int(self.key_offsets[position])
        end = int(self.key_offsets[position + 1])
        return self.keys_blob[start:end].decode("utf-8")

    def postings_for_position(self, position: int) -> np.ndarray:
        """Entity id codes for the key at ``position`` (a view, no copy)."""
        return self.postings[self.postings_offsets[position] : self.postings_offsets[position + 1]]

    # -- build --------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunk_iterator: Iterable[pd.DataFrame],
        source: str,
        prefix: str,
        key_field: str = "name_norm",
        entity_column: str = "entity_id",
        log: Optional[logging.Logger] = None,
        total_rows: Optional[int] = None,
    ) -> "ExactNameIndex":
        """Build an index from an iterable of prepared chunks.

        Memory: O(rows) transiently - roughly ``16 * rows`` bytes for hashes and
        ids plus the UTF-8 key blob (~25 bytes/row here). For S2 that peaks near
        250MB and drops back after grouping. Chunking keeps the *reader* flat;
        the accumulators are inherent to a single-pass build.

        Args:
            chunk_iterator: yields prepared chunks containing ``entity_column``
                and ``key_field``.
            source: logical source name.
            prefix: id prefix (``"S2"``/``"S3"``).
            key_field: normalized column to index.
            entity_column: id column.
            log: logger for progress.
            total_rows: expected row count, for progress reporting.

        Returns:
            A populated :class:`ExactNameIndex`.
        """
        log = log or logger
        hash_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        blob = bytearray()
        offset_parts: list[np.ndarray] = []

        rows_seen = 0
        rows_kept = 0
        rows_skipped = 0

        for chunk in chunk_iterator:
            if entity_column not in chunk.columns or key_field not in chunk.columns:
                raise KeyError(
                    f"prepared chunk is missing {entity_column!r} or {key_field!r}; "
                    f"found {list(chunk.columns)}"
                )
            rows_seen += len(chunk)

            keys = chunk[key_field].to_numpy(dtype=object)
            ids = chunk[entity_column].to_numpy(dtype=object)

            # Records whose normalized name is empty carry no blocking signal.
            # Indexing them would put every such record in one giant bucket and
            # generate all-pairs noise for zero recall benefit.
            keep_mask = np.fromiter(
                (isinstance(k, str) and len(k) > 0 for k in keys), dtype=bool, count=len(keys)
            )
            skipped = int((~keep_mask).sum())
            if skipped:
                rows_skipped += skipped
                keys = keys[keep_mask]
                ids = ids[keep_mask]
            if len(keys) == 0:
                continue

            rows_kept += len(keys)
            hash_parts.append(stable_hash64(keys))

            # Store globally packed codes (source_code * 10**10 + numeric, see
            # utils.encode_entity_id). Packing the source in - rather than the
            # bare numeric id - is what lets a union across S2 and S3 stay
            # self-describing: every code knows which source it came from, so no
            # separate provenance array is needed to decode it back to "S2-123".
            # Verified against the full dataset: ids have no leading zeros, so
            # this round-trips losslessly.
            id_parts.append(encode_entity_ids(pd.Series(ids)))

            # UTF-8 key blob. Encoded once here and reused for offsets.
            encoded = [k.encode("utf-8") for k in keys]
            lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=len(encoded))
            chunk_offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
            np.cumsum(lengths, out=chunk_offsets[1:])
            chunk_offsets += len(blob)
            offset_parts.append(chunk_offsets[:-1])
            blob.extend(b"".join(encoded))

            if log:
                log.info(
                    "  read %s/%s rows (%s indexable, %s skipped empty)",
                    f"{rows_seen:,}",
                    f"{total_rows:,}" if total_rows else "?",
                    f"{rows_kept:,}",
                    f"{rows_skipped:,}",
                )

            del encoded, lengths, chunk_offsets, keys, ids

        if not hash_parts:
            raise ValueError(f"no indexable rows found for {source} (every {key_field} was empty?)")

        row_offsets = np.concatenate(offset_parts)
        hashes = np.concatenate(hash_parts)
        ids = np.concatenate(id_parts)
        del hash_parts, id_parts, offset_parts

        if log:
            log.info(
                "  grouping %s rows (%s) ...",
                f"{rows_kept:,}",
                human_bytes(hashes.nbytes + ids.nbytes + len(blob)),
            )

        # Sort by hash so identical keys become adjacent.
        order = np.argsort(hashes, kind="stable")
        sorted_hashes = hashes[order]
        sorted_ids = ids[order]

        # Group boundaries: first index of each distinct hash.
        is_new_group = np.empty(len(sorted_hashes), dtype=bool)
        is_new_group[0] = True
        if len(sorted_hashes) > 1:
            np.not_equal(sorted_hashes[1:], sorted_hashes[:-1], out=is_new_group[1:])
        group_starts = np.flatnonzero(is_new_group)

        key_hashes = sorted_hashes[group_starts]
        postings_offsets = np.empty(len(group_starts) + 1, dtype=np.int64)
        postings_offsets[:-1] = group_starts
        postings_offsets[-1] = len(sorted_hashes)
        postings = sorted_ids  # already grouped: rows within a group are contiguous

        # One representative key per group, kept for exact verification at query
        # time. row_offsets holds one entry per row, so a trailing sentinel is
        # needed before it can be used to slice the last row's key.
        representative_rows = order[group_starts]
        row_offsets_full = np.empty(len(row_offsets) + 1, dtype=np.int64)
        row_offsets_full[:-1] = row_offsets
        row_offsets_full[-1] = len(blob)
        rep_starts = row_offsets_full[representative_rows]
        rep_ends = row_offsets_full[representative_rows + 1]

        # Concatenate one representative key per group.
        #
        # This looks like the obvious place for a vectorized gather, and it was
        # implemented and measured as one: a single np.repeat/arange gather over
        # the whole blob. It came out 2.2x SLOWER than this loop (507ms vs 228ms
        # for 1M keys / 27MB). The reason is the index arithmetic: gathering
        # variable-length slices needs one int64 index per output BYTE, so 27MB of
        # keys costs ~650MB of temporaries across index_a/index_b/within. Keys are
        # ~27 bytes, far too short for the per-byte index to amortize. Blocking
        # the gather to bound memory does not help, it just re-pays the setup.
        # A leaner loop (zip over .tolist()) was ~1.1x faster but materializes two
        # 4M-element python int lists (~220MB at S2 scale) - not worth it.
        unique_blob = bytearray()
        unique_offsets = np.empty(len(representative_rows) + 1, dtype=np.int64)
        cursor = 0
        for i in range(len(representative_rows)):
            unique_offsets[i] = cursor
            unique_blob.extend(blob[int(rep_starts[i]) : int(rep_ends[i])])
            cursor += int(rep_ends[i] - rep_starts[i])
        unique_offsets[-1] = cursor

        del blob, row_offsets, row_offsets_full, rep_starts, rep_ends
        del hashes, ids, order, sorted_hashes, sorted_ids, is_new_group, group_starts
        gc.collect()

        index = cls(
            source=source,
            prefix=prefix,
            key_field=key_field,
            key_hashes=key_hashes,
            key_offsets=unique_offsets,
            keys_blob=bytes(unique_blob),
            postings_offsets=postings_offsets,
            postings=postings,
            n_entities_indexed=rows_kept,
            n_skipped_empty=rows_skipped,
        )

        if log:
            log.info(
                "  built %s index: %s rows -> %s unique keys (%s), skipped %s empty",
                source,
                f"{rows_kept:,}",
                f"{index.n_unique_keys:,}",
                human_bytes(index.memory_bytes()),
                f"{rows_skipped:,}",
            )
        return index

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> Path:
        """Persist the index as flat files in ``directory``. Returns the path."""
        directory = ensure_dir(directory)
        np.save(directory / KEY_HASHES_FILE, self.key_hashes)
        np.save(directory / KEY_OFFSETS_FILE, self.key_offsets)
        np.save(directory / POSTINGS_FILE, self.postings)
        np.save(directory / POSTINGS_OFFSETS_FILE, self.postings_offsets)
        with open(directory / KEYS_FILE, "wb") as handle:
            handle.write(self.keys_blob)
        write_json(
            directory / META_FILE,
            {
                "index_version": INDEX_VERSION,
                "hash": HASH_NAME,
                "blocker": BLOCKER_EXACT_NAME,
                "source": self.source,
                "prefix": self.prefix,
                "key_field": self.key_field,
                "n_entities_indexed": int(self.n_entities_indexed),
                "n_skipped_empty_keys": int(self.n_skipped_empty),
                "n_unique_keys": int(self.n_unique_keys),
                "n_postings": int(self.n_postings),
                "files": {
                    "key_hashes": KEY_HASHES_FILE,
                    "key_offsets": KEY_OFFSETS_FILE,
                    "keys_blob": KEYS_FILE,
                    "postings": POSTINGS_FILE,
                    "postings_offsets": POSTINGS_OFFSETS_FILE,
                },
            },
        )
        return directory

    @classmethod
    def load(cls, directory: str | os.PathLike, log: Optional[logging.Logger] = None) -> "ExactNameIndex":
        """Load a persisted index, validating the format version."""
        directory = Path(directory)
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no index at {directory}\n  Run: python scripts/build_indexes.py"
            )
        meta = read_json(meta_path)
        version = meta.get("index_version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"index at {directory} has version {version}, expected {INDEX_VERSION}. "
                f"Rebuild it (delete the directory and rerun build_indexes.py)."
            )
        if meta.get("hash") != HASH_NAME:
            raise ValueError(f"index hash scheme {meta.get('hash')!r} != {HASH_NAME!r}; rebuild the index")

        with open(directory / KEYS_FILE, "rb") as handle:
            keys_blob = handle.read()

        index = cls(
            source=meta["source"],
            prefix=meta["prefix"],
            key_field=meta["key_field"],
            key_hashes=np.load(directory / KEY_HASHES_FILE),
            key_offsets=np.load(directory / KEY_OFFSETS_FILE),
            keys_blob=keys_blob,
            postings_offsets=np.load(directory / POSTINGS_OFFSETS_FILE),
            postings=np.load(directory / POSTINGS_FILE),
            n_entities_indexed=int(meta.get("n_entities_indexed", 0)),
            n_skipped_empty=int(meta.get("n_skipped_empty_keys", 0)),
        )
        if log:
            log.info("loaded index %s: %s", directory.name, index.describe())
        return index

    # -- query --------------------------------------------------------------
    _EMPTY = np.empty(0, dtype=np.int64)

    def _positions_for_hashes(self, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Binary-search hashes. Returns (position, found_mask) with -1 for misses."""
        n_keys = self.n_unique_keys
        if n_keys == 0:
            return np.full(len(hashes), -1, dtype=np.int64), np.zeros(len(hashes), dtype=bool)

        positions = np.searchsorted(self.key_hashes, hashes).astype(np.int64)
        in_range = positions < n_keys
        clipped = np.minimum(positions, n_keys - 1)
        found = in_range & (self.key_hashes[clipped] == hashes)
        positions = np.where(found, clipped, -1)
        return positions, found

    def _verify_strings(self, positions: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Confirm hash matches by comparing the stored key strings.

        Decodes each distinct position once, so the common case (many S1 sharing
        a key) costs a handful of decodes rather than one per query.

        WHY THIS IS NOT VECTORIZED. It was rewritten to compare UTF-8 byte slices
        straight out of the blob (one flat uint8 view, np.repeat/arange index
        arithmetic, reduceat over the mismatches) and measured against this
        version: 277ms vs 113ms for 300k queries over 180k keys, i.e. 2.5x
        SLOWER. Keys average ~20 bytes, so a per-byte comparison needs ~8 bytes of
        int64 index per compared byte - three index arrays plus the mismatch
        vector cost far more to build than the 300k short-string comparisons they
        replace. An intermediate variant that kept ``str`` comparison but let
        numpy do it in C via a ``<U`` array was also slower (153ms).
        ``str.__eq__`` on short strings is already near-optimal; leaving this
        alone is the measured-best choice, not an oversight.
        """
        valid = positions >= 0
        if not valid.any():
            return valid
        uniq_positions, inverse = np.unique(positions[valid], return_inverse=True)
        decoded = np.array([self.key_at(int(p)) for p in uniq_positions], dtype=object)
        matched = np.empty(len(positions), dtype=bool)
        matched[:] = False
        matched[valid] = decoded[inverse] == keys[valid]
        return matched

    def lookup_codes(self, key: str) -> np.ndarray:
        """Entity id codes for an exact key match. Empty array when absent."""
        if not key:
            return self._EMPTY
        hashed = np.array([stable_hash64(key)], dtype=np.uint64)
        positions, found = self._positions_for_hashes(hashed)
        if not found[0] or self.key_at(int(positions[0])) != key:
            return self._EMPTY
        return self.postings_for_position(int(positions[0]))

    def lookup_many(self, keys: pd.Series | Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Batched lookup returning ``(positions, counts)``.

        ``positions[i]`` is the unique-key position for ``keys[i]`` (-1 on miss)
        and ``counts[i]`` the number of postings. Empty keys always miss.

        Memory: O(len(keys)) int64s - 2.2M S1 entities is ~35MB.
        """
        key_array = keys.to_numpy(dtype=object) if isinstance(keys, pd.Series) else np.asarray(keys, dtype=object)
        non_empty = np.fromiter(
            (isinstance(k, str) and len(k) > 0 for k in key_array), dtype=bool, count=len(key_array)
        )
        positions = np.full(len(key_array), -1, dtype=np.int64)
        counts = np.zeros(len(key_array), dtype=np.int64)
        if not non_empty.any():
            return positions, counts

        hashed = stable_hash64(key_array[non_empty])
        found_positions, found = self._positions_for_hashes(hashed)
        verified = self._verify_strings(found_positions, key_array[non_empty])
        final_positions = np.where(found & verified, found_positions, -1)

        # Sized to the non-empty subset, not the whole query: sizing it to the query
        # crashed any chunk holding an empty key (a name that normalizes to "").
        subset_counts = np.zeros(len(final_positions), dtype=np.int64)
        valid = final_positions >= 0
        if valid.any():
            offset_index = final_positions[valid]
            subset_counts[valid] = (
                self.postings_offsets[offset_index + 1] - self.postings_offsets[offset_index]
            )

        positions[non_empty] = final_positions
        counts[non_empty] = subset_counts
        return positions, counts

    def expand(self, positions: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand CSR-style ``(positions, counts)`` into flat postings.

        Returns ``(owner_index, entity_codes)`` where ``owner_index[i]`` is the
        query row that produced ``entity_codes[i]``. Uses the standard
        repeat/arange trick, so it is fully vectorized.

        Memory: O(total candidates) - which is exactly why candidate volume is
        tracked and capped rather than assumed small.
        """
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        valid = positions >= 0
        start_offsets = self.postings_offsets[positions[valid]]
        starts_flat = np.repeat(start_offsets, counts[valid])
        group_base = np.repeat(np.cumsum(counts[valid]) - counts[valid], counts[valid])
        within_group = np.arange(total, dtype=np.int64) - group_base
        flat_indices = starts_flat + within_group
        entity_codes = self.postings[flat_indices]
        owner_index = np.repeat(np.flatnonzero(valid), counts[valid])
        return owner_index, entity_codes

    def query(self, values: Sequence[str] | pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Blocking keys for one S1 chunk, packed and deduplicated.

        The uniform blocker entry point: every index in :data:`INDEX_BUILDERS`
        exposes ``query(values) -> (packed_pairs, evidence)``, where
        ``packed_pairs`` is the sorted unique ``pack_pairs(owner_row, entity_code)``
        array for this chunk and ``evidence`` names any per-pair values the blocker
        carries. For exact name matching that is
        :meth:`lookup_many` followed by :meth:`expand` followed by one
        ``np.unique`` - the same three steps the candidate generator has always
        performed - so behaviour is unchanged.

        Args:
            values: the chunk's values of :attr:`key_field`.

        Returns:
            ``(packed_pairs, evidence)``; ``evidence`` is always empty here.
        """
        positions, counts = self.lookup_many(values)
        owner, codes = self.expand(positions, counts)
        if len(codes) == 0:
            return _EMPTY_INT64, {}
        return np.unique(pack_pairs(owner, codes)), {}


# ---------------------------------------------------------------------------
# Key codecs: tokenization and character trigrams
# ---------------------------------------------------------------------------
# Both blockers key on codes rather than strings, and both codecs are injective,
# so a key hit is a key match and neither needs ExactNameIndex's post-lookup string
# comparison. The two functions below are the production home of the semantics the
# calibration scripts measured; tests/test_production_blocking.py pins them to the
# calibration's own functions so the two cannot drift.

_MISSING_TOKEN_CODE = -1
_EMPTY_INT64 = np.empty(0, dtype=np.int64)

# Three code points packed into 63 bits: 21 bits each covers the whole Unicode
# range (max code point 0x10FFFF < 2**21), so the packing is a bijection.
_CODE_POINT_BITS = 21
_CODE_POINT_MASK = (1 << _CODE_POINT_BITS) - 1


def tokenize(text: str) -> list[str]:
    """Tokens of an already-normalized field.

    ``str.split()`` with no argument splits on any Unicode whitespace run and
    drops empty fields - Phase 0's ``_token_set`` is exactly ``set(text.split())``.
    A list is returned so the caller keeps the ordering decisions explicit; the
    deduplication happens in :meth:`TokenVocabulary.codes_for_text`, since a
    *document* frequency is what the token blocker is defined on.

    Kept as a named function so the semantics have one home and the tests can
    point at it rather than at ``str.split``.
    """
    return text.split() if text else []


def trigram_codes(text: str) -> np.ndarray:
    """Distinct character-trigram codes of ``text``, sorted ascending.

    Mirrors Phase 0.1's ``{a[i:i+3] for i in range(len(a) - 2)}`` exactly: the same
    sliding window over code points and the same deduplication, since the signal is
    a set overlap. A name shorter than three code points yields no trigrams,
    matching the reference having nothing to intersect - and, practically, meaning
    such a name can never be retrieved and never needs verifying.

    Sorted output is what makes selection deterministic: python's ``set`` iteration
    order over strings varies with ``PYTHONHASHSEED``, so an unsorted set would make
    the index unreproducible across runs.
    """
    points = np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
    if points.size < 3:
        return _EMPTY_INT64
    wide = points.astype(np.int64)
    return np.unique((wide[:-2] << 42) | (wide[1:-1] << 21) | wide[2:])


def decode_trigram_code(code: int) -> str:
    """Inverse of :func:`trigram_codes` for one code, for reporting."""
    return (
        chr((int(code) >> 42) & _CODE_POINT_MASK)
        + chr((int(code) >> 21) & _CODE_POINT_MASK)
        + chr(int(code) & _CODE_POINT_MASK)
    )


def _trigram_codes_for_list(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Codes for many strings at once, plus a flat owner index.

    Returns ``(codes, owners)`` with ``owners[i]`` the row that produced
    ``codes[i]``. One flat pair of arrays beats a list of per-row arrays when the
    caller is going to concatenate them anyway.
    """
    parts: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for index, text in enumerate(texts):
        codes = trigram_codes(text)
        if codes.size:
            parts.append(codes)
            owners.append(np.full(codes.size, index, dtype=np.int64))
    if not parts:
        return _EMPTY_INT64, _EMPTY_INT64
    return np.concatenate(parts), np.concatenate(owners)


def _rank_within_runs(sorted_keys: np.ndarray) -> np.ndarray:
    """Position of each element within its run of equal adjacent values.

    ``sorted_keys`` must be sorted, so equal values are contiguous. This is what
    turns an entity's df-ordered key list into ranks 0, 1, 2, ... and therefore
    what makes a ``rank < K`` filter a prefix of each entity's keys.
    """
    if sorted_keys.size == 0:
        return _EMPTY_INT64
    starts = np.flatnonzero(np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1])))
    spans = np.diff(np.append(starts, sorted_keys.size))
    return np.arange(sorted_keys.size, dtype=np.int64) - np.repeat(starts, spans)


def _rarest_keep_mask(
    owners: np.ndarray,
    codes: np.ndarray,
    dfs: np.ndarray,
    df_cap: int,
    rarest_k: int,
) -> np.ndarray:
    """Which ``(owner, code, df)`` triples a ``(df_cap, rarest_k)`` cell keeps.

    The rule, applied identically on both sides of a lookup:

    1. a key is *eligible* when ``0 < df <= df_cap`` - a key this target corpus
       never saw scores ``df == 0`` and a key that is too common scores above the
       cap;
    2. among its eligible keys an entity keeps the ``rarest_k`` rarest, ordered by
       ``(df ascending, code ascending)`` so the choice is a total order and the
       selection is reproducible run to run.

    Eligibility is applied **before** ranking, which is what the calibration does,
    and it has to be: an absent key scores ``df == 0``, so ranking first would let a
    key that can never be retrieved consume one of the entity's K slots.

    This is also why a build at the tight cell is equivalent to a build at the
    loosest cell filtered down. Let ``E_tight`` be the eligible keys under a tighter
    cap and ``E_loose`` those under a looser one; ``E_tight`` is a subset, and
    because both are ordered by ``(df, code)`` the position of a key inside
    ``E_tight`` equals its position inside ``E_loose`` - every key ordered before it
    has a smaller df, hence is in both. So ``rank_tight(t) == rank_loose(t)`` for
    every ``t`` in ``E_tight``, and truncating the loose selection at the tight K
    keeps exactly the tight selection.

    Args:
        owners: row that produced each entry.
        codes: key code of each entry.
        dfs: document frequency of each entry.
        df_cap: inclusive upper bound on an eligible key's document frequency.
        rarest_k: how many of an entity's rarest eligible keys to keep.

    Returns:
        Boolean mask aligned to the input.
    """
    keep = np.zeros(len(owners), dtype=bool)
    selected = np.flatnonzero((dfs > 0) & (dfs <= df_cap))
    if selected.size == 0:
        return keep
    order = np.lexsort((codes[selected], dfs[selected], owners[selected]))
    ranks = _rank_within_runs(owners[selected][order])
    keep[selected[order[ranks < rarest_k]]] = True
    return keep


def _trigram_jaccard(a: str, b: str) -> float:
    """Jaccard overlap of character trigrams; character sets for short strings.

    Bit-for-bit the Phase 0.1 reference
    (``scripts/analyze_name_differences._trigram_jaccard``), which is the function
    the char blocker's threshold was calibrated against. It is reproduced rather than
    imported because ``src/`` must not depend on ``scripts/``; a test asserts the two
    agree on a fixture, so the duplication cannot drift.

    Two details matter. The comparison is on **characters**, not code points or
    bytes, and the ``len < 3`` branch falls back to bare character sets. The second
    branch is unreachable for a retrieved char pair - a name shorter than three
    characters has no trigram, so it is never in the index, never retrieved and never
    verified - but it is kept so this is the same function, not a similar one.
    """
    if len(a) < 3 or len(b) < 3:
        set_a, set_b = set(a), set(b)
    else:
        set_a = {a[i : i + 3] for i in range(len(a) - 2)}
        set_b = {b[i : i + 3] for i in range(len(b) - 2)}
    if not set_a or not set_b:
        return 0.0
    shared = len(set_a & set_b)
    union = len(set_a) + len(set_b) - shared
    return shared / union if union else 0.0


def _as_object_array(values: Sequence[str] | pd.Series) -> np.ndarray:
    """A chunk's key values as an object array, without copying when possible."""
    array = values.to_numpy(dtype=object) if isinstance(values, pd.Series) else np.asarray(values)
    return array if array.dtype == object else array.astype(object)


def _chunk_factory(chunks: Iterable[pd.DataFrame] | Callable[[], Iterator[pd.DataFrame]]):
    """Normalize a chunk source to a zero-argument factory.

    Multi-pass builders (df counting, then the index build) must be able to read
    the source twice. Passing a bare generator and calling it twice silently yields
    an empty second pass, so builders ask for a factory and the multi-pass ones
    reject a source that produced rows once and then nothing.
    """
    if callable(chunks):
        return chunks

    def _factory() -> Iterator[pd.DataFrame]:
        return iter(chunks)

    return _factory


# ---------------------------------------------------------------------------
# Corpus-relative document frequency
# ---------------------------------------------------------------------------
class DocumentFrequency:
    """Key code -> document frequency, over one target source's corpus.

    Corpus-relative by construction: counted from the prepared target table of this
    run, so each target source has its own notion of which keys are rare. ``codes``
    is sorted, which turns a lookup into one ``searchsorted`` over the whole query
    array.

    A code the table does not carry scores ``0``. That is the correct reading for
    both callers: a trigram this corpus never contained and a token this corpus
    never contained are equally ineligible, and ``df == 0`` is how the blocker says
    so.
    """

    __slots__ = ("codes", "values")

    def __init__(self, codes: np.ndarray, values: np.ndarray) -> None:
        self.codes = codes
        self.values = values

    def __len__(self) -> int:
        return len(self.codes)

    def lookup(self, codes: np.ndarray) -> np.ndarray:
        """Document frequency of each code; ``0`` for a code never seen."""
        if codes.size == 0 or self.codes.size == 0:
            return np.zeros(codes.size, dtype=np.int64)
        positions = np.searchsorted(self.codes, codes).astype(np.int64)
        np.clip(positions, 0, len(self.codes) - 1, out=positions)
        return np.where(self.codes[positions] == codes, self.values[positions], 0)

    def describe(self) -> dict:
        if len(self.values) == 0:
            return {"n_distinct_keys": 0}
        return {
            "n_distinct_keys": int(len(self.codes)),
            "total_occurrences": int(self.values.sum()),
            "max_df": int(self.values.max()),
            "median_df": float(np.median(self.values)),
        }

    def save(self, directory: str | os.PathLike) -> None:
        directory = ensure_dir(directory)
        np.save(Path(directory) / DF_CODES_FILE, self.codes)
        np.save(Path(directory) / DF_VALUES_FILE, self.values)

    @classmethod
    def load(cls, directory: str | os.PathLike) -> "DocumentFrequency":
        directory = Path(directory)
        return cls(codes=np.load(directory / DF_CODES_FILE), values=np.load(directory / DF_VALUES_FILE))


def _count_code_df(
    chunks: Iterable[pd.DataFrame],
    field: str,
    encoder: Callable[[Sequence[str]], tuple[np.ndarray, np.ndarray]],
    log: logging.Logger,
    label: str,
    kind: str,
) -> DocumentFrequency:
    """Count key document frequency over a streamed target table.

    Per-chunk dedup, then one global merge: each chunk contributes its distinct codes
    with counts, and the pieces are concatenated, sorted once and folded with
    ``add.reduceat``. Deduplicating per chunk keeps the merge proportional to the
    distinct key count rather than to the total occurrence count.
    """
    parts_codes: list[np.ndarray] = []
    parts_counts: list[np.ndarray] = []
    rows = 0

    for frame in chunks:
        texts = frame[field].to_numpy(dtype=object)
        rows += len(texts)
        codes, _ = encoder(texts)
        if codes.size:
            unique, counts = np.unique(codes, return_counts=True)
            parts_codes.append(unique)
            parts_counts.append(counts.astype(np.int64))
        log.info("  %s: %s rows scanned for %s df", label, f"{rows:,}", kind)

    if not parts_codes:
        log.warning("%s: no %s found in %s rows", label, kind, f"{rows:,}")
        return DocumentFrequency(_EMPTY_INT64, _EMPTY_INT64)

    all_codes = np.concatenate(parts_codes)
    all_counts = np.concatenate(parts_counts)
    del parts_codes, parts_counts
    gc.collect()

    order = np.argsort(all_codes, kind="stable")
    all_codes, all_counts = all_codes[order], all_counts[order]
    del order
    gc.collect()

    starts = np.flatnonzero(np.concatenate(([True], all_codes[1:] != all_codes[:-1])))
    table = DocumentFrequency(all_codes[starts], np.add.reduceat(all_counts, starts))
    log.info(
        "%s: %s distinct %s over %s rows",
        label,
        f"{len(table):,}",
        kind,
        f"{rows:,}",
    )
    return table


# ---------------------------------------------------------------------------
# Token vocabulary
# ---------------------------------------------------------------------------
class TokenVocabulary:
    """Bijective token -> dense int64 code map, per target source.

    Dense codes in order of first appearance, deliberately not hashes. A hash-keyed
    index must re-verify the string after a lookup because two tokens can collide;
    with a bijection there is nothing to verify, which is exactly the property that
    lets one multi-key index serve both the token and the char blocker.

    Insertion order depends only on corpus order, never on chunk size, so the
    vocabulary is reproducible run to run and a rebuilt index renumbers identically.
    The codes are part of the on-disk format, so they are persisted rather than
    re-derived: a re-derived vocabulary that disagreed by one token would silently
    invalidate every key in the index.
    """

    __slots__ = ("source", "_codes", "_tokens")

    def __init__(self, source: str) -> None:
        self.source = source
        self._codes: dict[str, int] = {}
        self._tokens: list[str] = []

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> list[str]:
        return self._tokens

    def intern(self, token: str) -> int:
        """Existing code for ``token``, assigning the next one if it is new."""
        code = self._codes.get(token)
        if code is None:
            code = len(self._tokens)
            self._codes[token] = code
            self._tokens.append(token)
        return code

    def codes_for_text(self, text: str) -> np.ndarray:
        """Distinct token codes of one text, ascending. Empty when it has no token.

        The dedup is what makes the frequency a DOCUMENT frequency: a name that says
        "acme" twice contributes one count for "acme", because the token signal is
        set overlap and Phase 0's ``_token_set`` is a set.
        """
        tokens = tokenize(text)
        if not tokens:
            return _EMPTY_INT64
        if len(tokens) == 1:
            return np.asarray([self.intern(tokens[0])], dtype=np.int64)
        seen: set[int] = set()
        intern = self.intern
        for token in tokens:
            seen.add(intern(token))
        out = np.fromiter(seen, dtype=np.int64, count=len(seen))
        out.sort()
        return out

    def codes_for_texts(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(codes, owners)`` for a target-corpus pass. Grows the vocabulary.

        The token twin of :func:`_trigram_codes_for_list`: the same contract (codes
        ascending within a row, owners giving the row each code came from), so every
        downstream algorithm consuming it is unchanged.
        """
        parts: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for row, text in enumerate(texts):
            codes = self.codes_for_text(text)
            if codes.size == 0:
                continue
            parts.append(codes)
            owners.append(np.full(codes.size, row, dtype=np.int64))
        if not parts:
            return _EMPTY_INT64, _EMPTY_INT64
        return np.concatenate(parts), np.concatenate(owners)

    def lookup_texts(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, int]:
        """``(codes, owners, n_texts_without_tokens)`` for tokens already known.

        Read-only, and that is the point: a token this source's corpus never contained
        resolves to a negative code instead of being interned, so an S1-only token
        lands in the "absent from this source" bucket (``df == 0``) rather than
        inventing a key. Interning during a query would grow the vocabulary and
        invalidate the index it is being used to query.
        """
        parts: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        empty = 0
        codes_of = self._codes
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                empty += 1
                continue
            # Two absent tokens are two distinct tokens, not one. Giving each its own
            # negative code (real codes are >= 0) stops the dedup from collapsing
            # them, so "acme private limited" against a corpus that knows only "acme"
            # reports two absent tokens rather than one. Every negative code scores 0
            # in DocumentFrequency.lookup, so all of them land in the absent bucket
            # whichever sentinel they got.
            seen: set[int] = set()
            missing: dict[str, int] = {}
            for token in tokens:
                code = codes_of.get(token)
                if code is None:
                    code = missing.get(token)
                    if code is None:
                        code = _MISSING_TOKEN_CODE - len(missing)
                        missing[token] = code
                seen.add(code)
            codes = np.fromiter(seen, dtype=np.int64, count=len(seen))
            codes.sort()
            parts.append(codes)
            owners.append(np.full(codes.size, row, dtype=np.int64))
        if not parts:
            return _EMPTY_INT64, _EMPTY_INT64, empty
        return np.concatenate(parts), np.concatenate(owners), empty

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> None:
        directory = ensure_dir(directory)
        blob = b"\x00".join(token.encode("utf-8") for token in self._tokens)
        (Path(directory) / VOCAB_FILE).write_bytes(blob)
        # The byte count is recorded so ``load`` can detect a truncated blob: a blob
        # cut mid-token still splits into the right number of NUL-separated fields
        # (one short final field), so the token count alone cannot see the damage.
        write_json(
            Path(directory) / VOCAB_META_FILE,
            {"source": self.source, "n_tokens": len(self._tokens), "blob_bytes": len(blob)},
        )

    @classmethod
    def load(cls, directory: str | os.PathLike, source: str) -> "TokenVocabulary":
        directory = Path(directory)
        meta = read_json(directory / VOCAB_META_FILE)
        blob = (directory / VOCAB_FILE).read_bytes()
        vocab = cls(source)
        n_tokens = int(meta["n_tokens"])
        if n_tokens == 0:
            return vocab
        expected_bytes = int(meta["blob_bytes"])
        if len(blob) != expected_bytes:
            raise ValueError(
                f"{directory}: vocabulary blob is {len(blob)} bytes but the meta says "
                f"{expected_bytes}"
            )
        # One split rather than a million slices: the blob is NUL-joined and no token
        # can contain NUL (normalization maps punctuation to spaces), so the split is
        # exact.
        tokens = blob.split(b"\x00")
        if len(tokens) != n_tokens:
            raise ValueError(
                f"{directory}: vocabulary blob holds {len(tokens)} tokens but the meta "
                f"says {n_tokens}"
            )
        vocab._tokens = [token.decode("utf-8") for token in tokens]
        vocab._codes = {token: code for code, token in enumerate(vocab._tokens)}
        return vocab


# ---------------------------------------------------------------------------
# Multi-key inverted index (token and char-ngram)
# ---------------------------------------------------------------------------
def _read_index_meta(directory: Path, blocker: str) -> dict:
    """Read and validate an index's meta.json for a specific blocker."""
    meta_path = directory / META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"no index at {directory}\n  Run: python scripts/build_indexes.py"
        )
    meta = read_json(meta_path)
    version = meta.get("index_version")
    if version != INDEX_VERSION:
        raise ValueError(
            f"index at {directory} has version {version}, expected {INDEX_VERSION}. "
            f"Rebuild it (delete the directory and rerun build_indexes.py)."
        )
    if meta.get("blocker") != blocker:
        raise ValueError(
            f"index at {directory} was built by blocker {meta.get('blocker')!r}, not "
            f"{blocker!r}; the directory is mislabelled - rebuild it"
        )
    return meta


class MultiKeyIndex:
    """CSR inverted index in which one entity contributes many keys.

    The same shape as :class:`ExactNameIndex` - sorted keys, offsets into a flat
    posting array - with the three differences a token or n-gram blocker requires:

    * **multi-key.** One entity contributes many keys, so the query side holds a key
      list per row rather than one key per row. ``ExactNameIndex``'s
      one-key-per-entity layout cannot express that.
    * **exact codes.** Keys are dense token codes or 63-bit trigram codes, never
      hashes. Both encodings are injective, so there is no collision to verify away
      and a lookup needs no string comparison.
    * **key_df.** Every key carries its target-corpus document frequency, which makes
      the ``df_cap`` a mask on this array and the index self-contained.

    Postings within a key are ordered by entity code, so a rebuild of the same corpus
    produces the same arrays. Posting order does not affect the candidate *set* - the
    union sorts by packed pair - but it does have to be a total order for the index to
    be reproducible.

    Attributes:
        keys: sorted unique key codes.
        key_df: document frequency of each key, aligned to ``keys``.
        postings: entity id codes, grouped by key.
        df: the **full** target-corpus document-frequency table, over every code the
            corpus contained and not just the keys the index kept. Queries rank an
            entity's keys against this table, and a key outside the index must still
            score its true df - scoring it 0 would exempt it from ranking, free one of
            the entity's ``rarest_k`` slots, and retrieve candidates a direct build at
            the same cell would not.
    """

    __slots__ = (
        "source",
        "prefix",
        "key_field",
        "keys",
        "key_df",
        "postings_offsets",
        "postings",
        "df_cap",
        "rarest_k",
        "n_entities_indexed",
        "n_rows_without_key",
        "df",
        "directory",
    )

    def __init__(
        self,
        source: str,
        prefix: str,
        key_field: str,
        keys: np.ndarray,
        key_df: np.ndarray,
        postings_offsets: np.ndarray,
        postings: np.ndarray,
        df_cap: int,
        rarest_k: int,
        df: DocumentFrequency,
        n_entities_indexed: int = 0,
        n_rows_without_key: int = 0,
        directory: Optional[Path] = None,
    ) -> None:
        self.source = source
        self.prefix = prefix
        self.key_field = key_field
        self.keys = keys
        self.key_df = key_df
        self.postings_offsets = postings_offsets
        self.postings = postings
        self.df = df
        self.df_cap = int(df_cap)
        self.rarest_k = int(rarest_k)
        self.n_entities_indexed = int(n_entities_indexed)
        self.n_rows_without_key = int(n_rows_without_key)
        self.directory = directory

    # -- introspection ------------------------------------------------------
    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def n_postings(self) -> int:
        return len(self.postings)

    @property
    def average_postings_per_key(self) -> float:
        return self.n_postings / self.n_keys if self.n_keys else 0.0

    def memory_bytes(self) -> int:
        return (
            self.keys.nbytes
            + self.key_df.nbytes
            + self.postings_offsets.nbytes
            + self.postings.nbytes
        )

    def describe(self) -> dict:
        counts = np.diff(self.postings_offsets)
        return {
            "source": self.source,
            "key_field": self.key_field,
            "df_cap": self.df_cap,
            "rarest_k": self.rarest_k,
            "n_entities_indexed": int(self.n_entities_indexed),
            "n_rows_without_key": int(self.n_rows_without_key),
            "n_unique_keys": int(self.n_keys),
            "n_distinct_keys_in_corpus": int(len(self.df)),
            "n_postings": int(self.n_postings),
            "avg_postings_per_key": round(self.average_postings_per_key, 3),
            "max_postings_per_key": int(counts.max()) if counts.size else 0,
            "index_memory": human_bytes(self.memory_bytes()),
        }

    def key_count_at(self, position: int) -> int:
        """Postings held by the key at ``position`` (a CSR slice length)."""
        return int(self.postings_offsets[position + 1] - self.postings_offsets[position])

    # -- query --------------------------------------------------------------
    def positions_for_codes(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Binary search for key codes; returns ``(position, found)``.

        The int64-code twin of ``ExactNameIndex._positions_for_hashes``, minus the
        string re-verification that a hash-based key needs and an injective encoding
        does not.
        """
        if codes.size == 0:
            return _EMPTY_INT64, np.zeros(0, dtype=bool)
        if self.n_keys == 0:
            return np.full(codes.size, -1, dtype=np.int64), np.zeros(codes.size, dtype=bool)
        positions = np.searchsorted(self.keys, codes).astype(np.int64)
        clipped = np.minimum(positions, self.n_keys - 1)
        found = (positions < self.n_keys) & (self.keys[clipped] == codes)
        return np.where(found, clipped, -1), found

    def lookup_many(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched lookup returning ``(positions, counts)``, ``-1`` on a miss."""
        positions, found = self.positions_for_codes(np.asarray(codes, dtype=np.int64))
        counts = np.zeros(positions.size, dtype=np.int64)
        if found.any():
            index = positions[found]
            counts[found] = self.postings_offsets[index + 1] - self.postings_offsets[index]
        return positions, counts

    def _flat_indices(
        self, positions: np.ndarray, counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(owner_index, flat_posting_index)`` for CSR ``(positions, counts)``.

        The one implementation of the repeat/arange expansion; :meth:`expand` and
        every ``query`` are built on it so they cannot disagree. ``owner_index[i]``
        is the position **in the array ``positions`` was indexed by** that produced
        posting ``i`` - the query row when ``positions`` covers every key of every
        row (as in :meth:`expand`), and the position within the looked-up key subset
        when a query first filters its keys down to the rarest ``rarest_k``. A query
        that filtered must map back through its subset before packing, or pairs get
        attributed to the wrong row.
        """
        total = int(counts.sum())
        if total == 0:
            return _EMPTY_INT64, _EMPTY_INT64
        valid = positions >= 0
        start_offsets = self.postings_offsets[positions[valid]]
        starts_flat = np.repeat(start_offsets, counts[valid])
        group_base = np.repeat(np.cumsum(counts[valid]) - counts[valid], counts[valid])
        within_group = np.arange(total, dtype=np.int64) - group_base
        owner_index = np.repeat(np.flatnonzero(valid), counts[valid])
        return owner_index, starts_flat + within_group

    def expand(self, positions: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """CSR expand into ``(owner_index, entity_codes)``.

        Deliberately identical to ``ExactNameIndex.expand`` - the same repeat/arange
        arithmetic producing the same ordering - so candidate ordering downstream
        matches the shipped generator.
        """
        owners, flat = self._flat_indices(positions, counts)
        if owners.size == 0:
            return owners, _EMPTY_INT64
        return owners, self.postings[flat]

    # -- shared query steps -------------------------------------------------
    def _query_keys(
        self, codes: np.ndarray, owners: np.ndarray, dfs: np.ndarray
    ) -> np.ndarray:
        """Positions of the keys a ``(df_cap, rarest_k)`` cell would look up.

        The index's own ``df_cap``/``rarest_k`` are the cell it was built at, so a
        query needs no parameters: an index *is* a blocker configuration.
        """
        keep = _rarest_keep_mask(owners, codes, dfs, self.df_cap, self.rarest_k)
        return np.flatnonzero(keep)

    # -- persistence --------------------------------------------------------
    def _save_common(self, directory: Path) -> None:
        np.save(directory / MKEYS_FILE, self.keys)
        np.save(directory / KEY_DF_FILE, self.key_df)
        np.save(directory / POSTINGS_OFFSETS_FILE, self.postings_offsets)
        np.save(directory / POSTINGS_FILE, self.postings)

    def _common_meta(self, blocker: str, files: dict[str, str], extra: Optional[dict] = None) -> dict:
        meta = {
            "index_version": INDEX_VERSION,
            "blocker": blocker,
            "source": self.source,
            "prefix": self.prefix,
            "key_field": self.key_field,
            "df_cap": int(self.df_cap),
            "rarest_k": int(self.rarest_k),
            "n_entities_indexed": int(self.n_entities_indexed),
            "n_rows_without_key": int(self.n_rows_without_key),
            "n_unique_keys": int(self.n_keys),
            "n_postings": int(self.n_postings),
            "files": {
                "keys": MKEYS_FILE,
                "key_df": KEY_DF_FILE,
                "postings": POSTINGS_FILE,
                "postings_offsets": POSTINGS_OFFSETS_FILE,
                **files,
            },
        }
        if extra:
            meta.update(extra)
        return meta

    @classmethod
    def _load_common(cls, directory: Path):
        return (
            np.load(directory / MKEYS_FILE),
            np.load(directory / KEY_DF_FILE),
            np.load(directory / POSTINGS_OFFSETS_FILE),
            np.load(directory / POSTINGS_FILE),
        )


class TokenIndex(MultiKeyIndex):
    """Blocking on shared rare tokens of ``name_norm``.

    The token signal is **boolean**: sharing one eligible token *is* the blocker's
    decision, so there is no verification stage and no threshold. What makes it
    selective is the rarity rule - a token is eligible only when ``0 < df <= df_cap``
    over the target corpus, and an entity spends its ``rarest_k`` slots on its rarest
    eligible tokens - so "acme" (df in the millions) is never a key, while a rare
    brand token is.
    """

    __slots__ = ("vocabulary",)

    def __init__(self, *args: Any, vocabulary: TokenVocabulary, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vocabulary = vocabulary

    def describe(self) -> dict:
        out = super().describe()
        out["vocabulary_size"] = int(len(self.vocabulary))
        return out

    # -- build --------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunks: Iterable[pd.DataFrame] | Callable[[], Iterator[pd.DataFrame]],
        source: str,
        prefix: str,
        key_field: str = NAME_NORM,
        log: Optional[logging.Logger] = None,
        total_rows: Optional[int] = None,
        df_cap: int = 1000,
        rarest_k: int = 1,
    ) -> "TokenIndex":
        """Build the token index for one target source.

        Two passes over the target table: the first counts token document frequency
        **and builds the vocabulary** (codes are assigned on first sight, so the df
        table and the index that follows share one numbering), the second selects each
        entity's rarest ``rarest_k`` tokens and appends its postings. The df pass has to
        come first for either to mean anything - "rare" is defined relative to the
        corpus the index covers.
        """
        log = log or logger
        factory = _chunk_factory(chunks)
        vocabulary = TokenVocabulary(source)
        label = f"[{source}] token"

        df = _count_code_df(
            factory(), key_field, vocabulary.codes_for_texts, log, label, "tokens"
        )

        parts_keys: list[np.ndarray] = []
        parts_codes: list[np.ndarray] = []
        rows = 0
        indexed_entities = 0
        rows_without_key = 0

        for frame in factory():
            texts = frame[key_field].to_numpy(dtype=object)
            entity_codes = encode_entity_ids(frame["entity_id"])
            codes, owners = vocabulary.codes_for_texts(texts)
            rows += len(texts)

            if codes.size == 0:
                rows_without_key += len(texts)
                continue

            dfs = df.lookup(codes)
            selected = _rarest_keep_mask(owners, codes, dfs, df_cap, rarest_k)
            if not selected.any():
                rows_without_key += len(texts)
                continue

            survivors = np.unique(owners[selected])
            indexed_entities += int(survivors.size)
            rows_without_key += len(texts) - int(survivors.size)

            parts_keys.append(codes[selected])
            parts_codes.append(entity_codes[owners[selected]])

        if not parts_keys:
            log.warning("%s: no indexable tokens; the df cap of %s removed every key", label, df_cap)
            return cls(
                source=source,
                prefix=prefix,
                key_field=key_field,
                keys=_EMPTY_INT64,
                key_df=np.empty(0, dtype=np.int32),
                postings_offsets=np.zeros(1, dtype=np.int64),
                postings=_EMPTY_INT64,
                df_cap=df_cap,
                rarest_k=rarest_k,
                vocabulary=vocabulary,
                df=df,
            )

        keys, postings, offsets = _group_postings(parts_keys, parts_codes)
        log.info(
            "%s: token index built (%s); %s entities indexed, %s rows contributed no "
            "surviving key",
            label,
            f"{len(keys):,} keys / {len(postings):,} postings",
            f"{indexed_entities:,}",
            f"{rows_without_key:,}",
        )
        return cls(
            source=source,
            prefix=prefix,
            key_field=key_field,
            keys=keys,
            key_df=df.lookup(keys).astype(np.int32),
            postings_offsets=offsets,
            postings=postings,
            df_cap=df_cap,
            rarest_k=rarest_k,
            n_entities_indexed=indexed_entities,
            n_rows_without_key=rows_without_key,
            vocabulary=vocabulary,
            df=df,
        )

    # -- query --------------------------------------------------------------
    def query(self, values: Sequence[str] | pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Token candidates for one S1 chunk, plus each pair's rarest shared token df.

        Read-only over the vocabulary: an S1 token this source never contained stays
        absent (``df == 0``) instead of being interned, so it cannot consume one of
        the entity's K slots - which is exactly why eligibility is checked before
        ranking.

        The evidence is the smallest target-corpus df among the entity's kept tokens
        that retrieved the pair, i.e. how rare the shared token actually was. It is
        descriptive; the blocker's decision is unchanged by it.
        """
        texts = _as_object_array(values)
        if texts.size == 0:
            return _EMPTY_INT64, {}
        codes, owners, _ = self.vocabulary.lookup_texts(texts)
        if codes.size == 0:
            return _EMPTY_INT64, {}

        dfs = self.df.lookup(codes)
        take = self._query_keys(codes, owners, dfs)
        if take.size == 0:
            return _EMPTY_INT64, {}

        positions, counts = self.lookup_many(codes[take])
        if int(counts.sum()) == 0:
            return _EMPTY_INT64, {}
        key_owner, flat = self._flat_indices(positions, counts)
        # ``key_owner`` indexes the rarest-K subset, so it is mapped back to the
        # original query row before packing; evidence is read off the subset.
        packed = pack_pairs(owners[take][key_owner], self.postings[flat])
        shared_df = dfs[take][key_owner].astype(np.float64)

        unique_packed, inverse = np.unique(packed, return_inverse=True)
        rarest = np.full(unique_packed.size, np.inf, dtype=np.float64)
        np.minimum.at(rarest, inverse, shared_df)
        return unique_packed, {"token_df": rarest}

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> Path:
        directory = ensure_dir(directory)
        self._save_common(Path(directory))
        self.vocabulary.save(directory)
        self.df.save(directory)
        write_json(
            Path(directory) / META_FILE,
            self._common_meta(
                BLOCKER_TOKEN,
                {"vocab": VOCAB_FILE, "vocab_meta": VOCAB_META_FILE, "df_codes": DF_CODES_FILE, "df_values": DF_VALUES_FILE},
                {"vocabulary_size": int(len(self.vocabulary))},
            ),
        )
        self.directory = Path(directory)
        return Path(directory)

    @classmethod
    def load(cls, directory: str | os.PathLike, log: Optional[logging.Logger] = None) -> "TokenIndex":
        directory = Path(directory)
        meta = _read_index_meta(directory, BLOCKER_TOKEN)
        keys, key_df, postings_offsets, postings = cls._load_common(directory)
        index = cls(
            source=meta["source"],
            prefix=meta["prefix"],
            key_field=meta["key_field"],
            keys=keys,
            key_df=key_df,
            postings_offsets=postings_offsets,
            postings=postings,
            df_cap=int(meta["df_cap"]),
            rarest_k=int(meta["rarest_k"]),
            n_entities_indexed=int(meta.get("n_entities_indexed", 0)),
            n_rows_without_key=int(meta.get("n_rows_without_key", 0)),
            vocabulary=TokenVocabulary.load(directory, meta["source"]),
            df=DocumentFrequency.load(directory),
            directory=directory,
        )
        if log:
            log.info("loaded index %s: %s", directory.name, index.describe())
        return index


def _group_postings(
    parts_keys: Sequence[np.ndarray],
    parts_postings: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort ``(key, posting)`` pairs key-major; return ``(keys, postings, offsets)``.

    Postings are ordered by entity code inside a key. ``lexsort`` is stable and the
    pair is a total order once entity codes are distinct, so the layout is
    reproducible - which is what makes a rebuilt index produce identical arrays.
    """
    all_keys = np.concatenate(parts_keys)
    all_postings = np.concatenate(parts_postings)
    order = np.lexsort((all_postings, all_keys))
    all_keys, all_postings = all_keys[order], all_postings[order]
    starts = np.flatnonzero(np.concatenate(([True], all_keys[1:] != all_keys[:-1])))
    offsets = np.concatenate((starts, [len(all_keys)])).astype(np.int64)
    return all_keys[starts], all_postings, offsets


# ---------------------------------------------------------------------------
# Character n-gram index and its verification stage
# ---------------------------------------------------------------------------
# Jaccard payloads are batched to bound the transient list of decoded strings a
# worker holds; the calibration used the same shape.
VERIFY_CHUNK_PAIRS = 50_000

_CHAR_WORKER_STATE: dict[str, Any] = {}


def _char_verify_init(directory: str) -> None:
    """Record where the target name-key blob lives; load it on first use.

    Passed as a path rather than a value: the blob is hundreds of MB, and pickling it
    to every worker would cost more than the verification it supports. Lazy loading
    also means a worker that only ever sees source2 never pays for source3.
    """
    _CHAR_WORKER_STATE.clear()
    _CHAR_WORKER_STATE["directory"] = directory
    _CHAR_WORKER_STATE["store"] = None


def _char_verify_store() -> tuple[np.ndarray, bytes]:
    store = _CHAR_WORKER_STATE.get("store")
    if store is None:
        directory = Path(_CHAR_WORKER_STATE["directory"])
        store = (
            np.load(directory / NAME_KEY_OFFSETS_FILE),
            (directory / NAME_KEY_BLOB_FILE).read_bytes(),
        )
        _CHAR_WORKER_STATE["store"] = store
    return store


def _target_name_at(offsets: np.ndarray, blob: bytes, row: int) -> str:
    """The ``name_key`` of one target table row, decoded from the blob."""
    return blob[int(offsets[row]) : int(offsets[row + 1])].decode("utf-8")


def _char_verify_chunk(payload: tuple[list[str], np.ndarray]) -> np.ndarray:
    """Exact trigram Jaccard for one payload of pairs.

    Calls :func:`_trigram_jaccard` unchanged, on decoded names, so the threshold a
    query applies is the same signal the calibration measured - including the branch
    that falls back to bare character sets for names shorter than three characters.
    """
    s1_keys, target_rows = payload
    offsets, blob = _char_verify_store()
    out = np.zeros(len(s1_keys), dtype=np.float64)
    for index in range(len(s1_keys)):
        row = int(target_rows[index])
        if row < 0:
            continue
        out[index] = _trigram_jaccard(s1_keys[index], _target_name_at(offsets, blob, row))
    return out


class CharNgramIndex(MultiKeyIndex):
    """Blocking on shared character trigrams of ``name_key``, then Jaccard.

    Retrieval is by rare trigram, because a trigram alone is far too weak a signal to
    propose a pair: the df cap and the rarest-K rule are what keep common trigrams
    ("ing", "the") from pairing half the corpus with the other half. On top of that
    the blocker **verifies**: a retrieved pair is kept only when the trigram Jaccard
    of the two ``name_key`` values reaches the threshold. Retrieval is therefore a
    proposal and the Jaccard is the decision, which is why the threshold is part of
    the blocker's identity and not a downstream ranking knob.

    The target-side ``name_key`` values are stored alongside the postings, so the
    index is self-contained: verification needs both strings, and the S1 side arrives
    with the query.

    Attributes:
        posting_rows: table row of each posting, so a pair can be verified without a
            second lookup structure.
        jaccard_threshold: inclusive Jaccard cut-off for a retrieved pair.
        workers: process count for verification. A runtime property, not semantics -
            results are identical at any worker count.
    """

    __slots__ = (
        "posting_rows",
        "name_key_offsets",
        "name_key_blob",
        "jaccard_threshold",
        "workers",
        "verify_chunk_pairs",
    )

    def __init__(
        self,
        *args: Any,
        posting_rows: np.ndarray,
        name_key_offsets: np.ndarray,
        name_key_blob: bytes,
        jaccard_threshold: float,
        workers: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.posting_rows = posting_rows
        self.name_key_offsets = name_key_offsets
        self.name_key_blob = name_key_blob
        self.jaccard_threshold = float(jaccard_threshold)
        self.workers = max(1, int(workers))
        self.verify_chunk_pairs = VERIFY_CHUNK_PAIRS

    @property
    def n_rows(self) -> int:
        return len(self.name_key_offsets) - 1

    def name_key_at_row(self, row: int) -> str:
        return self.name_key_blob[int(self.name_key_offsets[row]) : int(self.name_key_offsets[row + 1])].decode("utf-8")

    def memory_bytes(self) -> int:
        return super().memory_bytes() + self.posting_rows.nbytes + self.name_key_offsets.nbytes + len(self.name_key_blob)

    def describe(self) -> dict:
        out = super().describe()
        out.update(
            {
                "jaccard": self.jaccard_threshold,
                "n_target_rows": int(self.n_rows),
                "name_key_blob": human_bytes(len(self.name_key_blob)),
            }
        )
        return out

    # -- build --------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunks: Iterable[pd.DataFrame] | Callable[[], Iterator[pd.DataFrame]],
        source: str,
        prefix: str,
        key_field: str = NAME_KEY,
        log: Optional[logging.Logger] = None,
        total_rows: Optional[int] = None,
        df_cap: int = 1000,
        rarest_k: int = 5,
        jaccard: float = 0.3,
    ) -> "CharNgramIndex":
        """Build the trigram index plus the target ``name_key`` store, in two passes.

        Pass one counts trigram document frequency over the target corpus. Pass two
        selects each entity's rarest ``rarest_k`` eligible trigrams, appends its
        postings - carrying the table row so verification can find the name - and
        streams the ``name_key`` values into one blob in table order.

        The blob stays in table order rather than being permuted into posting order:
        permuting millions of variable-length strings at build time would cost more
        than the ``posting_rows`` indirection it would save.
        """
        log = log or logger
        factory = _chunk_factory(chunks)
        label = f"[{source}] char"

        df = _count_code_df(
            factory(),
            key_field,
            lambda texts: _trigram_codes_for_list(texts),
            log,
            label,
            "trigrams",
        )

        parts_keys: list[np.ndarray] = []
        parts_codes: list[np.ndarray] = []
        parts_rows: list[np.ndarray] = []
        blob = bytearray()
        offsets = [0]
        rows = 0
        indexed_entities = 0
        rows_without_key = 0

        for frame in factory():
            texts = frame[key_field].to_numpy(dtype=object)
            entity_codes = encode_entity_ids(frame["entity_id"])
            codes, owners = _trigram_codes_for_list(texts)
            row_base = rows
            rows += len(texts)

            for text in texts:
                blob.extend(text.encode("utf-8"))
                offsets.append(len(blob))

            if codes.size == 0:
                rows_without_key += len(texts)
                continue

            dfs = df.lookup(codes)
            selected = _rarest_keep_mask(owners, codes, dfs, df_cap, rarest_k)
            if not selected.any():
                rows_without_key += len(texts)
                continue

            survivors = np.unique(owners[selected])
            indexed_entities += int(survivors.size)
            rows_without_key += len(texts) - int(survivors.size)

            parts_keys.append(codes[selected])
            parts_codes.append(entity_codes[owners[selected]])
            parts_rows.append(owners[selected] + row_base)

        name_key_offsets = np.asarray(offsets, dtype=np.int64)
        if not parts_keys:
            log.warning("%s: no indexable trigrams; the df cap of %s removed every key", label, df_cap)
            index = cls(
                source=source,
                prefix=prefix,
                key_field=key_field,
                keys=_EMPTY_INT64,
                key_df=np.empty(0, dtype=np.int32),
                postings_offsets=np.zeros(1, dtype=np.int64),
                postings=_EMPTY_INT64,
                df_cap=df_cap,
                rarest_k=rarest_k,
                df=df,
                posting_rows=_EMPTY_INT64,
                name_key_offsets=name_key_offsets,
                name_key_blob=bytes(blob),
                jaccard_threshold=jaccard,
            )
            return index

        all_keys = np.concatenate(parts_keys)
        all_postings = np.concatenate(parts_codes)
        all_rows = np.concatenate(parts_rows)
        order = np.lexsort((all_postings, all_keys))
        all_keys, all_postings, all_rows = all_keys[order], all_postings[order], all_rows[order]
        starts = np.flatnonzero(np.concatenate(([True], all_keys[1:] != all_keys[:-1])))
        postings_offsets = np.concatenate((starts, [len(all_keys)]))

        index = cls(
            source=source,
            prefix=prefix,
            key_field=key_field,
            keys=all_keys[starts],
            key_df=df.lookup(all_keys[starts]).astype(np.int32),
            postings_offsets=postings_offsets,
            postings=all_postings,
            df_cap=df_cap,
            rarest_k=rarest_k,
            df=df,
            n_entities_indexed=indexed_entities,
            n_rows_without_key=rows_without_key,
            posting_rows=all_rows,
            name_key_offsets=name_key_offsets,
            name_key_blob=bytes(blob),
            jaccard_threshold=jaccard,
        )
        log.info(
            "%s: trigram index built (%s, J>=%s); %s entities indexed, %s rows contributed "
            "no surviving key",
            label,
            f"{index.n_keys:,} keys / {index.n_postings:,} postings",
            jaccard,
            f"{indexed_entities:,}",
            f"{rows_without_key:,}",
        )
        return index

    # -- query --------------------------------------------------------------
    def query(self, values: Sequence[str] | pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Char candidates for one S1 chunk, with each pair's Jaccard.

        Retrieve by shared rare trigram, deduplicate to unique pairs, verify every
        pair against the target ``name_key``, then apply the threshold. Deduplicating
        *before* verifying is deliberate: verification is the expensive part, and the
        Jaccard is a function of the pair, so a pair retrieved through three shared
        trigrams needs one evaluation rather than three.
        """
        texts = _as_object_array(values)
        if texts.size == 0:
            return _EMPTY_INT64, {}
        codes, owners = _trigram_codes_for_list(texts)
        if codes.size == 0:
            return _EMPTY_INT64, {}

        dfs = self.df.lookup(codes)
        take = self._query_keys(codes, owners, dfs)
        if take.size == 0:
            return _EMPTY_INT64, {}

        positions, counts = self.lookup_many(codes[take])
        if int(counts.sum()) == 0:
            return _EMPTY_INT64, {}
        key_owner, flat = self._flat_indices(positions, counts)
        # ``key_owner`` indexes the rarest-K subset, so it is mapped back to the
        # original query row before packing.
        packed = pack_pairs(owners[take][key_owner], self.postings[flat])
        unique_packed, first = np.unique(packed, return_index=True)
        s1_positions, _ = unpack_pairs(unique_packed)
        target_rows = self.posting_rows[flat[first]]

        similarities = self._verify(s1_positions, target_rows, texts)
        keep = similarities >= self.jaccard_threshold
        if not keep.any():
            return _EMPTY_INT64, {"char_jaccard": np.empty(0, dtype=np.float64)}
        return unique_packed[keep], {"char_jaccard": similarities[keep]}

    def _verify(self, s1_positions: np.ndarray, target_rows: np.ndarray, texts: np.ndarray) -> np.ndarray:
        """Trigram Jaccard for every pair, sharded, consumed in submission order.

        Chunks are consumed strictly in submission order, so the similarity array is a
        deterministic function of the input regardless of worker count - which is what
        lets ``compute.num_workers`` be a pure performance knob here.
        """
        total = len(target_rows)
        out = np.zeros(total, dtype=np.float64)
        if total == 0:
            return out
        chunk_pairs = max(1, int(self.verify_chunk_pairs))
        spans = [(start, min(start + chunk_pairs, total)) for start in range(0, total, chunk_pairs)]

        def payload_for(start: int, stop: int) -> tuple[list[str], np.ndarray]:
            return [texts[int(position)] for position in s1_positions[start:stop]], target_rows[start:stop]

        if self.workers <= 1 or self.directory is None:
            for start, stop in spans:
                out[start:stop] = self._verify_sequential(
                    s1_positions[start:stop], target_rows[start:stop], texts
                )
            return out

        pending: deque = deque()
        window = max(2, self.workers * 4)
        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_char_verify_init,
            initargs=(str(self.directory),),
        ) as pool:
            for start, stop in spans:
                pending.append((start, stop, pool.submit(_char_verify_chunk, payload_for(start, stop))))
                if len(pending) >= window:
                    begin, end, future = pending.popleft()
                    out[begin:end] = future.result()
            while pending:
                begin, end, future = pending.popleft()
                out[begin:end] = future.result()
        return out

    def _verify_sequential(
        self, s1_positions: np.ndarray, target_rows: np.ndarray, texts: np.ndarray
    ) -> np.ndarray:
        """One shard of Jaccard values, off the in-memory blob.

        The single-worker path, and the only path available to an index that was built
        in memory and never saved - a process pool cannot be told where to find the
        target names if there is no directory. Uses the same :func:`_trigram_jaccard`
        as the worker path, so the two cannot diverge.
        """
        out = np.zeros(len(target_rows), dtype=np.float64)
        offsets, blob = self.name_key_offsets, self.name_key_blob
        for index in range(len(target_rows)):
            row = int(target_rows[index])
            if row < 0:
                continue
            out[index] = _trigram_jaccard(
                texts[int(s1_positions[index])], _target_name_at(offsets, blob, row)
            )
        return out

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> Path:
        directory = ensure_dir(directory)
        self._save_common(Path(directory))
        np.save(Path(directory) / POSTING_ROWS_FILE, self.posting_rows)
        np.save(Path(directory) / NAME_KEY_OFFSETS_FILE, self.name_key_offsets)
        with open(Path(directory) / NAME_KEY_BLOB_FILE, "wb") as handle:
            handle.write(self.name_key_blob)
        self.df.save(directory)
        write_json(
            Path(directory) / META_FILE,
            self._common_meta(
                BLOCKER_CHAR_NGRAM,
                {
                    "posting_rows": POSTING_ROWS_FILE,
                    "name_key_offsets": NAME_KEY_OFFSETS_FILE,
                    "name_key_blob": NAME_KEY_BLOB_FILE,
                    "df_codes": DF_CODES_FILE,
                    "df_values": DF_VALUES_FILE,
                },
                {"jaccard": self.jaccard_threshold, "n_target_rows": int(self.n_rows)},
            ),
        )
        self.directory = Path(directory)
        return Path(directory)

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike,
        log: Optional[logging.Logger] = None,
        workers: int = 1,
    ) -> "CharNgramIndex":
        directory = Path(directory)
        meta = _read_index_meta(directory, BLOCKER_CHAR_NGRAM)
        keys, key_df, postings_offsets, postings = cls._load_common(directory)
        with open(directory / NAME_KEY_BLOB_FILE, "rb") as handle:
            blob = handle.read()
        index = cls(
            source=meta["source"],
            prefix=meta["prefix"],
            key_field=meta["key_field"],
            keys=keys,
            key_df=key_df,
            postings_offsets=postings_offsets,
            postings=postings,
            df_cap=int(meta["df_cap"]),
            rarest_k=int(meta["rarest_k"]),
            n_entities_indexed=int(meta.get("n_entities_indexed", 0)),
            n_rows_without_key=int(meta.get("n_rows_without_key", 0)),
            posting_rows=np.load(directory / POSTING_ROWS_FILE),
            name_key_offsets=np.load(directory / NAME_KEY_OFFSETS_FILE),
            name_key_blob=blob,
            jaccard_threshold=float(meta["jaccard"]),
            workers=workers,
            df=DocumentFrequency.load(directory),
            directory=directory,
        )
        if log:
            log.info("loaded index %s: %s", directory.name, index.describe())
        return index


# ---------------------------------------------------------------------------
# Dense multilingual retrieval (bge-m3 + FAISS)
# ---------------------------------------------------------------------------
# Why this blocker exists: the three lexical generators can reach at most 82.19% of
# true pairs, and 134,718 of the unreachable ones are cross-script (a Devanagari S1
# name against a romanized target). A character signal cannot see those by
# construction; a multilingual encoder maps both spellings near each other.
#
# Semantics: each target name is embedded once (L2-normalized), S1 names are
# embedded per chunk, and every S1 keeps its ``top_k`` nearest targets whose cosine
# is at least ``min_score``. The cosine is carried as ``dense_cosine`` evidence.
DENSE_BACKEND_SENTENCE_TRANSFORMERS = "sentence_transformers"
# Deterministic char-trigram feature hashing. PLUMBING TESTS ONLY: it needs no model
# download, so the index/union/persistence path can be tested anywhere, but it has
# no multilingual knowledge and must never be used for a real run.
DENSE_BACKEND_HASHING = "hashing"
DENSE_BACKENDS = (DENSE_BACKEND_SENTENCE_TRANSFORMERS, DENSE_BACKEND_HASHING)
HASHING_DIM = 256

DENSE_EMBEDDINGS_FILE = "embeddings.npy"
DENSE_FAISS_FILE = "faiss.index"
DENSE_SEARCH_BATCH = 16_384
DENSE_ADD_BATCH = 262_144

DENSE_DEPENDENCY_MESSAGE = (
    "the dense blocker needs torch, sentence-transformers and faiss:\n"
    "  pip install torch sentence-transformers faiss-cpu   # faiss-gpu on a CUDA node"
)

_ENCODERS: dict[tuple, Any] = {}
# One-entry memo of the last query batch: generate_candidates queries the S2 and the
# S3 dense index with the same S1 chunk, and encoding it twice would double the most
# expensive step of the query.
_LAST_QUERY: dict[str, Any] = {"key": None, "embeddings": None}


def _import_faiss():
    try:
        import faiss  # noqa: WPS433 - optional dependency, imported on use
    except ImportError as exc:
        raise ImportError(DENSE_DEPENDENCY_MESSAGE) from exc
    return faiss


class HashingEncoder:
    """Char-trigram feature hashing into ``HASHING_DIM`` dims. Test-only (see above)."""

    def __init__(self, dim: int = HASHING_DIM) -> None:
        self.dim = int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f" {text} "
            for start in range(len(padded) - 2):
                out[row, zlib.crc32(padded[start : start + 3].encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEncoder:
    """A sentence-transformers model (bge-m3 by default), L2-normalized output.

    Loaded strictly from local files when ``local_files_only`` - the challenge
    forbids external calls at runtime, so a compute node must never reach the hub.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        max_length: int,
        batch_size: int,
        local_files_only: bool,
    ) -> None:
        if local_files_only:
            # Belt and braces: the kwarg covers the model files, the variables cover
            # every other hub call (tokenizer configs, telemetry) transformers makes.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(DENSE_DEPENDENCY_MESSAGE) from exc

        location = Path(model_name_or_path).expanduser()
        source = str(location) if location.exists() else model_name_or_path
        try:
            model = SentenceTransformer(source, device=device, local_files_only=local_files_only)
        except OSError as exc:
            raise OSError(
                f"cannot load dense model {model_name_or_path!r} from local files "
                f"(local_files_only={local_files_only}).\n"
                "  Fetch it once on a machine with internet access:\n"
                "    python scripts/fetch_dense_model.py --output /path/to/models/bge-m3\n"
                "  then set blocking.dense.model_name_or_path to that directory."
            ) from exc
        model.max_seq_length = int(max_length)
        if str(device).startswith("cuda"):
            model.half()
        self.model = model
        self.batch_size = int(batch_size)
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        self.dim = int(getter())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def _encoder_key(settings: dict) -> tuple:
    return (
        settings["backend"],
        settings["model_name_or_path"],
        str(settings["device"]),
        int(settings["max_length"]),
        bool(settings["local_files_only"]),
    )


def get_encoder(settings: dict):
    """The encoder for a dense cell, loaded once per process and shared by S2/S3."""
    key = _encoder_key(settings)
    encoder = _ENCODERS.get(key)
    if encoder is None:
        if settings["backend"] == DENSE_BACKEND_HASHING:
            encoder = HashingEncoder()
        else:
            encoder = SentenceTransformerEncoder(
                settings["model_name_or_path"],
                device=str(settings["device"]),
                max_length=int(settings["max_length"]),
                batch_size=int(settings["batch_size"]),
                local_files_only=bool(settings["local_files_only"]),
            )
        _ENCODERS[key] = encoder
    else:
        # Batch size is a runtime knob; honour the latest value without a reload.
        if hasattr(encoder, "batch_size"):
            encoder.batch_size = int(settings["batch_size"])
    return encoder


def _encode_queries(settings: dict, texts: np.ndarray) -> np.ndarray:
    """Encode a query batch, reusing the previous result for an identical batch."""
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x1f")
    key = (_encoder_key(settings), len(texts), digest.hexdigest())
    if _LAST_QUERY["key"] == key:
        return _LAST_QUERY["embeddings"]
    embeddings = get_encoder(settings).encode(texts)
    _LAST_QUERY["key"], _LAST_QUERY["embeddings"] = key, embeddings
    return embeddings


class DenseIndex:
    """FAISS inner-product index over normalized target-name embeddings.

    Attributes:
        postings: packed entity code of each FAISS row (row order = table order of
            the target rows that had a non-empty key).
        settings: the resolved dense cell. Build-identity settings come from the
            index itself; query-time settings (top_k, min_score, nprobe, device,
            batch_size) may be overridden at load without rebuilding.
    """

    def __init__(
        self,
        source: str,
        prefix: str,
        key_field: str,
        postings: np.ndarray,
        faiss_index: Any,
        settings: dict,
        dim: int,
        n_entities_indexed: int = 0,
        n_rows_without_key: int = 0,
        embeddings: Optional[np.ndarray] = None,
        directory: Optional[Path] = None,
    ) -> None:
        self.source = source
        self.prefix = prefix
        self.key_field = key_field
        self.postings = postings
        self.faiss_index = faiss_index
        self.settings = dict(settings)
        self.dim = int(dim)
        self.n_entities_indexed = int(n_entities_indexed)
        self.n_rows_without_key = int(n_rows_without_key)
        self.embeddings = embeddings
        self.directory = directory
        self._gpu_resources = None
        self._apply_search_parameters()

    # -- introspection ------------------------------------------------------
    @property
    def n_keys(self) -> int:
        return int(self.faiss_index.ntotal)

    @property
    def top_k(self) -> int:
        return int(self.settings["top_k"])

    @property
    def min_score(self) -> float:
        return float(self.settings["min_score"])

    def memory_bytes(self) -> int:
        vectors = self.n_keys * self.dim * 4 if "Flat" in self.settings["faiss_factory"] else 0
        stored = self.embeddings.nbytes if self.embeddings is not None else 0
        return int(self.postings.nbytes + vectors + stored)

    def describe(self) -> dict:
        return {
            "source": self.source,
            "key_field": self.key_field,
            "backend": self.settings["backend"],
            "model": self.settings["model_name_or_path"],
            "dim": self.dim,
            "faiss_factory": self.settings["faiss_factory"],
            "top_k": self.top_k,
            "min_score": self.min_score,
            "device": str(self.settings["device"]),
            "n_entities_indexed": self.n_entities_indexed,
            "n_rows_without_key": self.n_rows_without_key,
            "n_unique_keys": self.n_keys,
            "n_postings": int(len(self.postings)),
            "index_memory": human_bytes(self.memory_bytes()),
        }

    # -- build --------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunks: Iterable[pd.DataFrame] | Callable[[], Iterator[pd.DataFrame]],
        source: str,
        prefix: str,
        key_field: str = NAME_NORM,
        log: Optional[logging.Logger] = None,
        total_rows: Optional[int] = None,
        **settings: Any,
    ) -> "DenseIndex":
        """Embed every non-empty target key once and index it.

        One pass. Embeddings are kept as float16 (half the RAM, negligible effect on
        cosine ranking) and converted to float32 only in FAISS-sized batches. An
        empty key is skipped exactly as the lexical blockers skip it: it carries no
        signal, and embedding "" would put every nameless record next to every other.
        """
        log = log or logger
        factory = _chunk_factory(chunks)
        encoder = get_encoder(settings)
        label = f"[{source}] dense"

        parts_embeddings: list[np.ndarray] = []
        parts_codes: list[np.ndarray] = []
        rows = 0
        skipped = 0
        started = time.time()
        for frame in factory():
            texts = frame[key_field].to_numpy(dtype=object)
            keep = np.fromiter((isinstance(t, str) and len(t) > 0 for t in texts), dtype=bool, count=len(texts))
            rows += len(texts)
            skipped += int((~keep).sum())
            if not keep.any():
                continue
            codes = encode_entity_ids(frame["entity_id"].to_numpy(dtype=object)[keep])
            embeddings = encoder.encode(texts[keep])
            parts_embeddings.append(embeddings.astype(np.float16))
            parts_codes.append(codes)
            elapsed = max(time.time() - started, 1e-9)
            log.info(
                "  %s: %s/%s rows embedded (%.0f rows/s)",
                label,
                f"{rows:,}",
                f"{total_rows:,}" if total_rows else "?",
                rows / elapsed,
            )

        if parts_embeddings:
            embeddings = np.concatenate(parts_embeddings)
            postings = np.concatenate(parts_codes).astype(np.int64)
        else:
            dim = getattr(encoder, "dim", HASHING_DIM)
            embeddings = np.empty((0, dim), dtype=np.float16)
            postings = _EMPTY_INT64
            log.warning("%s: no non-empty %s values to index", label, key_field)
        del parts_embeddings, parts_codes

        faiss_index = _build_faiss_index(embeddings, settings, log, label)
        log.info(
            "%s: %s vectors (dim %s, %s), %s rows skipped for an empty key, %.1f s",
            label,
            f"{faiss_index.ntotal:,}",
            embeddings.shape[1],
            settings["faiss_factory"],
            f"{skipped:,}",
            time.time() - started,
        )
        return cls(
            source=source,
            prefix=prefix,
            key_field=key_field,
            postings=postings,
            faiss_index=faiss_index,
            settings=settings,
            dim=embeddings.shape[1],
            n_entities_indexed=len(postings),
            n_rows_without_key=skipped,
            embeddings=embeddings,
        )

    # -- query --------------------------------------------------------------
    def query(self, values: Sequence[str] | pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Dense candidates for one S1 chunk, with each pair's cosine.

        Every S1 keeps its ``top_k`` nearest targets with cosine >= ``min_score``;
        the output is sorted by packed pair, like every other blocker's.
        """
        texts = _as_object_array(values)
        if texts.size == 0 or self.n_keys == 0:
            return _EMPTY_INT64, {}
        non_empty = np.flatnonzero(
            np.fromiter((isinstance(t, str) and len(t) > 0 for t in texts), dtype=bool, count=len(texts))
        )
        if non_empty.size == 0:
            return _EMPTY_INT64, {}

        queries = _encode_queries(self.settings, texts[non_empty])
        k = min(self.top_k, self.n_keys)
        packed_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        for start in range(0, len(non_empty), DENSE_SEARCH_BATCH):
            stop = min(start + DENSE_SEARCH_BATCH, len(non_empty))
            scores, rows = self.faiss_index.search(
                np.ascontiguousarray(queries[start:stop], dtype=np.float32), k
            )
            keep = (rows >= 0) & (scores >= self.min_score)
            if not keep.any():
                continue
            owners = np.repeat(non_empty[start:stop], k).reshape(stop - start, k)[keep]
            packed_parts.append(pack_pairs(owners, self.postings[rows[keep]]))
            score_parts.append(np.clip(scores[keep].astype(np.float64), -1.0, 1.0))

        if not packed_parts:
            return _EMPTY_INT64, {"dense_cosine": np.empty(0, dtype=np.float64)}
        packed = np.concatenate(packed_parts)
        cosine = np.concatenate(score_parts)
        # Target rows are distinct per query and entity ids are unique per source, so
        # pairs are already unique; np.unique only sorts, and keeps evidence aligned.
        unique_packed, first = np.unique(packed, return_index=True)
        return unique_packed, {"dense_cosine": cosine[first]}

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> Path:
        faiss = _import_faiss()
        directory = ensure_dir(directory)
        np.save(directory / POSTINGS_FILE, self.postings)
        if self.embeddings is not None:
            # Kept for downstream use: a dense_cosine feature on *every* candidate
            # pair (not only the dense-proposed ones) is a gather over these.
            np.save(directory / DENSE_EMBEDDINGS_FILE, self.embeddings)
        index = self.faiss_index
        if self._gpu_resources is not None:  # pragma: no cover - needs a GPU build
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, str(directory / DENSE_FAISS_FILE))
        write_json(
            directory / META_FILE,
            {
                "index_version": INDEX_VERSION,
                "blocker": BLOCKER_DENSE,
                "source": self.source,
                "prefix": self.prefix,
                "key_field": self.key_field,
                "dim": self.dim,
                "settings": _jsonable_settings(self.settings),
                "n_entities_indexed": self.n_entities_indexed,
                "n_rows_without_key": self.n_rows_without_key,
                "n_unique_keys": self.n_keys,
                "n_postings": int(len(self.postings)),
                "files": {
                    "postings": POSTINGS_FILE,
                    "faiss": DENSE_FAISS_FILE,
                    "embeddings": DENSE_EMBEDDINGS_FILE if self.embeddings is not None else None,
                },
            },
        )
        self.directory = Path(directory)
        return Path(directory)

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike,
        log: Optional[logging.Logger] = None,
        settings: Optional[dict] = None,
    ) -> "DenseIndex":
        """Load a persisted dense index.

        ``settings`` (the config's resolved cell) supplies the query-time knobs; the
        build-identity settings always come from the index, since they describe the
        vectors actually stored. A mismatch between the two is caught by
        :func:`index_staleness` before any query runs.
        """
        faiss = _import_faiss()
        directory = Path(directory)
        meta = _read_index_meta(directory, BLOCKER_DENSE)
        stored = dict(meta.get("settings") or {})
        merged = dict(stored)
        if settings:
            for name, value in settings.items():
                if name not in BUILD_IDENTITY_SETTINGS[BLOCKER_DENSE]:
                    merged[name] = value
        index = cls(
            source=meta["source"],
            prefix=meta["prefix"],
            key_field=meta["key_field"],
            postings=np.load(directory / POSTINGS_FILE),
            faiss_index=faiss.read_index(str(directory / DENSE_FAISS_FILE)),
            settings=merged,
            dim=int(meta["dim"]),
            n_entities_indexed=int(meta.get("n_entities_indexed", 0)),
            n_rows_without_key=int(meta.get("n_rows_without_key", 0)),
            directory=directory,
        )
        index._maybe_to_gpu()
        if log:
            log.info("loaded index %s: %s", directory.name, index.describe())
        return index

    # -- internals ----------------------------------------------------------
    def _apply_search_parameters(self) -> None:
        if "IVF" in str(self.settings.get("faiss_factory", "")):
            faiss = _import_faiss()
            faiss.ParameterSpace().set_index_parameter(
                self.faiss_index, "nprobe", int(self.settings.get("nprobe", 32))
            )

    def _maybe_to_gpu(self) -> None:
        """Move the index to the configured GPU when this FAISS build supports it."""
        device = str(self.settings.get("device", "cpu"))
        faiss = _import_faiss()
        if not device.startswith("cuda") or not hasattr(faiss, "StandardGpuResources"):
            return
        gpu = int(device.split(":", 1)[1]) if ":" in device else 0  # pragma: no cover - GPU only
        resources = faiss.StandardGpuResources()  # pragma: no cover
        options = faiss.GpuClonerOptions()  # pragma: no cover
        options.useFloat16 = True  # pragma: no cover
        self.faiss_index = faiss.index_cpu_to_gpu(resources, gpu, self.faiss_index, options)  # pragma: no cover
        self._gpu_resources = resources  # pragma: no cover


def _build_faiss_index(embeddings: np.ndarray, settings: dict, log: logging.Logger, label: str):
    """Create, train (if the factory needs it) and fill a FAISS inner-product index."""
    faiss = _import_faiss()
    dim = int(embeddings.shape[1])
    index = faiss.index_factory(dim, str(settings["faiss_factory"]), faiss.METRIC_INNER_PRODUCT)
    n = len(embeddings)
    if not index.is_trained and n:
        take = min(n, int(settings["train_sample"]))
        rows = np.sort(np.random.default_rng(42).choice(n, take, replace=False))
        log.info("  %s: training %s on %s vectors", label, settings["faiss_factory"], f"{take:,}")
        index.train(np.ascontiguousarray(embeddings[rows], dtype=np.float32))
    for start in range(0, n, DENSE_ADD_BATCH):
        index.add(np.ascontiguousarray(embeddings[start : start + DENSE_ADD_BATCH], dtype=np.float32))
    return index


def _jsonable_settings(settings: dict) -> dict:
    return {
        name: (value if isinstance(value, (int, float, str, bool)) or value is None else str(value))
        for name, value in settings.items()
    }


def _build_exact_name(
    chunks: Iterable[pd.DataFrame] | Callable[[], Iterator[pd.DataFrame]],
    source: str,
    prefix: str,
    key_field: str,
    log: Optional[logging.Logger],
    total_rows: Optional[int],
    **_settings: Any,
) -> ExactNameIndex:
    """Adapter giving the exact-name build the uniform builder signature.

    ``ExactNameIndex.build`` takes no cell settings and needs one pass, so it keeps
    its own signature; the registry entry is this thin wrapper rather than adding
    unused parameters to a class the calibration scripts also use. ``**_settings``
    absorbs the (empty) resolved cell so every registry entry is called the same way.
    """
    return ExactNameIndex.build(
        _chunk_factory(chunks)(),
        source=source,
        prefix=prefix,
        key_field=key_field,
        log=log,
        total_rows=total_rows,
    )


# Every entry takes ``(chunks, source, prefix, key_field, log, total_rows, settings)``,
# where ``chunks`` is a zero-argument factory so a multi-pass builder can read the
# table twice. The settings dict is the blocker's resolved cell
# (:func:`resolve_blocker_settings`), empty for the blockers that take none.
INDEX_BUILDERS = {
    BLOCKER_EXACT_NAME: _build_exact_name,
    BLOCKER_TOKEN: TokenIndex.build,
    BLOCKER_CHAR_NGRAM: CharNgramIndex.build,
    BLOCKER_DENSE: DenseIndex.build,
}

# Loader per blocker. Kept explicit rather than derived from INDEX_BUILDERS because
# the char index takes a runtime worker count and the dense index its query-time
# settings, which the others do not.
INDEX_LOADERS: dict[str, Callable[..., Any]] = {
    BLOCKER_EXACT_NAME: ExactNameIndex.load,
    BLOCKER_TOKEN: TokenIndex.load,
    BLOCKER_CHAR_NGRAM: CharNgramIndex.load,
    BLOCKER_DENSE: DenseIndex.load,
}


# ---------------------------------------------------------------------------
# Build records: detecting stale indexes instead of silently reusing them
# ---------------------------------------------------------------------------
def _prepared_signature(config: dict, split: str, source: str) -> Optional[dict]:
    """Size + mtime of the prepared table an index is built from."""
    path = prepared_path(config, split, source)
    if not path.is_file():
        return None
    stat = path.stat()
    return {"file": path.name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def expected_build_record(
    config: dict, split: str, source: str, blocker: str, limit: Optional[int] = None
) -> dict:
    """What an index built *now*, from this config, would record about itself."""
    settings = resolve_blocker_settings(config, blocker)
    return {
        "limit": int(limit) if limit else None,
        "key_field": _key_field_for(config, blocker),
        "settings": _jsonable_settings(identity_settings(blocker, settings)),
        "prepared": _prepared_signature(config, split, source),
    }


def index_staleness(directory: Path, expected: dict, check_limit: bool = True) -> tuple[list[str], dict]:
    """Differences between a persisted index and ``expected``; empty means reusable.

    Returns ``(problems, meta)``. An index without a ``build`` record predates this
    check: its settings are compared from the fields it did record, and the caller is
    told (``meta['build']`` is absent) that its row limit cannot be verified.
    """
    meta = read_json(directory / META_FILE)
    problems: list[str] = []
    if meta.get("index_version") != INDEX_VERSION:
        problems.append(f"index_version {meta.get('index_version')} != {INDEX_VERSION}")
    record = meta.get("build")
    if meta.get("key_field") != expected["key_field"]:
        problems.append(f"key_field {meta.get('key_field')!r} != config {expected['key_field']!r}")
    recorded_settings = (record or {}).get("settings") or meta.get("settings") or meta
    for name, value in expected["settings"].items():
        if name in recorded_settings and recorded_settings[name] != value:
            problems.append(f"{name} {recorded_settings[name]!r} != config {value!r}")
    if record is not None:
        if check_limit and record.get("limit") != expected["limit"]:
            problems.append(
                f"built with limit={record.get('limit')} but limit={expected['limit']} was requested"
            )
        built_from, now = record.get("prepared"), expected["prepared"]
        if built_from and now and built_from != now:
            problems.append(
                f"prepared table {now['file']} changed since the index was built "
                f"(size/mtime {built_from['size']}/{built_from['mtime_ns']} -> {now['size']}/{now['mtime_ns']})"
            )
    return problems, meta


def build_index(
    config: dict,
    split: str,
    source: str,
    blocker: str = BLOCKER_EXACT_NAME,
    limit: Optional[int] = None,
    log: Optional[logging.Logger] = None,
    overwrite: bool = False,
    total_rows: Optional[int] = None,
) -> MultiKeyIndex:
    """Build (and persist) one blocker index for one target source.

    Args:
        config: loaded config.
        split: ``train`` or ``test``.
        source: ``source2`` or ``source3``.
        blocker: blocker name from :data:`KNOWN_BLOCKERS`.
        limit: read at most this many rows (for smoke tests on a laptop).
        log: logger.
        overwrite: rebuild even if a valid index already exists.

    Returns:
        The built index (also written to ``config['resolved']['index_dir']``).

    Raises:
        ValueError: on an unknown blocker, or a blocker setting the config got wrong.
    """
    if source not in TARGET_SOURCES:
        raise ValueError(f"indexes are for target sources {TARGET_SOURCES}, got {source!r}")
    if blocker not in INDEX_BUILDERS:
        raise ValueError(f"unknown blocker {blocker!r}; expected one of {KNOWN_BLOCKERS}")

    log = log or logger
    settings = resolve_blocker_settings(config, blocker)
    directory = index_dir_for(config, split, source, blocker)
    expected = expected_build_record(config, split, source, blocker, limit)

    if not overwrite:
        meta_path = directory / META_FILE
        if meta_path.is_file():
            # Reuse only an index that is provably the one this call would build: a
            # smoke-test index (--limit) or one built at a different cell, or from a
            # prepared table that has since been rewritten, is rebuilt - reusing it
            # would silently run the full pipeline on the wrong candidates.
            problems, meta = index_staleness(directory, expected)
            if not problems:
                if meta.get("build") is None:
                    log.warning(
                        "index at %s predates build records: its row limit cannot be "
                        "verified. Pass --overwrite if it may be a smoke-test index.",
                        directory,
                    )
                log.info("index already exists at %s - loading instead of rebuilding", directory)
                return _load_index_at(directory, blocker, log=log, settings=settings)
            log.warning("index at %s is stale (%s) - rebuilding", directory, "; ".join(problems))

    key_field = _key_field_for(config, blocker)
    if key_field not in _prepared_columns(config, split, source):
        raise ValueError(
            f"blocker {blocker!r} keys on {key_field!r}, which the prepared {source} table does "
            f"not have; rerun scripts/prepare_data.py"
        )
    path = prepared_path(config, split, source)
    require_file(path, hint="Run: python scripts/prepare_data.py")

    log.info(
        "building %s index | source=%s split=%s key_field=%s settings=%s limit=%s",
        blocker,
        source,
        split,
        key_field,
        settings or "none",
        limit or "none",
    )

    def _chunks() -> Iterator[pd.DataFrame]:
        columns = ["entity_id", key_field]
        rows = 0
        chunksize = config.get("io", {}).get("chunksize", 500_000)
        if limit:
            chunksize = min(chunksize, limit)
        for chunk in iter_prepared(config, split, source, columns=columns, chunksize=chunksize):
            if limit is not None and rows + len(chunk) > limit:
                chunk = chunk.iloc[: limit - rows]
            rows += len(chunk)
            yield chunk
            if limit is not None and rows >= limit:
                break

    builder = INDEX_BUILDERS[blocker]
    index = builder(
        _chunks,
        source=source,
        prefix=SOURCE_PREFIX[source],
        key_field=key_field,
        log=log,
        total_rows=total_rows,
        **settings,
    )
    index.save(directory)
    meta = read_json(directory / META_FILE)
    meta["build"] = expected
    write_json(directory / META_FILE, meta)
    log.info("saved index to %s", directory)
    return index


def _prepared_columns(config: dict, split: str, source: str) -> set[str]:
    """Column names of a prepared table, from its header only.

    The header costs one line; a full read of a 500MB table to validate a column name
    would not be worth it.
    """
    from .data_loader import iter_tsv

    try:
        for chunk in iter_tsv(prepared_path(config, split, source), chunksize=1):
            return set(chunk.columns)
    except FileNotFoundError:
        return set()
    return set()


def _load_index_at(
    directory: Path,
    blocker: str,
    log: Optional[logging.Logger] = None,
    workers: int = 1,
    settings: Optional[dict] = None,
):
    """Load a persisted index of any implemented blocker from its directory."""
    loader = INDEX_LOADERS.get(blocker)
    if loader is None:
        raise NotImplementedError(
            f"blocker {blocker!r} has no index implementation; implemented: {UNION_BLOCKERS}"
        )
    if blocker == BLOCKER_CHAR_NGRAM:
        return CharNgramIndex.load(directory, log=log, workers=workers)
    if blocker == BLOCKER_DENSE:
        return DenseIndex.load(directory, log=log, settings=settings)
    return loader(directory, log=log)


def load_index(
    config: dict,
    split: str,
    source: str,
    blocker: str = BLOCKER_EXACT_NAME,
    log: Optional[logging.Logger] = None,
    workers: int = 1,
    verify: bool = False,
):
    """Load a persisted index, with a clear error if it is missing.

    Args:
        workers: verification processes for the char blocker. Ignored by the others,
            which have no per-pair work to shard. Purely a performance knob: results
            do not depend on it.
        verify: refuse an index that no longer matches the config (different cell,
            key field or prepared table). A row-limited (smoke-test) index is allowed
            but logged loudly, since a smoke run legitimately uses one.

    Raises:
        ValueError: ``verify`` and the index is stale.
    """
    directory = index_dir_for(config, split, source, blocker)
    settings = resolve_blocker_settings(config, blocker) if blocker == BLOCKER_DENSE else None
    if verify and (directory / META_FILE).is_file():
        expected = expected_build_record(config, split, source, blocker)
        problems, meta = index_staleness(directory, expected, check_limit=False)
        if problems:
            raise ValueError(
                f"index {directory} does not match the config: {'; '.join(problems)}\n"
                f"  Rebuild it: python scripts/build_indexes.py --split {split} "
                f"--sources {source} --blockers {blocker}"
            )
        built_limit = (meta.get("build") or {}).get("limit")
        if built_limit:
            (log or logger).warning(
                "index %s was built from only the first %s rows (a smoke-test index)",
                directory.name,
                f"{built_limit:,}",
            )
    return _load_index_at(directory, blocker, log=log, workers=workers, settings=settings)


# ---------------------------------------------------------------------------
# Union of blockers
# ---------------------------------------------------------------------------
def pack_pairs(s1_positions: np.ndarray, entity_codes: np.ndarray) -> np.ndarray:
    """Pack ``(s1_position, entity_code)`` pairs into single int64s.

    Packing is what makes the union a one-liner: ``np.unique`` on packed ints
    simultaneously deduplicates, unions and sorts by (S1, entity).
    """
    if len(s1_positions) != len(entity_codes):
        raise ValueError("s1_positions and entity_codes must be the same length")
    return s1_positions.astype(np.int64) * PAIR_MULTIPLIER + entity_codes.astype(np.int64)


def unpack_pairs(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_pairs`."""
    s1_positions, entity_codes = np.divmod(packed, PAIR_MULTIPLIER)
    return s1_positions, entity_codes


def _positions_of(unique_packed: np.ndarray, packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each value of ``packed`` sits in the sorted ``unique_packed``.

    ``searchsorted`` + equality: a set membership test without building python sets
    over millions of elements.

    Returns:
        ``(positions, present)`` - both aligned to ``packed``; ``positions`` is
        meaningless where ``present`` is false.
    """
    positions = np.searchsorted(unique_packed, packed)
    np.clip(positions, 0, len(unique_packed) - 1, out=positions)
    present = unique_packed[positions] == packed
    return positions, present


def union_blockers(
    blocker_pairs: dict[str, np.ndarray],
    blocker_evidence: Optional[dict[str, dict[str, np.ndarray]]] = None,
    log: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Union candidate pairs from several blockers, deduplicating per S1.

    Unions, never intersects: a pair proposed by any blocker is a candidate. The union
    is a single ``np.unique`` over packed integers, which deduplicates and sorts by
    (s1_position, entity_code) in the same call.

    Args:
        blocker_pairs: ``{blocker_name: packed_pairs}`` exactly as produced by
            :func:`pack_pairs`, or by any index's ``query``. Every array refers to the
            same S1 positions, and each is already deduplicated within its blocker.
        blocker_evidence: ``{blocker_name: {column: values}}`` with each value array
            aligned to that blocker's ``packed_pairs``. Carried through to the output;
            it never changes which pairs are emitted.
        log: logger.

    Returns:
        ``(s1_positions, entity_codes, blockers_per_pair, evidence)``, sorted by
        (s1_position, entity_code). ``blockers_per_pair`` is an object array of
        comma-joined blocker names in :data:`UNION_BLOCKERS` order, so provenance
        survives the union - which is what lets us audit which blocker earned its keep
        and is what the matcher will use to tell a char-only pair from an exact one.
        ``evidence`` is ``{column: float64 array}`` aligned to the output. A blocker
        that proposed no pair contributes no column at all; a blocker that proposed
        some pairs contributes a column that is NaN wherever it did not propose *this*
        pair, which is the normal case for a union - an exact-name pair has no char
        Jaccard. Callers writing a fixed schema should therefore read a column with
        ``evidence.get(column)`` and render the absence as blank.
    """
    non_empty = {name: arr for name, arr in blocker_pairs.items() if len(arr)}
    if not non_empty:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=object), {}

    all_packed = np.concatenate(list(non_empty.values()))
    unique_packed = np.unique(all_packed)
    s1_positions, entity_codes = unpack_pairs(unique_packed)
    n_pairs = len(unique_packed)

    # Canonical, fixed order rather than dict-insertion order, so the provenance
    # string for a pair does not depend on how the caller built its dict. Names may
    # be bare blocker names or ``source:blocker`` labels; the blocker part decides.
    def _rank(name: str) -> tuple[int, str]:
        blocker = name.rsplit(":", 1)[-1]
        return (UNION_BLOCKERS.index(blocker) if blocker in UNION_BLOCKERS else len(UNION_BLOCKERS), name)

    names = sorted(non_empty, key=_rank)

    evidence_out: dict[str, np.ndarray] = {}
    for name in names:
        for column in (blocker_evidence or {}).get(name, {}) or {}:
            evidence_out.setdefault(column, np.full(n_pairs, np.inf, dtype=np.float64))

    provenance = np.full(n_pairs, "", dtype=object)
    if len(names) == 1:
        provenance[:] = names[0]
    for name in names:
        packed = non_empty[name]
        positions, present = _positions_of(unique_packed, packed)
        if not present.all():
            positions = positions[present]
        if len(names) > 1:
            selected = provenance[positions]
            provenance[positions] = np.where(
                selected == "", name, np.char.add(selected.astype(str), f",{name}")
            )
        for column, values in ((blocker_evidence or {}).get(name, {}) or {}).items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape[0] != present.shape[0]:
                raise ValueError(
                    f"evidence {column!r} for blocker {name!r} has {values.shape[0]} values "
                    f"but {present.shape[0]} pairs"
                )
            target = evidence_out[column]
            if not present.all():
                # Rare, and only possible if a caller handed over a non-unique array:
                # fall back to the buffered path so duplicates cannot be dropped.
                positions, values = _positions_of(unique_packed, packed)[0][present], values[present]
            if positions.size:
                np.minimum.at(target, positions, values)

    for column, target in evidence_out.items():
        target[np.isinf(target)] = np.nan

    if log:
        log.info(
            "union: %s blocker(s) -> %s unique pairs",
            len(non_empty),
            f"{n_pairs:,}",
        )
    return s1_positions, entity_codes, provenance, evidence_out


def truncate_per_group(
    s1_positions_sorted: np.ndarray,
    cap: int,
    log: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Boolean keep-mask limiting each S1 to its first ``cap`` candidates.

    Relies on ``s1_positions_sorted`` being sorted (which the union guarantees).
    Deterministic: "first" is by entity code, so a rerun keeps the same pairs.

    Args:
        s1_positions_sorted: sorted S1 positions.
        cap: maximum candidates per S1. ``<=0`` means no limit.
        log: logger.

    Returns:
        Boolean array aligned to the input.
    """
    if cap is None or cap <= 0 or len(s1_positions_sorted) == 0:
        return np.ones(len(s1_positions_sorted), dtype=bool)

    is_new_group = np.empty(len(s1_positions_sorted), dtype=bool)
    is_new_group[0] = True
    if len(s1_positions_sorted) > 1:
        np.not_equal(s1_positions_sorted[1:], s1_positions_sorted[:-1], out=is_new_group[1:])
    group_starts = np.flatnonzero(is_new_group)
    group_sizes = np.diff(np.append(group_starts, len(s1_positions_sorted)))
    position_in_group = np.arange(len(s1_positions_sorted), dtype=np.int64) - np.repeat(
        group_starts, group_sizes
    )
    keep = position_in_group < cap
    if log:
        dropped = int((~keep).sum())
        if dropped:
            log.warning("cap=%s dropped %s candidate pairs across %s S1 entities", cap, f"{dropped:,}", f"{len(group_starts):,}")
    return keep


def decode_candidates(entity_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn packed entity codes into ``("S2-12345", "S2")`` arrays.

    Works on codes from any mix of sources, because the source is packed into the
    code (see ``utils.encode_entity_id``). Returns both the id strings and the
    source labels, since the candidate file records both.
    """
    codes = np.asarray(entity_codes, dtype=np.int64)
    if len(codes) == 0:
        return np.empty(0, dtype=object), np.empty(0, dtype=object)

    source_codes = codes // ID_NUMERIC_MODULUS
    target_ids = decode_entity_ids(codes)
    source_labels = np.full(len(codes), "", dtype=object)
    known = np.zeros(len(codes), dtype=bool)
    for source_code, prefix in ID_SOURCE_NAMES.items():
        mask = source_codes == source_code
        if mask.any():
            source_labels[mask] = prefix
            known |= mask
    if not known.all():  # pragma: no cover - encode/decode are guarded by the codec
        raise ValueError(
            "packed entity codes contained an unknown source; the index is stale - rebuild it"
        )
    return target_ids, source_labels
