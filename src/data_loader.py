"""Configuration loading, streaming TSV access, and ground-truth handling.

Design rules enforced here:

* **Nothing is hardcoded.** Data locations come from ``configs/config.yaml`` and
  can be overridden with environment variables, so the same code runs on a
  Windows laptop, on the HPC login node, and inside a batch job.
* **Nothing large is materialized by accident.** Sources are read through
  ``iter_*`` chunk generators. Peak RAM scales with ``chunksize`` (500k rows by
  default), not with the 10.3M-row dataset.
* **The ground truth is stored compactly.** 7.6M match ids would cost ~400MB as
  strings; they live as int64 codes (~61MB) plus an S1 lookup dict.

Typical use::

    cfg = load_config()
    for chunk in iter_prepared(cfg, split="train", source="source2"):
        ...
    gt = load_ground_truth(cfg)
    gt.matches("S1-965667")
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .utils import (
    encode_entity_ids,
    human_bytes,
    log_memory,
    split_entity_id,
    stable_hash64,
)

logger = logging.getLogger(__name__)

# Repository root = the parent of this file's package directory. Relative paths
# in config.yaml are resolved against it so scripts work from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

# Logical sources. "source1" is the deduplicated reference set; source2 and
# source3 are the noisy sets we must match against.
SOURCES = ("source1", "source2", "source3")
TARGET_SOURCES = ("source2", "source3")
SOURCE_PREFIX = {"source1": "S1", "source2": "S2", "source3": "S3"}
SPLITS = ("train", "test")

# Environment variables that override config.yaml paths.
ENV_DATA_ROOT = "ER_DATA_ROOT"
ENV_TEST_DATA_ROOT = "ER_TEST_DATA_ROOT"
ENV_WORK_DIR = "ER_WORK_DIR"

# Strict TSV: a double quote is an ordinary character, never a field delimiter.
# Pandas' default (QUOTE_MINIMAL on '"') lets a business name that *starts* with an
# unmatched quote swallow the tabs and newlines after it, merging rows silently.
# The challenge files are plain tab-separated text, so no field can legitimately
# contain a tab or a newline and nothing ever needs quoting.
TSV_QUOTING = csv.QUOTE_NONE
# Recorded in every prepared-table sidecar, so a table written under the old
# quoting rule is detected as stale rather than read under the new one.
TSV_DIALECT = "tab/quote-none"

# Legacy candidate file name, from before candidate files were split-aware. Only
# ever *read* (for train), so existing HPC artifacts stay usable.
LEGACY_CANDIDATES_STEM = "candidate_pairs"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(
    config_path: Optional[str | os.PathLike] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Load config.yaml and add a resolved, absolute-path block.

    Precedence, highest first: ``overrides`` argument, environment variable,
    config.yaml value, built-in default.

    Args:
        config_path: path to the YAML file. Defaults to ``configs/config.yaml``.
        overrides: e.g. ``{"data_root": "/scratch/data/train"}``. Intended for
            script CLIs (``--data-root``) so jobs never need edited configs.

    Returns:
        The config dict, plus ``config["resolved"]`` mapping path keys to
        absolute ``Path`` objects.

    Raises:
        FileNotFoundError: if the config file does not exist.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"config file not found: {path}\n"
            f"Expected {DEFAULT_CONFIG_PATH}. Pass --config to point elsewhere."
        )

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config.setdefault("paths", {})
    paths = config["paths"]
    overrides = overrides or {}

    data_root = _first_defined(
        overrides.get("data_root"),
        os.environ.get(ENV_DATA_ROOT),
        paths.get("data_root"),
        "train",
    )
    test_data_root = _first_defined(
        overrides.get("test_data_root"),
        os.environ.get(ENV_TEST_DATA_ROOT),
        paths.get("test_data_root"),
        "test",
    )
    work_dir = _first_defined(
        overrides.get("work_dir"),
        os.environ.get(ENV_WORK_DIR),
        paths.get("work_dir"),
        "outputs",
    )

    def _abs(value: str, base: Path) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    resolved = {
        "data_root": _abs(data_root, REPO_ROOT),
        "test_data_root": _abs(test_data_root, REPO_ROOT),
        "work_dir": _abs(work_dir, REPO_ROOT),
    }
    # Sub-directories default to living under work_dir.
    for key, default in (
        ("prepared_dir", "prepared"),
        ("index_dir", "indexes"),
        ("candidates_dir", "candidates"),
    ):
        configured = paths.get(key)
        if configured:
            resolved[key] = _abs(configured, REPO_ROOT)
        else:
            resolved[key] = resolved["work_dir"] / default
    resolved["log_dir"] = _abs(paths.get("log_dir", "logs"), REPO_ROOT)

    config["resolved"] = resolved
    config["config_path"] = str(path)
    return config


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def data_root_for_split(config: dict, split: str) -> Path:
    """Directory holding the raw TSVs for ``split``."""
    resolved = config["resolved"]
    return resolved["data_root"] if split == "train" else resolved["test_data_root"]


def raw_path(config: dict, split: str, key: str) -> Path:
    """Absolute path of a raw input file.

    ``key`` is one of ``source1``/``source2``/``source3``/``ground_truth``.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    files = config.get("paths", {}).get("files", {}).get(split, {})
    filename = files.get(key)
    if not filename:
        raise KeyError(f"config has no file entry for split={split!r}, key={key!r}")
    return data_root_for_split(config, split) / filename


def prepared_path(config: dict, split: str, source: str) -> Path:
    """Absolute path of a normalized output produced by ``prepare_data.py``."""
    suffix = ".tsv" if config.get("io", {}).get("prepared_format", "tsv") == "tsv" else ".parquet"
    compression = config.get("io", {}).get("compression")
    if suffix == ".tsv" and compression:
        suffix += f".{compression}"
    return config["resolved"]["prepared_dir"] / f"{split}_{source}_norm{suffix}"


def candidates_path(
    config: dict,
    name: str = "candidate_pairs",
    split: str = "train",
    legacy_fallback: bool = True,
) -> Path:
    """Absolute path for a candidate-pairs file of one split: ``{split}_{name}.tsv``.

    The split is part of the name so a test run can never overwrite the train
    candidates (they used to share ``candidate_pairs.tsv``).

    Args:
        name: file stem.
        split: ``train`` or ``test``.
        legacy_fallback: for **readers** only. When the split-aware train file does
            not exist but a pre-split-aware ``{name}.tsv`` does, return that one, so
            candidate files generated before this change stay readable. Writers pass
            ``False`` so they always write the split-aware name.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    suffix = ".parquet" if config.get("io", {}).get("candidates_format") == "parquet" else ".tsv"
    directory = Path(config["resolved"]["candidates_dir"])
    path = directory / f"{split}_{name}{suffix}"
    if legacy_fallback and split == "train" and not path.is_file():
        legacy = directory / f"{name}{suffix}"
        if legacy.is_file():
            logger.warning(
                "reading legacy candidate file %s (split-aware name %s not found)", legacy, path.name
            )
            return legacy
    return path


def require_file(path: Path, hint: str = "") -> Path:
    """Assert a file exists, with an actionable error message."""
    if not path.is_file():
        message = f"required file not found: {path}"
        if hint:
            message += f"\n  {hint}"
        message += (
            f"\n  If the data lives elsewhere, set {ENV_DATA_ROOT} "
            f"(and {ENV_TEST_DATA_ROOT}) or pass --data-root."
        )
        raise FileNotFoundError(message)
    return path


def check_data_available(config: dict, split: str = "train") -> dict:
    """Return the resolved raw paths for ``split``, raising if any are missing."""
    keys = ("source1", "source2", "source3") + (("ground_truth",) if split == "train" else ())
    out = {}
    for key in keys:
        out[key] = require_file(raw_path(config, split, key))
    return out


def describe_environment(config: dict) -> str:
    """One-line summary of the resolved layout, for log headers."""
    resolved = config["resolved"]
    return (
        f"config={config.get('config_path')} | data_root={resolved['data_root']} | "
        f"work_dir={resolved['work_dir']} | index_dir={resolved['index_dir']}"
    )


# ---------------------------------------------------------------------------
# Streaming readers
# ---------------------------------------------------------------------------
def _compression_for(path: Path) -> Optional[str]:
    """Infer pandas compression from the file suffix (``.gz``/``.bz2``/``.zip``)."""
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-1] in (".gz", ".bz2", ".zip", ".zst"):
        return suffixes[-1][1:]
    return None


def iter_tsv(
    path: str | os.PathLike,
    columns: Optional[Sequence[str]] = None,
    chunksize: int = 500_000,
    **read_csv_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Stream a TSV as chunks of DataFrames.

    Memory: O(chunksize). The file itself is never fully loaded.

    Args:
        path: TSV (optionally compressed).
        columns: subset of columns to keep - reading only what you need is the
            cheapest optimization available on these files.
        chunksize: rows per chunk.
        **read_csv_kwargs: forwarded to :func:`pandas.read_csv`.

    Yields:
        DataFrame chunks. All values are read as strings with
        ``keep_default_na=False`` so that literal ``"NA"`` is not mistaken for a
        missing value - business names legitimately contain such tokens.
    """
    path = Path(path)
    kwargs: dict = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "quoting": TSV_QUOTING,
        "chunksize": chunksize,
        "compression": _compression_for(path),
        "on_bad_lines": "warn",
    }
    if columns is not None:
        kwargs["usecols"] = list(columns)
    kwargs.update(read_csv_kwargs)
    yield from pd.read_csv(path, **kwargs)


def read_tsv(
    path: str | os.PathLike,
    columns: Optional[Sequence[str]] = None,
    nrows: Optional[int] = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a TSV fully. Only use on files that genuinely fit in RAM."""
    path = Path(path)
    kwargs: dict = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
        "quoting": TSV_QUOTING,
        "compression": _compression_for(path),
        "on_bad_lines": "warn",
    }
    if columns is not None:
        kwargs["usecols"] = list(columns)
    if nrows is not None:
        kwargs["nrows"] = nrows
    kwargs.update(read_csv_kwargs)
    return pd.read_csv(path, **kwargs)


def iter_prepared(
    config: dict,
    split: str,
    source: str,
    columns: Optional[Sequence[str]] = None,
    chunksize: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Stream a normalized table written by ``prepare_data.py``."""
    path = prepared_path(config, split, source)
    require_file(path, hint="Run: python scripts/prepare_data.py")
    chunksize = chunksize or config.get("io", {}).get("chunksize", 500_000)
    if path.suffix == ".parquet":  # pragma: no cover - needs pyarrow
        frame = pd.read_parquet(path, columns=list(columns) if columns else None)
        for start in range(0, len(frame), chunksize):
            yield frame.iloc[start : start + chunksize]
        return
    yield from iter_tsv(path, columns=columns, chunksize=chunksize)


def count_rows(path: str | os.PathLike) -> int:
    """Count data rows in a TSV without loading it (excludes the header)."""
    path = Path(path)
    compression = _compression_for(path)
    if compression is None:
        with open(path, "rb") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    total = 0
    for chunk in iter_tsv(path, columns=None, chunksize=1_000_000):
        total += len(chunk)
    return total


def count_data_lines(path: str | os.PathLike) -> int:
    """Non-blank physical lines after the header, by a byte scan (no parsing).

    The independent check on the parser: under strict TSV every non-blank line is
    exactly one record, so ``parsed rows != count_data_lines`` means rows were
    merged, split or dropped (``on_bad_lines`` only *warns*). Blank lines are not
    counted because pandas skips them too.
    """
    import gzip

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rb") as handle:
        header = handle.readline()
        if not header:
            return 0
        for line in handle:
            if line.strip(b"\r\n"):
                count += 1
    return count


class ChunkWriter:
    """Append DataFrame chunks to one TSV (optionally compressed).

    Incremental output is what keeps the candidate stage memory-bounded: pairs
    are written and forgotten instead of accumulated in a list.

    Usage::

        with ChunkWriter(path) as writer:
            for chunk in ...:
                writer.append(frame)
    """

    def __init__(self, path: str | os.PathLike, compression: Optional[str] = None) -> None:
        self.path = Path(path)
        self.compression = compression
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = False

    def append(self, frame: pd.DataFrame) -> int:
        """Write a chunk. Returns the number of rows written."""
        if frame is None or len(frame) == 0:
            return 0
        # QUOTE_NONE: fields are written verbatim, so a literal '"' round-trips as
        # itself. A field containing a tab or newline cannot be represented in strict
        # TSV and makes the csv writer raise - loudly, rather than corrupting a row.
        frame.to_csv(
            self.path,
            sep="\t",
            index=False,
            header=not self._wrote_header,
            mode="w" if not self._wrote_header else "a",
            compression=self.compression,
            encoding="utf-8",
            quoting=TSV_QUOTING,
        )
        self._wrote_header = True
        return len(frame)

    def append_header(self, columns: Sequence[str]) -> None:
        """Write a header-only file (a valid, empty table) if nothing was written yet."""
        if self._wrote_header:
            return
        pd.DataFrame(columns=list(columns)).to_csv(
            self.path,
            sep="\t",
            index=False,
            mode="w",
            compression=self.compression,
            encoding="utf-8",
            quoting=TSV_QUOTING,
        )
        self._wrote_header = True

    def __enter__(self) -> "ChunkWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Split assignment (by S1 entity - never by pair)
# ---------------------------------------------------------------------------
def assign_splits(
    entity_ids: pd.Series | Sequence[str],
    val_fraction: float = 0.2,
    mode: str = "hash",
    seed: int = 42,
) -> np.ndarray:
    """Assign each S1 entity to ``train`` or ``val``.

    Splitting is done at the **entity** level: every candidate pair of a given
    S1 lands in the same split. Splitting at pair level would leak, because a
    model would see the same S1 entity (and often near-identical address text)
    in both splits.

    ``hash`` mode is a pure function of the entity id, so the assignment is
    identical in every script and every run without any shared state - the
    candidate generator and the evaluator cannot drift apart.

    Args:
        entity_ids: S1 ids.
        val_fraction: share of entities held out.
        mode: ``hash`` (deterministic) or ``random`` (seeded shuffle).
        seed: used only by ``random`` mode.

    Returns:
        Array of ``"train"``/``"val"`` labels aligned to ``entity_ids``.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    series = entity_ids if isinstance(entity_ids, pd.Series) else pd.Series(entity_ids)
    if mode == "hash":
        # Map the 64-bit hash onto a fine grid so val_fraction is honoured
        # closely even for small inputs.
        buckets = stable_hash64(series) % np.uint64(1_000_000)
        is_val = buckets < np.uint64(int(val_fraction * 1_000_000))
    elif mode == "random":
        rng = np.random.default_rng(seed)
        is_val = rng.random(len(series)) < val_fraction
    else:
        raise ValueError(f"unknown split mode {mode!r}; expected 'hash' or 'random'")

    labels = np.where(is_val, "val", "train")
    return labels.astype(object)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
class GroundTruth:
    """Compact container for S1 -> matched target ids.

    Storage layout (mirrors a CSR matrix):

    * ``entity_ids``: object array of S1 ids, one per row
    * ``offsets``:    int64[n+1]; matches of entity ``i`` are
      ``codes[offsets[i]:offsets[i+1]]``
    * ``codes``:      int64 match id codes (see ``utils.encode_entity_id``)
    * ``index``:      dict S1 id -> row position

    Memory for the full training ground truth: ~61MB of codes, ~18MB of offsets,
    and ~350MB for the id->row dict. The dict is the floor cost of O(1) lookup by
    S1 id; a sorted-array lookup could remove it but is not worth the complexity
    at this scale.

    Entities with zero matches are present with an empty slice, not omitted -
    that distinction matters because 123,247 S1 entities have no match at all
    and any candidate produced for them is a false positive.
    """

    __slots__ = ("entity_ids", "offsets", "codes", "index", "_n_matches", "_axis_sorted", "_axis_order")

    def __init__(
        self,
        entity_ids: np.ndarray,
        offsets: np.ndarray,
        codes: np.ndarray,
        build_index: bool = True,
    ) -> None:
        self.entity_ids = entity_ids
        self.offsets = offsets
        self.codes = codes
        self._n_matches = int(len(codes))
        self.index = {entity_id: i for i, entity_id in enumerate(entity_ids)} if build_index else {}
        # Lazily built sorted numeric axis for vectorized id -> position lookup.
        self._axis_sorted: Optional[np.ndarray] = None
        self._axis_order: Optional[np.ndarray] = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_tsv(
        cls,
        path: str | os.PathLike,
        id_column: str = "source1_entity_id",
        match_column: str = "matched_entity_ids",
        chunksize: int = 500_000,
        progress_logger: Optional[logging.Logger] = None,
    ) -> "GroundTruth":
        """Stream a ground-truth TSV into a :class:`GroundTruth`.

        An empty ``matched_entity_ids`` field is a legitimate "this entity has no
        match" record, not a parse error.
        """
        path = Path(path)
        s1_parts: list[np.ndarray] = []
        code_parts: list[np.ndarray] = []
        length_parts: list[np.ndarray] = []
        total_rows = 0

        for chunk in iter_tsv(path, columns=[id_column, match_column], chunksize=chunksize):
            total_rows += len(chunk)
            s1_parts.append(chunk[id_column].to_numpy(dtype=object))

            raw = chunk[match_column].to_numpy(dtype=object)
            # Split per row rather than str.split+explode: explode would create a
            # 7.6M-row intermediate frame for no benefit.
            per_row = [token.split(",") if token.strip() else [] for token in raw]
            length_parts.append(np.fromiter((len(r) for r in per_row), dtype=np.int64, count=len(per_row)))

            flat = [token.strip() for row in per_row for token in row]
            if flat:
                code_parts.append(encode_entity_ids(pd.Series(flat, dtype="string")))

            if progress_logger:
                progress_logger.info("ground truth: %s rows read", f"{total_rows:,}")

        entity_ids = np.concatenate(s1_parts) if s1_parts else np.empty(0, dtype=object)
        lengths = np.concatenate(length_parts) if length_parts else np.empty(0, dtype=np.int64)
        codes = np.concatenate(code_parts) if code_parts else np.empty(0, dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])

        if offsets[-1] != len(codes):  # pragma: no cover - internal consistency
            raise ValueError(
                f"ground-truth parse inconsistency: offsets say {offsets[-1]} matches, found {len(codes)}"
            )
        return cls(entity_ids, offsets, codes)

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entity_ids)

    @property
    def n_matches(self) -> int:
        return self._n_matches

    @property
    def n_entities(self) -> int:
        return len(self.entity_ids)

    def matches(self, entity_id: str) -> np.ndarray:
        """Match codes for one S1 entity. Empty array if none or unknown id.

        Returns a **view** into the flat code array - no copy, no allocation.
        """
        position = self.index.get(entity_id)
        if position is None:
            return _EMPTY_CODES
        return self.codes[self.offsets[position] : self.offsets[position + 1]]

    def match_count(self, entity_id: str) -> int:
        """Number of true matches for one S1 entity."""
        position = self.index.get(entity_id)
        if position is None:
            return 0
        return int(self.offsets[position + 1] - self.offsets[position])

    def lengths(self) -> np.ndarray:
        """Per-entity match counts, as an int64 array of length ``n_entities``."""
        return np.diff(self.offsets)

    def _numeric_axis(self) -> tuple[np.ndarray, np.ndarray]:
        """Sorted numeric ids and the permutation that produced them (cached)."""
        if self._axis_sorted is None:
            numeric = np.array([split_entity_id(entity_id)[1] for entity_id in self.entity_ids], dtype=np.int64)
            order = np.argsort(numeric, kind="stable")
            self._axis_sorted = numeric[order]
            self._axis_order = order
        return self._axis_sorted, self._axis_order

    def positions_of(self, entity_ids: pd.Series | Sequence[str]) -> np.ndarray:
        """Map S1 ids to ground-truth row positions; ``-1`` for unknown ids.

        Vectorized with ``searchsorted`` over the numeric id axis rather than a
        pandas ``.map`` over the 2.2M-entry dict: the evaluator calls this on
        every candidate row (tens of millions), where a per-row dict probe
        dominates runtime.

        Safe because entity ids are unique within a source and (verified against
        the dataset) carry no leading zeros, so the numeric part identifies the
        row exactly.
        """
        series = entity_ids if isinstance(entity_ids, pd.Series) else pd.Series(entity_ids)
        numeric = series.str.slice(3).astype("int64").to_numpy()
        axis_sorted, axis_order = self._numeric_axis()
        positions = np.searchsorted(axis_sorted, numeric)
        np.clip(positions, 0, len(axis_sorted) - 1, out=positions)
        found = axis_sorted[positions] == numeric
        return np.where(found, axis_order[positions], -1).astype(np.int64)

    def describe(self) -> dict:
        """Summary statistics - cheap enough to log at startup."""
        lengths = self.lengths()
        return {
            "n_entities": int(self.n_entities),
            "n_triples": int(self.n_matches),
            "n_entities_with_matches": int((lengths > 0).sum()),
            "n_entities_without_matches": int((lengths == 0).sum()),
            "mean_matches_per_matched_entity": float(lengths[lengths > 0].mean()) if (lengths > 0).any() else 0.0,
            "max_matches": int(lengths.max()) if len(lengths) else 0,
            "storage": human_bytes(self.codes.nbytes + self.offsets.nbytes + self.entity_ids.nbytes)
            + " + lookup dict",
        }

    def memory_bytes(self) -> int:
        """Approximate footprint, including the S1 lookup dict."""
        # A python dict of 2.2M short-string keys costs roughly 100 bytes/entry.
        return self.codes.nbytes + self.offsets.nbytes + self.entity_ids.nbytes + 100 * len(self.index)


_EMPTY_CODES = np.empty(0, dtype=np.int64)


def load_ground_truth(
    config: dict,
    path: Optional[str | os.PathLike] = None,
    log: Optional[logging.Logger] = None,
) -> GroundTruth:
    """Load the training ground truth using config column names."""
    path = Path(path) if path else raw_path(config, "train", "ground_truth")
    require_file(path, hint="Ground truth is required for training and evaluation.")
    columns = config.get("columns", {})
    log = log or logger
    start_size = None
    if log:
        log.info("loading ground truth from %s", path)
        start_size = log_memory(log, "before ground truth")
    ground_truth = GroundTruth.from_tsv(
        path,
        id_column=columns.get("gt_source1_id", "source1_entity_id"),
        match_column=columns.get("gt_matched_ids", "matched_entity_ids"),
        chunksize=config.get("io", {}).get("chunksize", 500_000),
    )
    if log:
        log.info("ground truth loaded: %s", ground_truth.describe())
        log_memory(log, "after ground truth")
    return ground_truth
