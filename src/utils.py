"""Shared utilities: logging, progress reporting, hashing, memory helpers.

Kept deliberately small and dependency-light so the HPC environment only needs
numpy + pandas + pyyaml (+ optional tqdm/psutil) to run the preprocessing,
indexing and candidate-generation stages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Optional dependencies. The pipeline must run without them.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _tqdm = None
    _HAS_TQDM = False

try:  # pragma: no cover
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_OWNED_HANDLER_FLAG = "_er_setup_logging"


def setup_logging(
    name: str,
    log_dir: Optional[str | os.PathLike] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure a logger that writes to stderr and (optionally) a log file.

    Idempotent: calling it twice with the same name does not duplicate handlers,
    which matters because scripts call it at import time and again in ``main``.

    Args:
        name: logger name, conventionally the module or script name.
        log_dir: directory for the log file. Created if missing.
        level: logging level.
        log_file: explicit file name; defaults to ``<name>.log`` in ``log_dir``.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Idempotent on OUR handlers only. A handler someone else attached (pytest's
    # LogCaptureHandler, an embedding application's) must not count as "already
    # configured", or the run silently writes no log file.
    if any(getattr(handler, _OWNED_HANDLER_FLAG, False) for handler in logger.handlers):
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    setattr(stream_handler, _OWNED_HANDLER_FLAG, True)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (log_file or f"{name}.log")
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _OWNED_HANDLER_FLAG, True)
        logger.addHandler(file_handler)

    return logger


def track(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "",
    logger: Optional[logging.Logger] = None,
    log_every: Optional[int] = None,
    unit: str = "rows",
) -> Iterator[T]:
    """Iterate with progress reporting that degrades gracefully.

    Uses tqdm when it is installed and stderr is a terminal (a progress bar is
    noise in an HPC batch log). Otherwise, logs every ``log_every`` items, or
    every 10% when ``log_every`` is not given.

    Args:
        iterable: the iterable to wrap.
        total: expected number of items, if known.
        desc: label shown in the progress bar / log line.
        logger: logger used for the non-tqdm path.
        log_every: item interval between log lines in the non-tqdm path.
        unit: unit name for the progress output.
    """
    if _HAS_TQDM and sys.stderr.isatty():
        yield from _tqdm(iterable, total=total, desc=desc, unit=unit)
        return

    if logger is None:
        logger = logging.getLogger("progress")

    if log_every is None:
        log_every = max(1, (total // 10) if total else 100_000)

    start = time.time()
    count = 0
    for item in iterable:
        yield item
        count += 1
        if count % log_every == 0:
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0.0
            if total:
                pct = 100.0 * count / total
                eta = (total - count) / rate if rate > 0 else float("nan")
                logger.info(
                    "%s: %s/%s (%.1f%%) | %.0f %s/s | ETA %.1f min",
                    desc or "progress",
                    f"{count:,}",
                    f"{total:,}",
                    pct,
                    rate,
                    unit,
                    eta / 60.0,
                )
            else:
                logger.info(
                    "%s: %s %s | %.0f %s/s | %.1f min",
                    desc or "progress",
                    f"{count:,}",
                    unit,
                    rate,
                    unit,
                    elapsed / 60.0,
                )
    logger.info("%s: done, %s %s in %.1f min", desc or "progress", f"{count:,}", unit, (time.time() - start) / 60.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python/random/numpy. Deliberately does not touch torch.

    Importing torch here would make every CPU stage pay for it, and torch is not
    a core dependency. Accelerator-aware stages seed torch themselves via
    ``seed_torch`` when it is actually installed.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device string. CPU-first, GPU when one is actually available.

    This project is **CPU-first with automatic GPU acceleration**, not CPU-only.
    Accelerator-beneficial stages (embedding generation, dense retrieval, batched
    embedding similarity, transformer / cross-encoder inference) call this and
    transparently get the GPU when one exists, without any stage hardcoding
    ``"cuda"``.

    CPU is the default and the guaranteed fallback: torch is not a core
    dependency, so on a machine without it - or without CUDA - this returns
    ``"cpu"`` and every stage still runs to completion. Stages where CPU was
    found to be genuinely faster (normalization, string similarity, index build,
    GBDT training) do not call this at all; see the README "Compute
    architecture" section for the per-stage split.

    Args:
        preference: ``"auto"`` (detect), or an explicit device such as
            ``"cpu"``/``"cuda"``/``"cuda:1"``/``"mps"``.

    Returns:
        A device string suitable for ``torch.device(...)``.
    """
    if preference and preference != "auto":
        return preference
    try:
        import torch  # lazy: CPU-only runs never pay the import cost
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device_from_config(config: Optional[dict] = None) -> str:
    """Device for an accelerator-capable stage, from ``compute.device`` in config.

    The single place accelerator stages should get a device from, so a site can
    pin ``compute.device: cpu`` (or ``cuda:1``, or ``mps``) in config.yaml -
    including via a site-local override file - without any stage hardcoding a
    device string.

    Args:
        config: loaded config. Missing ``compute`` block means ``"auto"``.

    Returns:
        A device string, as :func:`resolve_device`.
    """
    section = (config or {}).get("compute", {}) or {}
    return resolve_device(str(section.get("device", "auto")))


def seed_torch(seed: int) -> bool:
    """Seed torch if it is installed. Returns whether it was.

    Kept separate from :func:`set_seed` so CPU-only environments never import
    torch, while accelerator stages still get reproducible runs.
    """
    try:
        import torch
    except ImportError:
        return False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return True


def describe_device(logger: Optional[logging.Logger] = None, config: Optional[dict] = None) -> dict:
    """Report the compute environment. Logged at the start of accelerator stages.

    Args:
        logger: optional logger.
        config: optional loaded config; when given, the reported device honours
            ``compute.device`` instead of always auto-detecting.
    """
    device = resolve_device_from_config(config) if config is not None else resolve_device()
    info: dict = {"device": device, "torch": None, "gpu": None, "gpu_memory": None, "cuda_available": False}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = human_bytes(torch.cuda.get_device_properties(0).total_memory)
    except ImportError:
        pass
    if logger:
        logger.info(
            "compute: device=%s torch=%s cuda=%s gpu=%s%s",
            info["device"],
            info["torch"] or "not installed",
            "yes" if info["cuda_available"] else "no",
            info["gpu"] or "none",
            f" ({info['gpu_memory']})" if info["gpu_memory"] else "",
        )
    return info


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
# Query used against nvidia-smi. ``nounits`` keeps the values numeric (MiB), so
# the parse is a plain float() rather than a unit-aware regex.
_NVIDIA_SMI_QUERY = "index,name,memory.total,memory.free"


def detect_hardware() -> dict:
    """Report CPUs, RAM, every GPU, and CUDA/torch status.

    Torch alone is **not** a sufficient GPU detector: a CPU-only torch build (the
    common case on a login node, and the case on a laptop with a usable GPU)
    reports no CUDA even when ``nvidia-smi`` can enumerate devices. So GPUs come
    from ``nvidia-smi`` first, and torch is used only to enrich or to fill in the
    device list when the tool is unavailable. Nothing here is fatal: a machine
    without nvidia-smi, without torch or without psutil still yields a report.

    Returns:
        A dict with ``cpu_logical``, ``cpu_physical``, ``ram_total``,
        ``ram_available``, ``gpus`` (list of ``{index, name, memory_total,
        memory_free}``, byte-valued, ``None`` where unknown), ``gpu_source``,
        ``torch``, ``cuda_available`` and ``cuda_device_count``.
    """
    info: dict = {
        "cpu_logical": os.cpu_count(),
        "cpu_physical": None,
        "ram_total": None,
        "ram_available": None,
        "gpus": [],
        "gpu_source": "nvidia-smi",
        "torch": None,
        "cuda_available": False,
        "cuda_device_count": 0,
    }

    if _psutil is not None:
        try:
            info["cpu_physical"] = _psutil.cpu_count(logical=False)
        except Exception:  # pragma: no cover - psutil is best-effort here
            pass
        try:
            memory = _psutil.virtual_memory()
            info["ram_total"] = int(memory.total)
            info["ram_available"] = int(memory.available)
        except Exception:  # pragma: no cover
            pass

    info["gpus"] = _nvidia_smi_gpus()

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = int(torch.cuda.device_count())
            if not info["gpus"]:
                # No nvidia-smi: fall back to what torch can see.
                info["gpu_source"] = "torch"
                for index in range(info["cuda_device_count"]):
                    try:
                        properties = torch.cuda.get_device_properties(index)
                        info["gpus"].append(
                            {
                                "index": index,
                                "name": torch.cuda.get_device_name(index),
                                "memory_total": int(properties.total_memory),
                                "memory_free": None,
                            }
                        )
                    except Exception:  # pragma: no cover - defensive
                        info["gpus"].append(
                            {"index": index, "name": "unknown", "memory_total": None, "memory_free": None}
                        )
    except ImportError:
        pass

    return info


def _nvidia_smi_gpus() -> list[dict]:
    """Enumerate GPUs via ``nvidia-smi``. Empty list when it is absent or fails."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return parse_nvidia_smi(completed.stdout)


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output.

    Malformed lines are skipped rather than raising, and unreadable memory
    figures (``[N/A]`` on some virtualised drivers) leave the GPU in the list
    with ``None`` memory instead of dropping it - the device still exists and
    still matters for the report.
    """
    gpus: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        gpu = {"index": index, "name": parts[1], "memory_total": None, "memory_free": None}
        if len(parts) >= 4:
            gpu["memory_total"] = _mib_to_bytes(parts[2])
            gpu["memory_free"] = _mib_to_bytes(parts[3])
        gpus.append(gpu)
    return gpus


def _mib_to_bytes(text: str) -> Optional[int]:
    """Convert an nvidia-smi MiB figure to bytes; ``None`` when unreadable."""
    try:
        return int(float(text)) * 1024 * 1024
    except (TypeError, ValueError):
        return None


def format_hardware_report(info: dict) -> str:
    """Render :func:`detect_hardware` as a short multi-line human summary."""
    physical = info.get("cpu_physical")
    logical = info.get("cpu_logical")
    if physical and logical and physical != logical:
        cpu = f"{physical} physical / {logical} logical"
    else:
        cpu = f"{physical or logical or 'unknown'} logical"
    lines = [f"cpu: {cpu}"]

    if info.get("ram_total"):
        available = (
            f", {human_bytes(info['ram_available'])} available" if info.get("ram_available") else ""
        )
        lines.append(f"ram: {human_bytes(info['ram_total'])}{available}")

    gpus = info.get("gpus") or []
    if not gpus:
        lines.append("gpus: none detected")
    for gpu in gpus:
        memory = human_bytes(gpu["memory_total"]) if gpu.get("memory_total") else "vram unknown"
        free = f", {human_bytes(gpu['memory_free'])} free" if gpu.get("memory_free") else ""
        lines.append(f"gpu {gpu['index']}: {gpu['name']} ({memory}{free}) [{info.get('gpu_source', '?')}]")

    torch_version = info.get("torch") or "not installed"
    if info.get("cuda_available"):
        lines.append(
            f"cuda: yes (torch {torch_version}, {info.get('cuda_device_count', 0)} device(s) usable by torch)"
        )
    elif gpus:
        lines.append(f"cuda: no (torch {torch_version}) - GPU(s) present but not usable by torch")
    else:
        lines.append(f"cuda: no (torch {torch_version})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker resolution
# ---------------------------------------------------------------------------
def auto_worker_count() -> int:
    """Default worker count when none is configured: physical cores, else logical."""
    physical = _physical_cpu_count()
    if physical:
        return max(1, int(physical))
    return max(1, os.cpu_count() or 1)


def _physical_cpu_count() -> Optional[int]:
    """Physical core count, or ``None`` when psutil cannot tell us."""
    if _psutil is None:
        return None
    try:
        return _psutil.cpu_count(logical=False)
    except Exception:  # pragma: no cover - defensive
        return None


def resolve_workers(
    requested: int = 0,
    configured: int = 0,
    n_chunks: int = 0,
    logger: Optional[logging.Logger] = None,
    label: str = "workers",
) -> int:
    """Resolve the CPU worker count: CLI > config > auto, clamped to the work available.

    The auto default is the **physical** core count. This stage is bound by python
    string/token work and memory bandwidth rather than by floating-point units, so
    hyperthread siblings mostly add contention; and the previous hard cap of 8
    silently threw away most of a large HPC node.

    Args:
        requested: value from the CLI (``0`` means "not specified").
        configured: value from ``compute.num_workers`` (``0`` means auto).
        n_chunks: number of work chunks; more workers than chunks is wasted.
        logger: optional logger, for the one-line explanation of the choice.
        label: label used in the log line.

    Returns:
        A worker count of at least 1, never above the chunk count.
    """
    total = auto_worker_count()
    workers = int(requested or configured or total)
    workers = max(1, min(workers, max(int(n_chunks), 1)))
    if logger:
        logger.info(
            "%s: %s (logical cpu=%s, physical cpu=%s, chunks=%s)",
            label,
            fmt_int(workers),
            fmt_int(os.cpu_count() or 0),
            fmt_int(_physical_cpu_count() or 0),
            fmt_int(n_chunks),
        )
    return workers


def plan_inflight_window(
    workers: int,
    chunk_pairs: int,
    bytes_per_pair: int,
    budget_bytes: int,
    logger: Optional[logging.Logger] = None,
    label: str = "chunks",
) -> tuple[int, int]:
    """Bound in-flight chunk payloads so a big worker count cannot exhaust RAM.

    A process pool keeps ``workers * 4`` chunks queued to stay fed, and each queued
    chunk holds its pair strings as python objects. At the core counts worth using
    on a large node that transient payload - not the report arrays - is what
    decides peak RSS, so it is sized against a real budget instead of a bare
    multiplier. When the budget does not fit, the payload is reduced and the
    adjustment is logged; the run continues rather than dying.

    Args:
        workers: number of worker processes.
        chunk_pairs: pairs per chunk as requested.
        bytes_per_pair: estimated payload bytes for one pair (all six fields).
        budget_bytes: RAM to spend on queued payloads.
        logger: optional logger, for the adjustment message.
        label: label used in the log line.

    Returns:
        ``(window, chunk_pairs)`` - the in-flight chunk limit and the possibly
        reduced chunk size. Both are safe to use as-is.
    """
    target_window = max(2, int(workers) * 4)
    bytes_per_pair = max(1, int(bytes_per_pair))
    chunk_pairs = max(1, int(chunk_pairs))

    requested_pairs = chunk_pairs
    estimate = chunk_pairs * bytes_per_pair
    if target_window * estimate > budget_bytes:
        # Reduce the payload first: it lowers peak memory and keeps the pool fed
        # with the same number of in-flight slots, which is the cheap fix.
        affordable_pairs = max(1, budget_bytes // (target_window * bytes_per_pair))
        chunk_pairs = min(chunk_pairs, affordable_pairs)
        estimate = chunk_pairs * bytes_per_pair

    affordable_window = max(1, budget_bytes // estimate)
    window = max(2, min(target_window, affordable_window))

    if logger and (chunk_pairs != requested_pairs or window != target_window):
        logger.info(
            "%s: payload bounded by RAM budget - chunk_pairs %s -> %s, in-flight window %s -> %s "
            "(est %s/pair, budget %s)",
            label,
            fmt_int(requested_pairs),
            fmt_int(chunk_pairs),
            fmt_int(target_window),
            fmt_int(window),
            human_bytes(bytes_per_pair),
            human_bytes(budget_bytes),
        )
    return window, chunk_pairs


def stable_hash64(values: str | Sequence[str]) -> np.ndarray | int:
    """Hash strings to 64-bit unsigned ints.

    Deliberately NOT python's builtin ``hash()``: that is salted per process
    (PYTHONHASHSEED), so an index built in one run would not be readable in the
    next. blake2b is stable across processes, machines and python versions.

    Collisions are possible in principle (64-bit space); every consumer here
    verifies the actual string after a hash lookup, so a collision degrades to a
    miss rather than a wrong answer. With ~4M keys the birthday probability is
    ~4e-7.

    Args:
        values: a single string, or an iterable/pandas Series of strings.

    Returns:
        A python int for a single string, else a uint64 numpy array.
    """
    if isinstance(values, str):
        return int.from_bytes(hashlib.blake2b(values.encode("utf-8"), digest_size=8).digest(), "little")

    import pandas as pd  # local import keeps this module importable without pandas

    if isinstance(values, pd.Series):
        raw = values.array
    else:
        raw = np.asarray(values, dtype=object)

    out = np.empty(len(raw), dtype=np.uint64)
    blake = hashlib.blake2b
    for i, value in enumerate(raw):
        out[i] = int.from_bytes(blake(value.encode("utf-8"), digest_size=8).digest(), "little")
    return out


# ---------------------------------------------------------------------------
# Memory / diagnostics
# ---------------------------------------------------------------------------
def human_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def nbytes_of(*arrays: np.ndarray) -> str:
    """Total size of the given arrays, formatted."""
    return human_bytes(sum(a.nbytes for a in arrays if a is not None))


def log_memory(logger: logging.Logger, label: str = "") -> Optional[str]:
    """Log current RSS if psutil is available. Returns the formatted string."""
    rss = current_rss_bytes()
    if rss is None:
        return None
    text = human_bytes(rss)
    logger.info("RSS%s: %s", f" ({label})" if label else "", text)
    return text


def current_rss_bytes() -> Optional[int]:
    """This process's current RSS in bytes, or ``None`` without psutil."""
    if _psutil is None:
        return None
    try:
        return int(_psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # pragma: no cover - defensive
        return None


def peak_rss_bytes() -> Optional[int]:
    """This process's peak RSS in bytes, or ``None`` when it cannot be determined.

    ``resource.getrusage`` reports the high-water mark, which is the number that
    matters for sizing a run rather than the instantaneous value. Returns ``None``
    on platforms where it is unavailable or meaningless (Windows) instead of a
    misleading zero.
    """
    try:
        import resource
    except ImportError:
        return None
    try:
        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover - defensive
        return None
    if not usage:
        return None
    # Linux reports kibibytes; macOS reports bytes.
    return usage if sys.platform == "darwin" else usage * 1024


# ---------------------------------------------------------------------------
# Entity id codec
# ---------------------------------------------------------------------------
# Challenge ids look like "S2-166376419": a source prefix plus a numeric suffix
# of 2-9 digits (verified against the full training set). We pack them into
# int64 as ``source_code * 10**10 + numeric``.
#
# Why bother: the ground truth holds 7.6M matched ids. As a numpy unicode array
# ('<U13') that is ~400MB; as int64 it is ~61MB. Integer codes also let the
# evaluation stage do set comparisons with np.isin instead of python string sets,
# which is several times faster on 7.6M pairs.
ID_SOURCE_CODES = {"S1": 1, "S2": 2, "S3": 3}
ID_SOURCE_NAMES = {code: name for name, code in ID_SOURCE_CODES.items()}
ID_NUMERIC_MODULUS = 10**10
_ID_MAX_NUMERIC = ID_NUMERIC_MODULUS - 1


def split_entity_id(entity_id: str) -> tuple[str, int]:
    """Split ``"S2-166376419"`` into ``("S2", 166376419)``."""
    prefix, _, numeric = entity_id.partition("-")
    if not numeric or not numeric.isdigit():
        raise ValueError(f"malformed entity id: {entity_id!r}")
    return prefix, int(numeric)


def encode_entity_id(entity_id: str) -> int:
    """Pack a single entity id into an int64-safe integer code."""
    prefix, numeric = split_entity_id(entity_id)
    try:
        source_code = ID_SOURCE_CODES[prefix]
    except KeyError:
        raise ValueError(f"unknown source prefix {prefix!r} in {entity_id!r}") from None
    if numeric > _ID_MAX_NUMERIC:  # pragma: no cover - guarded by the codec design
        raise ValueError(f"numeric part too large to pack: {entity_id!r}")
    return source_code * ID_NUMERIC_MODULUS + numeric


def decode_entity_id(code: int) -> str:
    """Inverse of :func:`encode_entity_id`."""
    code = int(code)
    source_code, numeric = divmod(code, ID_NUMERIC_MODULUS)
    try:
        prefix = ID_SOURCE_NAMES[source_code]
    except KeyError:
        raise ValueError(f"unknown source code {source_code} in packed id {code}") from None
    return f"{prefix}-{numeric}"


def decode_entity_ids(codes: np.ndarray) -> np.ndarray:
    """Vectorized :func:`decode_entity_id`. Returns an object-dtype string array."""
    codes = np.asarray(codes, dtype=np.int64)
    source_codes, numerics = np.divmod(codes, ID_NUMERIC_MODULUS)
    if len(codes) and not np.isin(source_codes, list(ID_SOURCE_NAMES)).all():
        raise ValueError("packed ids contain an unknown source code")
    prefixes = np.array([ID_SOURCE_NAMES[int(s)] for s in np.unique(source_codes)], dtype=object)
    lookup = {int(s): ID_SOURCE_NAMES[int(s)] for s in np.unique(source_codes)}
    out = np.empty(len(codes), dtype=object)
    # Group by source so the f-string loop is not re-deciding the prefix each row.
    for source_code, prefix in lookup.items():
        mask = source_codes == source_code
        if mask.any():
            out[mask] = [f"{prefix}-{n}" for n in numerics[mask]]
    del prefixes
    return out


def encode_entity_ids(values) -> np.ndarray:
    """Vectorized :func:`encode_entity_id` over a Series / array / list."""
    import pandas as pd

    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    prefixes = series.str.slice(0, 2)
    numerics = series.str.slice(3)
    if not numerics.str.isdigit().all():
        bad = series[~numerics.str.isdigit()].head(5).tolist()
        raise ValueError(f"malformed entity ids (expected e.g. 'S2-123'), got: {bad}")
    numbers = numerics.astype("int64").to_numpy()
    if len(numbers) and numbers.max() > _ID_MAX_NUMERIC:  # pragma: no cover
        raise ValueError("numeric part too large to pack")
    source_codes = prefixes.map(ID_SOURCE_CODES)
    if source_codes.isna().any():
        bad = series[source_codes.isna()].head(5).tolist()
        raise ValueError(f"unknown source prefix in ids: {bad}")
    return source_codes.to_numpy(dtype=np.int64) * ID_NUMERIC_MODULUS + numbers


# ---------------------------------------------------------------------------
# Tiny IO helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | os.PathLike, payload: Any) -> None:
    """Write JSON atomically (temp file + replace) so partial runs can't corrupt."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, target)


def read_json(path: str | os.PathLike) -> Any:
    """Read a JSON file produced by :func:`write_json`."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def fmt_int(value: int | float) -> str:
    """Thousands-separated integer string for log messages."""
    return f"{int(value):,}"
