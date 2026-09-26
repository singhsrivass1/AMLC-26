#!/usr/bin/env python
"""Stage 2: build the blocking indexes from the normalized tables.

For the exact-name blocker this is one inverted index per target source:
``normalized_name -> entity ids``. Blocking looks up ~2.2M S1 keys instead of
scanning 10.3M target records, which is what turns the 22.8-trillion-pair
cross product into a lookup.

Memory is estimated and logged **before** the build starts, because the build
stage is the one that holds accumulators for a whole source (unlike the
streaming stages).

    python scripts/build_indexes.py                                    # every blocker enabled in config
    python scripts/build_indexes.py --split test                       # the test-split indexes
    python scripts/build_indexes.py --blockers dense                   # one blocker only
    python scripts/build_indexes.py --sources source2 --limit 500000   # smoke test
    python scripts/build_indexes.py --overwrite                        # force rebuild

By default the blockers built are the ones enabled in ``config.yaml`` - the same
set ``generate_candidates.py`` loads - so the documented full run cannot stop at
stage 3 on a missing index.

An existing index is reused only when its build record (row limit, blocker cell,
key field, prepared-table size/mtime) matches this run. A smoke-test index built
with ``--limit`` is therefore rebuilt by a later full run instead of silently
standing in for the full corpus.

Outputs: ``outputs/indexes/{split}_{source}_{blocker}/`` with flat .npy arrays
(or, for ``dense``, the FAISS index and the float16 embeddings), and meta.json
(format version, counts, key field, build record).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (  # noqa: E402
    KNOWN_BLOCKERS,
    build_index,
    enabled_blockers,
    index_dir_for,
)
from src.data_loader import (  # noqa: E402
    SOURCE_PREFIX,
    TARGET_SOURCES,
    describe_environment,
    load_config,
    prepared_path,
)
from src.utils import (  # noqa: E402
    fmt_int,
    human_bytes,
    log_memory,
    read_json,
    set_seed,
    setup_logging,
    write_json,
)

LOG_NAME = "build_indexes"

# Rough per-row cost of the build accumulators, measured on this dataset:
#   uint64 key hash + int64 entity id + int32 key-offset  = 20 bytes
#   plus the UTF-8 key itself, ~25 bytes for a normalized business name
BYTES_PER_ROW_OVERHEAD = 20
AVG_KEY_BYTES = 25
# After grouping, only unique keys keep their bytes (S2 has ~4.0M unique out of
# 5.0M rows), so the steady-state index is smaller than the build peak.
UNIQUE_KEY_RATIO = 0.8


def estimate_build_memory(n_rows: int, log: logging.Logger) -> dict:
    """Log an a-priori memory estimate for an index build.

    Returns the estimate dict so it can be recorded in the summary. The point is
    that a user on a small machine sees the number before committing to the job.
    """
    peak = n_rows * (BYTES_PER_ROW_OVERHEAD + AVG_KEY_BYTES)
    steady = n_rows * BYTES_PER_ROW_OVERHEAD + int(n_rows * UNIQUE_KEY_RATIO) * AVG_KEY_BYTES
    estimate = {
        "rows": int(n_rows),
        "estimated_peak_build_bytes": int(peak),
        "estimated_steady_state_bytes": int(steady),
    }
    log.info(
        "  memory estimate: peak build ~%s, steady state ~%s (%s rows)",
        human_bytes(peak),
        human_bytes(steady),
        fmt_int(n_rows),
    )
    return estimate


def rows_for_source(config: dict, split: str, source: str, log: logging.Logger) -> int:
    """Row count for a prepared source, from the manifest when available.

    Falls back to a full scan only when the manifest is missing, so the common
    path does not pay for an extra pass over a 500MB file.
    """
    manifest_path = Path(config["resolved"]["prepared_dir"]) / "prepare_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            for entry in manifest.get("sources", []):
                if entry.get("split") == split and entry.get("source") == source:
                    return int(entry["rows"])
        except Exception:  # pragma: no cover - corrupt manifest should not be fatal
            log.warning("could not read %s; falling back to counting rows", manifest_path)

    from src.data_loader import count_rows

    path = prepared_path(config, split, source)
    if not path.is_file():
        return 0
    log.info("  counting rows in %s (manifest unavailable)", path.name)
    return count_rows(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build blocking indexes (stage 2).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--sources", default=",".join(TARGET_SOURCES))
    parser.add_argument(
        "--blockers",
        default=None,
        help=f"comma-separated from {KNOWN_BLOCKERS}. Default: the blockers enabled in "
        "config - the same set generate_candidates.py will load",
    )
    parser.add_argument("--limit", type=int, default=None, help="max rows per source (smoke tests)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild even if a valid index exists")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir}
    )
    log = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    set_seed(config.get("project", {}).get("seed", 42))

    log.info("=" * 78)
    log.info("build_indexes: inverted indexes for blocking")
    log.info(describe_environment(config))
    log.info("=" * 78)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    try:
        blockers = enabled_blockers(config, args.blockers)
    except ValueError as error:
        log.error("%s", error)
        return 2
    log.info("blockers: %s%s", ", ".join(blockers), "" if args.blockers else " (enabled in config)")
    for source in sources:
        if source not in TARGET_SOURCES:
            log.error("indexes are built for %s, got %r", TARGET_SOURCES, source)
            return 2

    for source in sources:
        path = prepared_path(config, args.split, source)
        if not path.is_file():
            log.error("prepared file missing: %s", path)
            log.error("Run first: python scripts/prepare_data.py --splits %s", args.split)
            return 2

    started = time.time()
    summary = []
    for blocker in blockers:
        for source in sources:
            log.info("-" * 78)
            log.info("source=%s blocker=%s", source, blocker)

            n_rows = rows_for_source(config, args.split, source, log)
            if args.limit:
                n_rows = min(n_rows, args.limit)
            estimate = estimate_build_memory(n_rows, log) if n_rows else None

            try:
                index = build_index(
                    config,
                    split=args.split,
                    source=source,
                    blocker=blocker,
                    limit=args.limit,
                    log=log,
                    overwrite=args.overwrite,
                    total_rows=n_rows or None,
                )
            except NotImplementedError as error:
                log.warning("%s", error)
                continue
            except Exception:
                log.exception("failed building %s index for %s", blocker, source)
                return 1

            describe = index.describe()
            log.info("  %s", describe)
            log_memory(log, f"after {source} index")
            summary.append(
                {
                    "split": args.split,
                    "source": source,
                    "prefix": SOURCE_PREFIX[source],
                    "blocker": blocker,
                    "directory": str(index_dir_for(config, args.split, source, blocker)),
                    "limit_applied": args.limit,
                    "estimate": estimate,
                    **describe,
                }
            )

    summary_path = Path(config["resolved"]["index_dir"]) / "index_summary.json"
    write_json(
        summary_path,
        {
            "generated_by": "scripts/build_indexes.py",
            "config": config.get("config_path"),
            "indexes": summary,
            "total_seconds": round(time.time() - started, 1),
        },
    )

    log.info("=" * 78)
    for entry in summary:
        log.info(
            "%-8s %-12s %12s rows -> %12s keys | %s",
            entry["source"],
            entry["blocker"],
            fmt_int(entry["n_entities_indexed"]),
            fmt_int(entry["n_unique_keys"]),
            entry["index_memory"],
        )
    log.info("summary: %s", summary_path)
    log.info("total %.1f min", (time.time() - started) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
