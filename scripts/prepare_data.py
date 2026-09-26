#!/usr/bin/env python
"""Stage 1: normalize the raw TSVs.

Reads ``train_source{1,2,3}.tsv`` (and the test split), appends the normalized
columns, tags S1 rows with their train/val split, and writes one normalized TSV
per source. Original columns are preserved, so nothing is lost.

Memory: O(chunksize). The 10.3M target records are never resident at once.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --splits train --limit 200000     # smoke test
    python scripts/prepare_data.py --data-root /scratch/data/train

Outputs (under ``outputs/prepared/``)::

    {split}_{source}_norm.tsv      entity_id, business_name, business_address,
                                   country, name_norm, name_key, address_norm,
                                   country_norm [, split]
    prepare_manifest.json          row counts + settings, for provenance
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Make ``import src.*`` work when the script is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    SOURCE_PREFIX,
    SOURCES,
    SPLITS,
    TSV_DIALECT,
    ChunkWriter,
    assign_splits,
    check_data_available,
    count_data_lines,
    describe_environment,
    load_config,
    prepared_path,
    raw_path,
)
from src.normalization import (  # noqa: E402
    ADDRESS_NORM,
    NAME_NORM,
    Normalizer,
    add_normalized_columns,
)
from src.utils import (  # noqa: E402
    ensure_dir,
    fmt_int,
    log_memory,
    read_json,
    set_seed,
    setup_logging,
    track,
    write_json,
)

LOG_NAME = "prepare_data"


def sidecar_path(output_path: Path) -> Path:
    """Provenance record written next to every prepared table."""
    return output_path.with_name(output_path.name + ".meta.json")


def expected_provenance(config: dict, split: str, source: str, normalizer: Normalizer, limit: int | None) -> dict:
    """Everything that decides a prepared table's content, for this request.

    A prepared table is reused only when its recorded provenance equals this. The
    row ``limit`` is the field that matters most: without it a ``--limit 100000``
    smoke table is indistinguishable from a full one, and the full run that follows
    the README's smoke test would silently use 100k rows.
    """
    input_path = raw_path(config, split, source)
    stat = input_path.stat()
    split_section = config.get("evaluation", {}).get("split", {}) or {}
    return {
        "limit": int(limit) if limit else None,
        "tsv_dialect": TSV_DIALECT,
        "normalization": {
            "unicode_form": normalizer.unicode_form,
            "fold_latin_accents": normalizer.fold_latin_accents,
            "case_mode": normalizer.case_mode,
            "punctuation_to_space": normalizer.punctuation_to_space,
            "max_length": normalizer.max_length,
        },
        "raw": {"file": input_path.name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)},
        # The split column depends on these, so they are part of the S1 table's identity.
        "split": (
            {
                "val_fraction": split_section.get("val_fraction", 0.2),
                "mode": split_section.get("mode", "hash"),
                "seed": config.get("project", {}).get("seed", 42),
            }
            if source == "source1" and split == "train"
            else None
        ),
    }


def reuse_problems(output_path: Path, expected: dict) -> list[str]:
    """Why an existing prepared table cannot be reused; empty means it can."""
    record_path = sidecar_path(output_path)
    if not record_path.is_file():
        return ["no provenance record (written before provenance tracking, or by a crashed run)"]
    try:
        recorded = read_json(record_path).get("request", {})
    except Exception:  # noqa: BLE001 - a corrupt record is simply not trusted
        return ["unreadable provenance record"]
    return [
        f"{key}: recorded {recorded.get(key)!r} != requested {value!r}"
        for key, value in expected.items()
        if recorded.get(key) != value
    ]


def prepare_one(
    config: dict,
    split: str,
    source: str,
    normalizer: Normalizer,
    limit: int | None,
    overwrite: bool,
    log,
    allow_row_loss: bool = False,
) -> dict:
    """Normalize one source of one split. Returns a stats dict."""
    from src.data_loader import iter_tsv  # local import: keeps module import light

    columns = config.get("columns", {})
    id_col = columns.get("entity_id", "entity_id")
    name_col = columns.get("name", "business_name")
    address_col = columns.get("address", "business_address")
    country_col = columns.get("country", "country")

    input_path = raw_path(config, split, source)
    output_path = prepared_path(config, split, source)
    expected = expected_provenance(config, split, source, normalizer, limit)

    if output_path.is_file() and not overwrite:
        problems = reuse_problems(output_path, expected)
        if not problems:
            existing = _count_output_rows(output_path)
            log.info("%s/%s already prepared (%s rows) - skipping (use --overwrite)", split, source, fmt_int(existing))
            return {"split": split, "source": source, "rows": existing, "skipped": True, "output": str(output_path)}
        log.warning(
            "%s/%s: existing %s cannot be reused (%s) - re-preparing",
            split,
            source,
            output_path.name,
            "; ".join(problems),
        )

    chunksize = config.get("io", {}).get("chunksize", 500_000)
    if limit:
        chunksize = min(chunksize, limit)
    compression = config.get("io", {}).get("compression")

    log.info("preparing %s/%s", split, source)
    log.info("  input : %s", input_path)
    log.info("  output: %s", output_path)

    started = time.time()
    rows_in = 0
    rows_out = 0
    blank_names = 0
    blank_addresses = 0
    split_counts = {"train": 0, "val": 0}
    output_columns_written: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # A partially written file must never be mistaken for a complete one, so
    # write to .partial and rename on success.
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    def _chunks():
        nonlocal rows_in
        for chunk in iter_tsv(input_path, chunksize=chunksize):
            if limit is not None and rows_in + len(chunk) > limit:
                chunk = chunk.iloc[: limit - rows_in]
            rows_in += len(chunk)
            yield chunk
            if limit is not None and rows_in >= limit:
                break

    with ChunkWriter(partial_path, compression=compression) as writer:
        for chunk in track(_chunks(), desc=f"{split}/{source}", logger=log):
            frame = add_normalized_columns(chunk, config, normalizer=normalizer, columns=columns)

            # S1 is the entity we split on; targets are never split.
            if source == "source1" and split == "train":
                labels = assign_splits(
                    frame[id_col],
                    val_fraction=config.get("evaluation", {}).get("split", {}).get("val_fraction", 0.2),
                    mode=config.get("evaluation", {}).get("split", {}).get("mode", "hash"),
                    seed=config.get("project", {}).get("seed", 42),
                )
                frame["split"] = labels
                values, counts = np.unique(labels, return_counts=True)
                for value, count in zip(values, counts):
                    split_counts[str(value)] = split_counts.get(str(value), 0) + int(count)

            blank_names += int((frame[NAME_NORM] == "").sum())
            blank_addresses += int((frame[ADDRESS_NORM] == "").sum())

            written = writer.append(frame)
            rows_out += written
            if not output_columns_written:
                output_columns_written = list(frame.columns)

            # Guard: normalization must never add or drop rows.
            if len(frame) != written:
                raise RuntimeError(f"row count changed while writing {split}/{source}: {len(frame)} -> {written}")

    elapsed = time.time() - started

    # Verify the file we just wrote has exactly the rows we intended.
    written_rows = _count_output_rows(partial_path)
    if written_rows != rows_out:
        raise RuntimeError(
            f"integrity check failed for {split}/{source}: wrote {rows_out} rows, file has {written_rows}"
        )

    # Verify the parser read every record of the raw file. Under strict TSV one
    # non-blank line is one record, so a mismatch means rows were merged or dropped
    # (on_bad_lines only warns). A lost S1 row is a missing submission row; a lost
    # target row is lost recall - neither may pass silently.
    raw_lines = None
    if not limit:
        raw_lines = count_data_lines(input_path)
        if raw_lines != rows_in:
            message = (
                f"{split}/{source}: raw file has {fmt_int(raw_lines)} data lines but the parser "
                f"read {fmt_int(rows_in)} rows - rows were merged or dropped (check for "
                f"malformed lines / stray tabs in {input_path.name})"
            )
            if not allow_row_loss:
                partial_path.unlink(missing_ok=True)
                raise RuntimeError(message + "; rerun with --allow-row-loss to accept this")
            log.error("%s (accepted: --allow-row-loss)", message)
    partial_path.replace(output_path)
    write_json(
        sidecar_path(output_path),
        {"request": expected, "rows": int(rows_out), "raw_data_lines": raw_lines},
    )

    stats = {
        "split": split,
        "source": source,
        "prefix": SOURCE_PREFIX[source],
        "rows": rows_out,
        "rows_read": rows_in,
        "blank_name_norm": blank_names,
        "blank_address_norm": blank_addresses,
        "split_counts": split_counts if source == "source1" and split == "train" else None,
        "columns": output_columns_written,
        "output": str(output_path),
        "seconds": round(elapsed, 1),
        "skipped": False,
    }
    log.info(
        "  done: %s rows in %.1f min | blank names=%s blank addresses=%s",
        fmt_int(rows_out),
        elapsed / 60.0,
        fmt_int(blank_names),
        fmt_int(blank_addresses),
    )
    if stats["split_counts"]:
        log.info(
            "  S1 split: train=%s val=%s",
            fmt_int(stats["split_counts"].get("train", 0)),
            fmt_int(stats["split_counts"].get("val", 0)),
        )
    log_memory(log, f"after {split}/{source}")
    return stats


def _count_output_rows(path: Path) -> int:
    """Count data rows in a prepared TSV (handles .gz)."""
    import gzip

    if not path.is_file():
        return 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw challenge TSVs (stage 1).")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--data-root", default=None, help="override paths.data_root (or set ER_DATA_ROOT)")
    parser.add_argument("--test-data-root", default=None, help="override paths.test_data_root")
    parser.add_argument("--work-dir", default=None, help="override paths.work_dir (or set ER_WORK_DIR)")
    parser.add_argument("--splits", default="train,test", help="comma-separated: train,test")
    parser.add_argument("--sources", default=",".join(SOURCES), help="comma-separated source names")
    parser.add_argument("--limit", type=int, default=None, help="max rows per source (smoke tests)")
    parser.add_argument("--chunksize", type=int, default=None, help="override io.chunksize")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-prepare even if a matching output exists (a mismatching one - other "
        "--limit, normalization or raw file - is always re-prepared)",
    )
    parser.add_argument(
        "--allow-row-loss",
        action="store_true",
        help="accept a raw file whose physical line count differs from the parsed row "
        "count (logged as an error instead of failing the run)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config,
        overrides={"data_root": args.data_root, "test_data_root": args.test_data_root, "work_dir": args.work_dir},
    )
    if args.chunksize:
        config.setdefault("io", {})["chunksize"] = args.chunksize

    log = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    set_seed(config.get("project", {}).get("seed", 42))

    log.info("=" * 78)
    log.info("prepare_data: normalize raw sources")
    log.info(describe_environment(config))
    log.info("=" * 78)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for split in splits:
        if split not in SPLITS:
            log.error("unknown split %r; expected one of %s", split, SPLITS)
            return 2

    normalizer = Normalizer.from_config(config)
    log.info(
        "normalization: form=%s fold_latin=%s case=%s punct_to_space=%s max_len=%s",
        normalizer.unicode_form,
        normalizer.fold_latin_accents,
        normalizer.case_mode,
        normalizer.punctuation_to_space,
        normalizer.max_length,
    )

    # Fail before doing any work if inputs are missing.
    for split in splits:
        try:
            check_data_available(config, split)
        except FileNotFoundError as error:
            log.error("%s", error)
            return 2

    ensure_dir(config["resolved"]["prepared_dir"])
    all_stats = []
    started = time.time()
    for split in splits:
        for source in sources:
            if source not in SOURCES:
                log.error("unknown source %r; expected one of %s", source, SOURCES)
                return 2
            try:
                all_stats.append(
                    prepare_one(
                        config,
                        split,
                        source,
                        normalizer,
                        args.limit,
                        args.overwrite,
                        log,
                        allow_row_loss=args.allow_row_loss,
                    )
                )
            except Exception:
                log.exception("failed preparing %s/%s", split, source)
                return 1

    manifest = {
        "generated_by": "scripts/prepare_data.py",
        "config": config.get("config_path"),
        "data_root": str(config["resolved"]["data_root"]),
        "normalization": {
            "unicode_form": normalizer.unicode_form,
            "fold_latin_accents": normalizer.fold_latin_accents,
            "case_mode": normalizer.case_mode,
            "punctuation_to_space": normalizer.punctuation_to_space,
            "max_length": normalizer.max_length,
        },
        "sources": all_stats,
        "total_seconds": round(time.time() - started, 1),
    }
    manifest_path = Path(config["resolved"]["prepared_dir"]) / "prepare_manifest.json"
    write_json(manifest_path, manifest)

    log.info("-" * 78)
    for stats in all_stats:
        log.info(
            "%-6s %-8s %12s rows%s",
            stats["split"],
            stats["source"],
            fmt_int(stats["rows"]),
            "  (skipped, already present)" if stats.get("skipped") else "",
        )
    log.info("manifest: %s", manifest_path)
    log.info("total %.1f min", (time.time() - started) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
