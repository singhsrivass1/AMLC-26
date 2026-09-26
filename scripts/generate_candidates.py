#!/usr/bin/env python
"""Stage 3: generate candidate pairs for every S1 entity.

Runs the configured blockers against the S2 and S3 indexes, unions their output,
deduplicates, and writes ``candidate_pairs.tsv`` incrementally.

The union is a **union**: a pair proposed by any enabled blocker is a candidate.
Blockers never intersect, because blocking's job is to be a safe over-approximation
- the matcher downstream can reject a false candidate, but nothing can recover a true
pair that blocking never proposed.

Each blocker queries its own key column (``name_norm`` for exact/token, ``name_key``
for char), which is why a single shared query column is not used, and each pair
carries the set of blockers that proposed it plus whatever per-pair evidence those
blockers measured.

Memory: O(rows per S1 chunk) - the candidate table is never held whole. Peak is
roughly ``chunk_s1 * avg_candidates * (8 bytes packed + ~60 bytes decoded)``.

    python scripts/generate_candidates.py
    python scripts/generate_candidates.py --limit-s1 100000            # smoke test
    python scripts/generate_candidates.py --max-candidates 50          # cap per S1
    python scripts/generate_candidates.py --blockers exact_name        # force one blocker
    python scripts/generate_candidates.py --workers 32                 # char verification

Outputs (under ``outputs/candidates/``, prefixed by the split so train and test
never overwrite each other)::

    {split}_candidate_pairs.tsv          source1_entity_id, matched_entity_id,
                                         source, blockers [, token_df]
                                         [, char_jaccard] [, dense_cosine]
    {split}_candidate_pairs_stats.json   volume + provenance statistics

The evidence columns are present only when their blocker is enabled. See
``src/blocking.py`` for the blocker registry and the semantics of each blocker.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (  # noqa: E402
    EVIDENCE_COLUMNS,
    decode_candidates,
    enabled_blockers,
    evidence_columns_for,
    load_index,
    truncate_per_group,
    union_blockers,
)
from src.data_loader import (  # noqa: E402
    ChunkWriter,
    SOURCE_PREFIX,
    TARGET_SOURCES,
    candidates_path,
    describe_environment,
    iter_prepared,
    load_config,
    prepared_path,
    require_file,
)
from src.utils import (  # noqa: E402
    auto_worker_count,
    fmt_int,
    log_memory,
    set_seed,
    setup_logging,
    track,
    write_json,
)

LOG_NAME = "generate_candidates"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate blocking candidate pairs (stage 3).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--sources", default=",".join(TARGET_SOURCES))
    parser.add_argument("--blockers", default=None,
                        help=f"comma-separated blockers to union; default: enabled blockers in config")
    parser.add_argument("--limit-s1", type=int, default=None, help="max S1 entities to process (smoke tests)")
    parser.add_argument("--chunksize", type=int, default=None, help="S1 rows per chunk")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="cap candidates per S1 (overrides blocking.max_candidates_per_source)")
    parser.add_argument("--workers", type=int, default=0,
                        help="verification processes for the char blocker (0 = auto)")
    parser.add_argument("--name", default="candidate_pairs", help="output file stem")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def resolve_verify_workers(config: dict, requested: int | None, log: logging.Logger) -> int:
    """Process count for the char blocker's Jaccard verification.

    ``resolve_workers`` is deliberately not used: it clamps the count to the number
    of chunks to process, and here the chunk count is not known until a query
    arrives - it depends on how many pairs a given S1 chunk retrieves.
    Over-provisioning is harmless (``CharNgramIndex._verify`` only hands out as many
    shards as there is work), so the clamp is simply not applied. Precedence is
    otherwise identical: CLI, then ``compute.num_workers``, then physical cores.

    This is a pure performance knob. Verification results do not depend on it.
    """
    configured = config.get("compute", {}).get("num_workers", 0)
    workers = max(1, int(requested or configured or 0) or auto_worker_count())
    log.info("char verification workers: %s", fmt_int(workers))
    return workers


def format_evidence(values: np.ndarray | None, fmt: str, size: int) -> np.ndarray:
    """One candidate-file evidence column, blank where the blocker supplied none.

    A pair proposed by ``token`` has no char Jaccard and vice versa, so blanks are
    the normal case rather than an error; they are written as an empty field so the
    column stays a single dtype and the file stays readable by the streaming readers.
    """
    out = np.full(size, "", dtype=object)
    if values is None or len(values) == 0:
        return out
    values = np.asarray(values, dtype=np.float64)
    present = ~np.isnan(values)
    if len(values) != size:
        raise ValueError(f"evidence has {len(values)} values but {size} pairs were written")
    if present.any():
        out[present] = np.char.mod(fmt, values[present])
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir})
    log = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    set_seed(config.get("project", {}).get("seed", 42))

    log.info("=" * 78)
    log.info("generate_candidates: blocking -> candidate pairs")
    log.info(describe_environment(config))
    log.info("=" * 78)

    s1_path = prepared_path(config, args.split, "source1")
    require_file(s1_path, hint="Run: python scripts/prepare_data.py")

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for source in sources:
        if source not in TARGET_SOURCES:
            log.error("candidates come from %s, got %r", TARGET_SOURCES, source)
            return 2

    try:
        blockers = enabled_blockers(config, args.blockers)
    except ValueError as error:
        log.error("%s", error)
        return 2
    log.info("blockers: %s", ", ".join(blockers))

    verify_workers = resolve_verify_workers(config, args.workers, log)

    # ---- load indexes -----------------------------------------------------
    indexes = {}
    for source in sources:
        for blocker in blockers:
            try:
                # verify=True: an index built at another cell, on another key field or
                # from a since-rewritten prepared table is refused, not silently used.
                index = load_index(
                    config, args.split, source, blocker, log=log, workers=verify_workers, verify=True
                )
            except (FileNotFoundError, NotImplementedError, ValueError, ImportError, OSError) as error:
                log.error("%s", error)
                return 2
            indexes[(source, blocker)] = index
            log.info("[%s] %s", source, index.describe())

    # Which normalized column each blocker keys on. These are expected to differ -
    # the token blocker keys on name_norm while the char blocker keys on name_key,
    # and forcing either onto the other's column would silently change the blocker
    # that was calibrated. Each index carries its own column's values, so the only
    # requirement is that every column exists in the S1 table.
    s1_key_fields = sorted({index.key_field for index in indexes.values()})
    log.info("S1 key fields: %s", ", ".join(s1_key_fields))

    max_candidates = (
        args.max_candidates
        if args.max_candidates is not None
        else config.get("blocking", {}).get("max_candidates_per_source", 0)
    )
    if max_candidates:
        log.info("candidate cap per S1: %s", fmt_int(max_candidates))

    # Split-aware name ({split}_{name}.tsv): a test run can never overwrite the train
    # candidates. legacy_fallback=False - a writer always writes the new name.
    output_path = candidates_path(config, args.name, split=args.split, legacy_fallback=False)
    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    log.info("output: %s", output_path)
    evidence_columns = evidence_columns_for(blockers)
    evidence_formats = {
        EVIDENCE_COLUMNS[b][0]: EVIDENCE_COLUMNS[b][1] for b in blockers if b in EVIDENCE_COLUMNS
    }
    if evidence_columns:
        log.info("evidence columns: %s", ", ".join(evidence_columns))

    chunksize = args.chunksize or config.get("io", {}).get("chunksize", 500_000)
    compression = config.get("io", {}).get("compression")

    # ---- stream S1 and emit candidates ------------------------------------
    started = time.time()
    s1_seen = 0
    total_pairs = 0
    s1_with_candidates = 0
    s1_without_candidates = 0
    pairs_by_source = {SOURCE_PREFIX[s]: 0 for s in sources}
    blocker_pair_counts = {b: 0 for b in blockers}
    max_per_s1 = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    def s1_chunks():
        nonlocal s1_seen
        for chunk in iter_prepared(
            config,
            args.split,
            "source1",
            columns=["entity_id", *s1_key_fields],
            chunksize=chunksize,
        ):
            if args.limit_s1 is not None and s1_seen + len(chunk) > args.limit_s1:
                chunk = chunk.iloc[: args.limit_s1 - s1_seen]
            s1_seen += len(chunk)
            yield chunk
            if args.limit_s1 is not None and s1_seen >= args.limit_s1:
                break

    with ChunkWriter(partial_path, compression=compression) as writer:
        for chunk in track(s1_chunks(), desc="generate", logger=log, total=args.limit_s1):
            # Chunk-local S1 positions are all the packing needs; the S1 id is
            # written alongside, and groups stay contiguous in the output.
            packed_by_blocker: dict[str, np.ndarray] = {}
            evidence_by_blocker: dict[str, dict[str, np.ndarray]] = {}

            for (source, blocker), index in indexes.items():
                # Each index queries its own key column. Nothing is looked up in
                # another blocker's column: a token key and a trigram key are not
                # comparable values, so a single shared query column is impossible
                # for a mixed union.
                packed, evidence = index.query(chunk[index.key_field])
                packed_by_blocker[f"{source}:{blocker}"] = packed
                if evidence:
                    evidence_by_blocker[f"{source}:{blocker}"] = evidence
                blocker_pair_counts[blocker] += int(len(packed))

            # --- union across (source, blocker) ---
            # Every array indexes the same chunk-local S1 positions, so packing
            # lets np.unique do union + dedupe + sort in a single call.
            s1_positions, entity_codes, provenance, evidence = union_blockers(
                packed_by_blocker, blocker_evidence=evidence_by_blocker, log=None
            )

            # --- optional cap ---
            if max_candidates:
                keep = truncate_per_group(s1_positions, max_candidates, log=log)
                if not keep.all():
                    s1_positions = s1_positions[keep]
                    entity_codes = entity_codes[keep]
                    provenance = provenance[keep]
                    evidence = {column: values[keep] for column, values in evidence.items()}

            counts_per_s1 = np.bincount(s1_positions, minlength=len(chunk)) if len(s1_positions) else np.zeros(len(chunk), dtype=np.int64)
            s1_with_candidates += int(np.count_nonzero(counts_per_s1))
            s1_without_candidates += int(np.count_nonzero(counts_per_s1 == 0))
            if len(counts_per_s1):
                max_per_s1 = max(max_per_s1, int(counts_per_s1.max()))

            if len(entity_codes) == 0:
                continue

            # --- decode and write ---
            # entity_codes are packed (source * 10**10 + numeric), so the source
            # label comes back with the id - no per-index bookkeeping needed even
            # though the union already merged S2 and S3 into one array.
            s1_ids = chunk["entity_id"].to_numpy(dtype=object)[s1_positions]
            target_ids, source_labels = decode_candidates(entity_codes)
            for prefix in pairs_by_source:
                pairs_by_source[prefix] += int(np.count_nonzero(source_labels == prefix))

            frame = pd.DataFrame(
                {
                    "source1_entity_id": s1_ids,
                    "matched_entity_id": target_ids,
                    "source": source_labels,
                    "blockers": provenance,
                }
            )
            # Evidence columns are appended only for the blockers that are enabled,
            # so an exact-name-only run keeps the original four-column schema.
            for column in evidence_columns:
                frame[column] = format_evidence(
                    evidence.get(column), evidence_formats[column], len(entity_codes)
                )
            total_pairs += writer.append(frame)
            log_memory(log, f"chunk done, {fmt_int(total_pairs)} pairs so far")

    # ---- finalize ---------------------------------------------------------
    if not partial_path.is_file():
        # No chunk produced a pair, so the writer never opened the file. Emit the
        # header-only file: "no candidates" is a valid result, not a crash.
        header = ["source1_entity_id", "matched_entity_id", "source", "blockers", *evidence_columns]
        ChunkWriter(partial_path, compression=compression).append_header(header)
    written = _count_rows(partial_path)
    if written != total_pairs:
        log.error("integrity check failed: wrote %s rows, file has %s", fmt_int(total_pairs), fmt_int(written))
        return 1
    partial_path.replace(output_path)

    elapsed = time.time() - started
    stats = {
        "generated_by": "scripts/generate_candidates.py",
        "split": args.split,
        "blockers": blockers,
        "s1_key_fields": s1_key_fields,
        "evidence_columns": evidence_columns,
        "char_verify_workers": verify_workers,
        "indexes": {f"{source}:{blocker}": index.describe() for (source, blocker), index in indexes.items()},
        "s1_entities_processed": int(s1_seen),
        "s1_with_candidates": int(s1_with_candidates),
        "s1_without_candidates": int(s1_without_candidates),
        "candidate_pairs": int(total_pairs),
        "avg_candidates_per_s1": total_pairs / s1_seen if s1_seen else 0.0,
        "max_candidates_per_s1_observed": int(max_per_s1),
        "candidate_cap": max_candidates or None,
        "pairs_by_source": pairs_by_source,
        "pairs_by_blocker_before_union": blocker_pair_counts,
        "union_dedupe_saved": int(sum(blocker_pair_counts.values()) - total_pairs),
        "output": str(output_path),
        "seconds": round(elapsed, 1),
    }
    write_json(stats_path, stats)

    log.info("=" * 78)
    log.info("S1 processed            : %s", fmt_int(s1_seen))
    log.info("S1 with candidates      : %s", fmt_int(s1_with_candidates))
    log.info("S1 without candidates   : %s", fmt_int(s1_without_candidates))
    log.info("candidate pairs         : %s", fmt_int(total_pairs))
    log.info("avg candidates per S1   : %.3f", stats["avg_candidates_per_s1"])
    log.info("max candidates for one S1: %s", fmt_int(max_per_s1))
    for prefix, count in pairs_by_source.items():
        log.info("  %s pairs: %s", prefix, fmt_int(count))
    log.info("blocker contributions (pre-union): %s", blocker_pair_counts)
    log.info("stats: %s", stats_path)
    log.info("next: python scripts/evaluate_blocking.py --split val")
    log.info("total %.1f min", elapsed / 60.0)
    return 0


def _count_rows(path: Path) -> int:
    import gzip

    if not path.is_file():
        return 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


if __name__ == "__main__":
    raise SystemExit(main())
