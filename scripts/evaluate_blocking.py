#!/usr/bin/env python
"""Stage 4: evaluate blocking quality against the training ground truth.

NOTE: this script is an addition to the file list in the project brief. The brief
requires evaluation against the ground truth (item 10) and requires heavy
computation to be runnable from the command line, and none of the five listed
scripts is the right home for it - ``train_model.py`` trains a matcher, which is
a later milestone. Move the logic if you prefer a different layout.

What it answers:

* Did blocking retrieve the true matches? (pair recall, S1 full/partial recall)
* At what cost? (candidates per S1, reduction ratio, precision)
* If every candidate were accepted, what F0.5 would we get? This is the ceiling
  the current blockers impose on any downstream matcher, and it says how much
  precision work is left for the ranking model.

Validation is reported **split by S1 entity**: candidate pairs of one S1 never
straddle train/val, so the val numbers are not contaminated.

    python scripts/evaluate_blocking.py --split val
    python scripts/evaluate_blocking.py --split all --limit-rows 2000000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    candidates_path,
    describe_environment,
    load_config,
    load_ground_truth,
)
from src.evaluation import (  # noqa: E402
    CandidateEvaluation,
    format_report,
    save_metrics,
    split_mask_for,
)
from src.utils import fmt_int, log_memory, read_json, set_seed, setup_logging  # noqa: E402

LOG_NAME = "evaluate_blocking"


def target_record_count(config: dict, args, log: logging.Logger) -> int | None:
    """Total S2+S3 record count, used for the reduction-ratio denominator.

    Read from the prepare manifest so we do not rescan 1GB of TSVs. Returns
    ``None`` when the manifest is unavailable, in which case the reduction ratio
    is simply omitted rather than guessed.
    """
    manifest_path = Path(config["resolved"]["prepared_dir"]) / "prepare_manifest.json"
    if not manifest_path.is_file():
        log.warning("no prepare manifest at %s; reduction ratio will be omitted", manifest_path)
        return None
    try:
        manifest = read_json(manifest_path)
    except Exception:
        log.warning("could not parse %s; reduction ratio will be omitted", manifest_path)
        return None

    total = 0
    for entry in manifest.get("sources", []):
        if entry.get("split") == args.split_candidates and entry.get("source") in ("source2", "source3"):
            total += int(entry.get("rows", 0))
    return total or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate blocking against ground truth (stage 4).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--candidates", default="candidate_pairs", help="candidate file stem")
    parser.add_argument("--split-candidates", default="train", help="which candidate run to evaluate")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "train", "all"],
        help="which S1 entities to score (val/train are subsets of the training GT)",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--limit-rows", type=int, default=None, help="max candidate rows to read (smoke tests)")
    parser.add_argument("--no-save", action="store_true", help="do not write the metrics JSON")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


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
    log.info("evaluate_blocking: recall and volume of the candidate set")
    log.info(describe_environment(config))
    log.info("=" * 78)

    ground_truth = load_ground_truth(config, log=log)
    log_memory(log, "after ground truth")

    n_targets = target_record_count(config, args, log)
    if n_targets:
        log.info("target records: %s", fmt_int(n_targets))

    k_values = config.get("evaluation", {}).get("k_values", [10, 25, 50, 100, 200])
    evaluator = CandidateEvaluation(ground_truth, n_target_records=n_targets, k_values=k_values, log=log)

    if args.split_candidates != "train":
        log.error("only train candidates can be scored: the ground truth covers the train split only")
        return 2
    candidate_file = candidates_path(config, args.candidates, split=args.split_candidates)
    started = time.time()
    metrics_all = evaluator.evaluate_file(candidate_file, chunksize=args.chunksize, max_rows=args.limit_rows)
    elapsed = time.time() - started
    log.info("read %s candidate rows in %.1f min", fmt_int(metrics_all["candidate_rows_read"]), elapsed / 60.0)

    # The evaluator accumulates counters once; the masks below only re-aggregate
    # them, so scoring three subsets costs nothing extra.
    reports = {}
    if args.split in ("val", "all"):
        mask = split_mask_for(ground_truth, config, split="val")
        reports["val"] = evaluator.compute_metrics(s1_mask=mask, split_label="val")
        log.info(
            "val split: %s of %s S1 entities (%.1f%%)",
            fmt_int(int(mask.sum())),
            fmt_int(len(mask)),
            100.0 * mask.mean(),
        )
    if args.split == "all":
        reports["all"] = metrics_all
        train_mask = split_mask_for(ground_truth, config, split="train")
        reports["train"] = evaluator.compute_metrics(s1_mask=train_mask, split_label="train")
    elif args.split == "train":
        train_mask = split_mask_for(ground_truth, config, split="train")
        reports["train"] = evaluator.compute_metrics(s1_mask=train_mask, split_label="train")

    for label, report in reports.items():
        print()
        print(format_report(report))

    if not args.no_save:
        for label, report in reports.items():
            out = Path(config["resolved"]["candidates_dir"]) / f"blocking_metrics_{args.candidates}_{label}.json"
            save_metrics(report, out)
            log.info("metrics saved: %s", out)

    if args.limit_rows:
        log.warning("--limit-rows was set: metrics describe a prefix of the candidate file, not the full run")

    log_memory(log, "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
