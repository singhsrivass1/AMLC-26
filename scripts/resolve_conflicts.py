#!/usr/bin/env python
"""Post-process raw match probabilities: target-conflict resolution + an F0.5 cut-off.

    # 1. raw scores from the model (every pair, before threshold/resolution)
    python scripts/predict.py --split test --scores-out auto
    # 2. tune the cut-off WITH resolution applied, on out-of-fold train scores
    python scripts/resolve_conflicts.py --tune --model-dir outputs/experiments/v1
    # 3. apply it -> outputs/submission/resolved/matching_results.tsv
    python scripts/resolve_conflicts.py --split test --threshold 0.62

Target-conflict resolution (greedy, global)
-------------------------------------------
S1 is the deduplicated reference list, so an S2/S3 record belongs to at most one S1.
All predictions are ordered by probability, descending; each target goes to its
first (highest-probability) claimant and every later, lower-probability claim on the
same target is dropped. An S1 may keep any number of targets. Because only the
target side is capacity-limited, this greedy walk is exactly "argmax per target",
computed with one sort (``src.submission.best_claim_mask``); an exact tie goes to
the S1 that comes first in S1 file order. Threshold and resolution commute - a
target's best claim passes the cut-off iff any claim does - so the order of the two
steps does not change the result.

Asymmetric threshold
--------------------
``--threshold`` is the probability cut-off for a match. Under F0.5 a false positive
costs 4x a false negative, so the right cut-off sits well above 0.5 - but it also
moves once resolution removes false positives, so a threshold tuned *without*
resolution is not optimal *with* it. ``--tune`` sweeps it with resolution applied,
on the out-of-fold probabilities train_model.py saved (scores no model was fit on;
in-sample scores would push the cut-off too high), and reports the macro F0.5
(``score_zero``, the challenge rule) at every cut-off, with and without resolution.
Caveat: test scores come from the fold-model ensemble, whose scores are slightly
smoother than a single fold model's - the same caveat the bundle's own threshold has.

Output: the exact submission format (``src.submission``): every S1 once, in S1 file
order; matched ids comma-joined, deduplicated, sorted; an exact empty string for an
S1 with no surviving match. Validated byte by byte before the run succeeds.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import describe_environment, iter_tsv, load_config, load_ground_truth, raw_path  # noqa: E402
from src.evaluation import CandidateEvaluation, split_mask_for  # noqa: E402
from src.submission import (  # noqa: E402
    SUBMISSION_FILENAME,
    SubmissionError,
    best_claim_mask,
    ground_truth_target_multiplicity,
    group_matches,
    load_s1_universe,
    resolve_target_conflicts,
    score_submission,
    validate_submission,
    write_submission,
)
from src.utils import encode_entity_ids, fmt_int, read_json, setup_logging, write_json  # noqa: E402

LOG_NAME = "resolve_conflicts"
S1_COLUMN, TARGET_COLUMN, SCORE_COLUMN = "source1_entity_id", "matched_entity_id", "probability"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target-conflict resolution + F0.5 cut-off -> submission.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--test-data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--scores", default=None,
                        help="raw scores TSV from predict.py --scores-out. Default: "
                        "<work_dir>/submission/{split}_scores.tsv")
    parser.add_argument("--threshold", default=None,
                        help="probability cut-off (e.g. 0.62); 'bundle' = the model's own tuned threshold. "
                        "Required unless --tune")
    parser.add_argument("--no-resolve", action="store_true",
                        help="apply the threshold only (for comparison runs)")
    parser.add_argument("--output", default=None,
                        help=f"default: <work_dir>/submission/resolved/{SUBMISSION_FILENAME} "
                        "(train: train_matching_results.tsv)")
    parser.add_argument("--chunksize", type=int, default=2_000_000, help="score rows per chunk")

    tune = parser.add_argument_group("tuning (out-of-fold, train)")
    tune.add_argument("--tune", action="store_true", help="sweep the cut-off instead of writing a submission")
    tune.add_argument("--model-dir", default=None,
                      help="train_model.py output dir (oof_probabilities.npy etc.). Default: <work_dir>/experiments/v1")
    tune.add_argument("--features", default=None,
                      help="the train features.tsv the model was trained on. Default: from v1_labels_report.json")
    tune.add_argument("--tune-population", default="val", choices=["val", "all"],
                      help="S1 entities the macro F0.5 is averaged over (out-of-fold scores make 'all' valid too)")
    tune.add_argument("--grid", type=int, default=99, help="cut-offs between 0.01 and 0.99")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if not args.tune and args.threshold is None:
        parser.error("--threshold is required (a number, or 'bundle'), unless --tune")
    return args


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def read_scores_above(path: Path, threshold: float, chunksize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(s1, target, score, rows_read)`` for the scored pairs at or above ``threshold``."""
    s1_parts, target_parts, score_parts, rows = [], [], [], 0
    for chunk in iter_tsv(path, columns=[S1_COLUMN, TARGET_COLUMN, SCORE_COLUMN], chunksize=chunksize):
        rows += len(chunk)
        score = chunk[SCORE_COLUMN].astype(np.float64).to_numpy()
        keep = score >= threshold
        if keep.any():
            s1_parts.append(chunk[S1_COLUMN].to_numpy(dtype=object)[keep])
            target_parts.append(chunk[TARGET_COLUMN].to_numpy(dtype=object)[keep])
            score_parts.append(score[keep])
    if not s1_parts:
        empty = np.empty(0, dtype=object)
        return empty, empty, np.empty(0, dtype=np.float64), rows
    return np.concatenate(s1_parts), np.concatenate(target_parts), np.concatenate(score_parts), rows


def apply(config: dict, args: argparse.Namespace, log: logging.Logger) -> int:
    work = Path(config["resolved"]["work_dir"])
    scores_path = Path(args.scores).expanduser().resolve() if args.scores else \
        work / "submission" / f"{args.split}_scores.tsv"
    meta_path = scores_path.with_name(scores_path.name + ".meta.json")
    if not scores_path.is_file():
        log.error("scores file not found: %s\n  Make it: python scripts/predict.py --split %s --scores-out auto",
                  scores_path, args.split)
        return 2
    meta = read_json(meta_path) if meta_path.is_file() else {}
    if not meta:
        log.warning("no %s next to the scores; split and score floor cannot be checked", meta_path.name)
    if meta.get("split") not in (None, args.split):
        log.error("scores were produced for split=%r, not %r", meta.get("split"), args.split)
        return 2

    if args.threshold == "bundle":
        if "bundle_threshold" not in meta:
            log.error("--threshold bundle needs the scores sidecar (bundle_threshold)")
            return 2
        threshold = float(meta["bundle_threshold"])
    else:
        threshold = float(args.threshold)
    floor = float(meta.get("floor", 0.0))
    if threshold < floor:
        # Pairs under the floor were never written, so a lower cut-off would silently
        # behave like the floor.
        log.error("--threshold %s is below the scores file's floor %s (rerun predict.py with a lower "
                  "--scores-floor)", threshold, floor)
        return 2
    if threshold <= 0.5:
        log.warning("threshold %.4f <= 0.5: under F0.5 a false positive costs 4x a miss; prefer a "
                    "cut-off from --tune", threshold)

    name = SUBMISSION_FILENAME if args.split == "test" else f"{args.split}_{SUBMISSION_FILENAME}"
    output = Path(args.output).expanduser().resolve() if args.output else work / "submission" / "resolved" / name
    log.info("scores=%s threshold=%.6g resolve=%s -> %s", scores_path, threshold, not args.no_resolve, output)

    try:
        universe = load_s1_universe(config, args.split, log)
    except (SubmissionError, FileNotFoundError) as error:
        log.error("%s", error)
        return 2
    started = time.time()
    s1, targets, scores, rows = read_scores_above(scores_path, threshold, args.chunksize)
    log.info("read %s scored pairs in %.1f s; %s at or above %.6g",
             fmt_int(rows), time.time() - started, fmt_int(len(s1)), threshold)
    unknown = pd.Index(s1).difference(pd.Index(universe))
    if len(unknown):
        log.error("scores name S1 ids outside the %s S1 universe, e.g. %s", args.split, list(unknown[:3]))
        return 2

    s1_order = {entity: position for position, entity in enumerate(universe)}
    keep, stats = resolve_target_conflicts(s1, targets, scores, s1_order)
    log.info("conflicts: %s of %s predicted targets had >1 S1 claimant (%s exact ties); dropped %s "
             "lower-probability claims; %s S1 lost every match",
             fmt_int(stats["targets_with_conflicts"]), fmt_int(stats["targets"]), fmt_int(stats["tied_best_scores"]),
             fmt_int(stats["pairs_dropped"]), fmt_int(stats["s1_left_without_matches"]))
    use = np.ones(len(s1), dtype=bool) if args.no_resolve else keep

    try:
        matches = group_matches(s1[use], targets[use])
        written = write_submission(output, universe, matches, config)
        validated = validate_submission(output, universe, config)
    except SubmissionError as error:
        log.error("%s", error)
        return 1
    if not args.no_resolve:
        owner: dict[str, str] = {}
        for entity, claimed in matches.items():
            for target in claimed:
                if owner.setdefault(target, entity) != entity:
                    log.error("invariant broken: %s written for both %s and %s", target, owner[target], entity)
                    return 1

    report = {
        "split": args.split, "scores": str(scores_path), "output": str(output), "threshold": threshold,
        "resolved": not args.no_resolve, "scores_meta": meta, "rows_read": rows, "pairs_at_or_above": int(len(s1)),
        "conflicts": stats, **{f"written_{k}": v for k, v in written.items()}, "validated": validated,
    }
    if args.split == "train" and raw_path(config, "train", "ground_truth").is_file():
        ground_truth = load_ground_truth(config, log=log)
        val = split_mask_for(ground_truth, config, "val")
        other = output.with_name(output.stem + ("_resolved" if args.no_resolve else "_unresolved") + output.suffix)
        other_use = keep if args.no_resolve else np.ones(len(s1), dtype=bool)
        write_submission(other, universe, group_matches(s1[other_use], targets[other_use]), config)
        mine, theirs = ("unresolved", "resolved") if args.no_resolve else ("resolved", "unresolved")
        report["score"] = {
            mine: {"all": score_submission(output, ground_truth, config=config)["macro_f05_score_zero"],
                   "val": score_submission(output, ground_truth, s1_mask=val, config=config)["macro_f05_score_zero"]},
            theirs: {"all": score_submission(other, ground_truth, config=config)["macro_f05_score_zero"],
                     "val": score_submission(other, ground_truth, s1_mask=val, config=config)["macro_f05_score_zero"]},
        }
        report["ground_truth_targets"] = ground_truth_target_multiplicity(ground_truth)
        log.info("dry-run macro F0.5 (score_zero): %s", report["score"])

    write_json(output.with_name(output.stem + "_report.json"), report)
    log.info("S1 rows %s | with a match %s | singletons %s | pairs %s -> %s",
             fmt_int(written["rows"]), fmt_int(written["matched_entities"]), fmt_int(written["singletons"]),
             fmt_int(written["pairs"]), output)
    return 0


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------
def load_oof(model_dir: Path, features: Optional[str], ground_truth, log: logging.Logger) -> dict:
    """Out-of-fold scores with owner / label / target code, row-aligned and verified.

    ``train_model.py`` saved the scores and labels aligned to the feature rows it
    labelled (the first ``rows_read`` rows of the feature file, minus rows whose S1
    is not in the ground truth). The target ids are not saved, so they are re-read
    from the feature file with the same row selection - and the recomputed owner
    index must equal the saved one, which proves the alignment instead of assuming it.
    """
    probabilities = np.load(model_dir / "oof_probabilities.npy")
    owners = np.load(model_dir / "val_owner_index.npy").astype(np.int64)
    labels = np.load(model_dir / "val_labels.npy").astype(bool)
    report = read_json(model_dir / "v1_labels_report.json")
    path = Path(features) if features else Path(report["features_path"])
    to_read = int(report["rows_read"])
    log.info("re-reading target ids for %s rows of %s", fmt_int(to_read), path)

    owner_parts, code_parts, read = [], [], 0
    for chunk in iter_tsv(path, columns=[S1_COLUMN, TARGET_COLUMN], chunksize=2_000_000):
        if read + len(chunk) > to_read:
            chunk = chunk.iloc[: to_read - read]
        read += len(chunk)
        position = ground_truth.positions_of(chunk[S1_COLUMN])
        known = position >= 0
        owner_parts.append(position[known])
        code_parts.append(encode_entity_ids(chunk[TARGET_COLUMN][known]))
        if read >= to_read:
            break
    recomputed = np.concatenate(owner_parts) if owner_parts else np.empty(0, dtype=np.int64)
    if len(recomputed) != len(probabilities) or not np.array_equal(recomputed, owners):
        raise ValueError(
            f"cannot align {path} with the saved out-of-fold scores ({len(recomputed)} rows vs "
            f"{len(probabilities)}); pass the exact feature file train_model.py was run on via --features"
        )
    return {"scores": probabilities.astype(np.float64), "owners": owners, "labels": labels,
            "targets": np.concatenate(code_parts)}


def tune(config: dict, args: argparse.Namespace, log: logging.Logger) -> int:
    work = Path(config["resolved"]["work_dir"])
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else work / "experiments" / "v1"
    ground_truth = load_ground_truth(config, log=log)
    try:
        oof = load_oof(model_dir, args.features, ground_truth, log)
    except (FileNotFoundError, KeyError, ValueError) as error:
        log.error("%s", error)
        return 2
    bundle_threshold = float(read_json(model_dir / "model" / "model_meta.json").get("threshold", float("nan")))

    grid = np.round(np.linspace(0.01, 0.99, args.grid), 4)
    extra = [bundle_threshold] + ([float(args.threshold)] if args.threshold not in (None, "bundle") else [])
    grid = np.unique(np.concatenate([grid, [t for t in extra if np.isfinite(t)]]))

    # Resolution is threshold-independent (it commutes with the cut-off), so the keep
    # mask is computed once over every row; each cut-off is then a filter.
    started = time.time()
    keep = best_claim_mask(oof["targets"], oof["scores"], oof["owners"])
    live = oof["scores"] >= grid.min()  # rows no cut-off in the grid can select are dropped once
    scores, owners, labels, keep = oof["scores"][live], oof["owners"][live], oof["labels"][live], keep[live]
    mask = split_mask_for(ground_truth, config, "val") if args.tune_population == "val" else \
        np.ones(ground_truth.n_entities, dtype=bool)
    lengths = ground_truth.lengths()[mask]
    n = ground_truth.n_entities

    def macro(selected: np.ndarray) -> tuple[float, int]:
        counts = np.bincount(owners[selected], minlength=n)[mask]
        hits = np.bincount(owners[selected & labels], minlength=n)[mask]
        return CandidateEvaluation._macro_f05(lengths, counts, hits, policy="score_zero"), int(counts.sum())

    rows = []
    for threshold in grid:
        passed = scores >= threshold
        resolved, resolved_pairs = macro(passed & keep)
        raw, raw_pairs = macro(passed)
        rows.append({"threshold": float(threshold), "macro_f05_resolved": resolved, "macro_f05_unresolved": raw,
                     "pairs_resolved": resolved_pairs, "pairs_unresolved": raw_pairs})
    table = pd.DataFrame(rows)
    best_resolved = table.loc[table["macro_f05_resolved"].idxmax()]
    best_raw = table.loc[table["macro_f05_unresolved"].idxmax()]
    at = lambda t: table.loc[(table["threshold"] - t).abs().idxmin()]  # noqa: E731

    out_dir = work / "submission" / "resolved"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "threshold_sweep_resolved.tsv", sep="\t", index=False)
    summary = {
        "model_dir": str(model_dir), "population": args.tune_population, "oof_rows": int(len(oof["scores"])),
        "best_resolved": best_resolved.to_dict(), "best_unresolved": best_raw.to_dict(),
        "bundle_threshold": bundle_threshold,
        "at_bundle_threshold": at(bundle_threshold).to_dict() if np.isfinite(bundle_threshold) else None,
        "at_requested_threshold": at(float(args.threshold)).to_dict()
        if args.threshold not in (None, "bundle") else None,
        "ground_truth_targets": ground_truth_target_multiplicity(ground_truth),
        "seconds": round(time.time() - started, 1),
    }
    write_json(out_dir / "threshold_tuning.json", summary)

    log.info("-" * 78)
    log.info("out-of-fold macro F0.5 (score_zero, %s S1 entities):", args.tune_population)
    log.info("  best WITH resolution    : %.4f at threshold %.4f",
             best_resolved["macro_f05_resolved"], best_resolved["threshold"])
    log.info("  best WITHOUT resolution : %.4f at threshold %.4f",
             best_raw["macro_f05_unresolved"], best_raw["threshold"])
    if summary["at_bundle_threshold"]:
        log.info("  bundle threshold %.4f   : %.4f with / %.4f without resolution", bundle_threshold,
                 summary["at_bundle_threshold"]["macro_f05_resolved"],
                 summary["at_bundle_threshold"]["macro_f05_unresolved"])
    if summary["at_requested_threshold"]:
        log.info("  requested %.4f          : %.4f with / %.4f without resolution", float(args.threshold),
                 summary["at_requested_threshold"]["macro_f05_resolved"],
                 summary["at_requested_threshold"]["macro_f05_unresolved"])
    shared = summary["ground_truth_targets"]["targets_with_multiple_s1"]
    if shared:
        log.warning("the ground truth gives %s targets to more than one S1: resolution can delete true "
                    "matches there - trust the with/without comparison above, not the premise", fmt_int(shared))
    log.info("apply: python scripts/resolve_conflicts.py --split test --threshold %.4f", best_resolved["threshold"])
    log.info("sweep: %s", out_dir / "threshold_sweep_resolved.tsv")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config,
        overrides={"data_root": args.data_root, "test_data_root": args.test_data_root, "work_dir": args.work_dir},
    )
    log = setup_logging(LOG_NAME, log_dir=config["resolved"]["log_dir"],
                        level=getattr(logging, args.log_level.upper(), logging.INFO))
    log.info("resolve_conflicts: %s | %s", "tune" if args.tune else "apply", describe_environment(config))
    return tune(config, args, log) if args.tune else apply(config, args, log)


if __name__ == "__main__":
    raise SystemExit(main())
