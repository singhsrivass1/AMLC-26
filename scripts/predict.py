#!/usr/bin/env python
"""Stage 6: score candidate features and write ``matching_results.tsv``.

    python scripts/predict.py                                   # test split -> submission
    python scripts/predict.py --split train                     # dry run, scored vs ground truth
    python scripts/predict.py --model-dir outputs/experiments/v1 --threshold 0.42

Inputs:

* a trained bundle (``scripts/train_model.py`` -> ``<model-dir>/model/``), which
  carries the feature columns, the fold models and the tuned threshold;
* the split's candidate features (``scripts/extract_pair_features.py --split test
  --sample-fraction 1.0``), streamed in chunks - never loaded whole;
* the split's S1 entity list, read from the raw S1 file: it is the definition of
  which rows the submission must have.

Decision rule: every candidate pair scoring ``>= threshold`` is a match (one-to-many;
no top-1 step). Output: one line per S1 entity in S1 file order; the matched ids
comma-joined, deduplicated and sorted; **an exact empty string for an entity with
no predicted match** - including every entity the blockers proposed nothing for,
which is absent from the feature file altogether.

The written file is then validated byte by byte (``src.submission``) and the run
fails - rather than leaving a plausible file behind - if any rule is violated.
Before scoring anything, the feature file's own report is checked: a feature file
from a sampled or partial extraction would silently turn every unsampled entity
into a singleton, so it is refused unless ``--allow-partial-features``.

Outputs, next to ``--output``: the submission and ``<stem>_report.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    describe_environment,
    load_config,
    load_ground_truth,
    raw_path,
)
from src.evaluation import split_mask_for  # noqa: E402
from src.matching_model import (  # noqa: E402
    ID_COLUMNS,
    decide,
    iter_feature_chunks,
    load_bundle,
    predict,
)
from src.submission import (  # noqa: E402
    SUBMISSION_FILENAME,
    SubmissionError,
    load_s1_universe,
    score_submission,
    validate_submission,
    write_submission,
)
from src.utils import fmt_int, read_json, setup_logging, write_json  # noqa: E402

LOG_NAME = "predict"
CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN = ID_COLUMNS[0], ID_COLUMNS[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write matching_results.tsv (stage 6).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--test-data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", default="test", choices=["test", "train"],
                        help="test = the submission; train = a dry run scored against the ground truth")
    parser.add_argument("--model-dir", default=None,
                        help="train_model.py output dir (holds model/). Default: <work_dir>/experiments/v1")
    parser.add_argument("--features", default=None,
                        help="features.tsv for --split. Default: <work_dir>/experiments/"
                        "step3_features[_test]/features.tsv")
    parser.add_argument("--output", default=None,
                        help=f"submission path. Default: <work_dir>/submission/{SUBMISSION_FILENAME} "
                        "(train: train_matching_results.tsv)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the bundle's tuned threshold")
    parser.add_argument("--chunksize", type=int, default=500_000, help="feature rows per chunk")
    parser.add_argument("--allow-partial-features", action="store_true",
                        help="accept a feature file from a sampled/limited extraction (smoke tests only)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def default_paths(config: dict, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    work_dir = Path(config["resolved"]["work_dir"])
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else work_dir / "experiments" / "v1"
    feature_dir = "step3_features" if args.split == "train" else f"step3_features_{args.split}"
    features = (
        Path(args.features).expanduser().resolve()
        if args.features
        else work_dir / "experiments" / feature_dir / "features.tsv"
    )
    name = SUBMISSION_FILENAME if args.split == "test" else f"{args.split}_{SUBMISSION_FILENAME}"
    output = Path(args.output).expanduser().resolve() if args.output else work_dir / "submission" / name
    return model_dir, features, output


def check_feature_coverage(features: Path, split: str, log: logging.Logger) -> list[str]:
    """Reasons the feature file cannot stand for the whole split (empty = it can)."""
    report_path = features.parent / "step3_features_report.json"
    if not report_path.is_file():
        return [f"no extraction report next to the features ({report_path.name}); coverage unverifiable"]
    inputs = read_json(report_path).get("inputs", {})
    problems = []
    if inputs.get("split") != split:
        problems.append(f"features were extracted for split={inputs.get('split')!r}, not {split!r}")
    if float(inputs.get("sample_fraction", 0.0)) < 1.0:
        problems.append(f"features cover a {inputs.get('sample_fraction')} sample of the entities")
    if inputs.get("population", "val") != "all":
        problems.append(f"features cover population={inputs.get('population', 'val')!r}, not 'all'")
    if inputs.get("limit_rows"):
        problems.append(f"extraction scanned only the first {inputs['limit_rows']} candidate rows")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config,
        overrides={"data_root": args.data_root, "test_data_root": args.test_data_root, "work_dir": args.work_dir},
    )
    log = setup_logging(LOG_NAME, log_dir=config["resolved"]["log_dir"],
                        level=getattr(logging, args.log_level.upper(), logging.INFO))
    model_dir, features_path, output_path = default_paths(config, args)

    log.info("=" * 78)
    log.info("predict: candidate features -> %s", output_path.name)
    log.info(describe_environment(config))
    log.info("split=%s model=%s features=%s", args.split, model_dir, features_path)
    log.info("=" * 78)

    # -- model ------------------------------------------------------------
    try:
        bundle = load_bundle(model_dir)
    except (FileNotFoundError, ImportError) as error:
        log.error("%s", error)
        return 2
    if args.threshold is not None:
        log.warning("threshold override: %s (bundle tuned %s)", args.threshold, bundle.threshold)
        bundle.threshold = float(args.threshold)
    if not np.isfinite(bundle.threshold):
        log.error("bundle has no finite threshold; retrain or pass --threshold")
        return 2
    log.info("model=%s threshold=%.6g features=%s", bundle.model, bundle.threshold, len(bundle.feature_columns))

    # -- inputs -------------------------------------------------------------
    if not features_path.is_file():
        log.error("feature file not found: %s", features_path)
        log.error("Build it: python scripts/extract_pair_features.py --split %s --sample-fraction 1.0", args.split)
        return 2
    coverage_problems = check_feature_coverage(features_path, args.split, log)
    if coverage_problems:
        for problem in coverage_problems:
            (log.warning if args.allow_partial_features else log.error)("feature coverage: %s", problem)
        if not args.allow_partial_features:
            log.error("refusing: unfeaturized entities would be emitted as singletons "
                      "(--allow-partial-features for a smoke test)")
            return 2
    try:
        universe = load_s1_universe(config, args.split, log)
    except (SubmissionError, FileNotFoundError) as error:
        log.error("%s", error)
        return 2
    universe_set = set(universe)

    # -- score ----------------------------------------------------------------
    with open(features_path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    missing = [c for c in (*ID_COLUMNS, *bundle.feature_columns) if c not in header]
    if missing:
        log.error("feature file lacks columns the model needs: %s", missing)
        return 2

    started = time.time()
    matches: dict[str, set[str]] = {}
    rows = kept = 0
    featurized: set[str] = set()
    for chunk in iter_feature_chunks(
        features_path, [*ID_COLUMNS, *bundle.feature_columns], chunksize=args.chunksize
    ):
        s1_ids = chunk[CANDIDATE_S1_COLUMN].to_numpy(dtype=object)
        unknown = [s1 for s1 in set(s1_ids) if s1 not in universe_set]
        if unknown:
            log.error("feature rows name S1 ids outside the %s S1 universe, e.g. %s", args.split, unknown[:3])
            return 2
        featurized.update(s1_ids)
        probabilities = predict(config, bundle, chunk[list(bundle.feature_columns)])
        decisions = decide(config, bundle, probabilities)
        targets = chunk[CANDIDATE_TARGET_COLUMN].to_numpy(dtype=object)
        for s1, target in zip(s1_ids[decisions], targets[decisions]):
            matches.setdefault(s1, set()).add(target)
        rows += len(chunk)
        kept += int(decisions.sum())
    log.info("scored %s candidate rows in %.1f s; %s at or above the threshold",
             fmt_int(rows), time.time() - started, fmt_int(kept))

    # -- write + validate -----------------------------------------------------
    try:
        written = write_submission(output_path, universe, matches, config)
        validated = validate_submission(output_path, universe, config)
    except SubmissionError as error:
        log.error("%s", error)
        return 1

    report = {
        "split": args.split,
        "output": str(output_path),
        "model_dir": str(model_dir),
        "model": bundle.model,
        "threshold": bundle.threshold,
        "features": str(features_path),
        "feature_rows_scored": rows,
        "pairs_at_or_above_threshold": kept,
        "s1_universe": len(universe),
        "s1_with_candidate_features": len(featurized),
        "s1_without_candidates": len(universe) - len(featurized),
        **{f"written_{k}": v for k, v in written.items()},
        "validated": validated,
    }

    if args.split == "train":
        gt_path = raw_path(config, "train", "ground_truth")
        if gt_path.is_file():
            ground_truth = load_ground_truth(config, log=log)
            report["score_all"] = score_submission(output_path, ground_truth, config=config)
            report["score_val"] = score_submission(
                output_path, ground_truth, s1_mask=split_mask_for(ground_truth, config, "val"), config=config
            )
            log.info("dry-run macro F0.5 (score_zero): all=%.4f val=%.4f",
                     report["score_all"]["macro_f05_score_zero"], report["score_val"]["macro_f05_score_zero"])

    report_path = output_path.with_name(output_path.stem + "_report.json")
    write_json(report_path, report)
    log.info("-" * 78)
    log.info("S1 rows written       : %s", fmt_int(written["rows"]))
    log.info("  with >=1 match      : %s", fmt_int(written["matched_entities"]))
    log.info("  singletons (\"\")    : %s", fmt_int(written["singletons"]))
    log.info("  without candidates  : %s", fmt_int(report["s1_without_candidates"]))
    log.info("matched pairs         : %s", fmt_int(written["pairs"]))
    log.info("submission            : %s", output_path)
    log.info("report                : %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
