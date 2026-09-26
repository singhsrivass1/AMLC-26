#!/usr/bin/env python
"""Stage 5, milestone 2: train the V1 match classifier.  **IMPLEMENTED (V1).**

Trains the matcher on the features ``scripts/extract_pair_features.py`` already
wrote and reports the challenge metric of a threshold chosen on out-of-fold scores.
The heavy lifting lives in ``src/matching_model.py``; this script is the CLI.

What one run does, in order:

1. **Labels** every candidate row pair-level: 1 iff the exact
   ``(source1_entity_id, matched_entity_id)`` pair is in the ground truth, using the
   evaluator's own membership test. A missed match does not make an entity's other
   pairs negative. The label artifact is compact and row-aligned with the feature
   file; the 67.3M-row feature TSV is read, never rewritten.
2. **Folds** the S1 entities - 5 by default, grouped so every candidate row of one
   entity lands in one fold - and asserts that no entity spans two folds.
3. **Trains out-of-fold**: one LightGBM model per fold, each scoring the entities it
   never saw. Every reported number comes from those out-of-fold scores.
4. **Sweeps the threshold** against the official macro F0.5 (``score_zero`` policy)
   and picks a stable point from the middle of the best plateau.
5. Writes the metrics, the sweep, the out-of-fold probabilities and the model.

The stage before this one: the blocking report says how many true matches the
candidate set contains at all. A matcher cannot recover a match the blocker never
proposed, so read that report first - ``candidate_pair_recall_of_the_feature_file``
in this run's own baselines is the ceiling every model here is working under.

V1 deliberately does not use embeddings, a cross-encoder, LambdaRank, resampling or
``scale_pos_weight``, and does not predict top-1: every pair at or above the
threshold is a match, so a one-to-many entity keeps all of its surviving matches.

Examples::

    # dependency-free end-to-end check of the whole path (no LightGBM needed)
    python scripts/train_model.py --model threshold --sample-rows 200000 \\
        --output-dir outputs/experiments/v1_smoke

    # the real V1 run
    python scripts/train_model.py --features outputs/experiments/step3_features/features.tsv

    # a quick real-data shakedown before committing to the full run
    python scripts/train_model.py --sample-rows 2000000 \\
        --output-dir outputs/experiments/v1_shakedown

The output directory holds ``v1_metrics.json`` (everything), ``threshold_sweep.csv``
(the full curve), ``oof_probabilities.npy``, ``v1_labels_report.json`` and
``model/`` with the per-fold boosters and their metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import describe_environment, load_config  # noqa: E402
from src.matching_model import (  # noqa: E402
    DEFAULT_FOLDS,
    FEATURE_COLUMNS,
    FOLD_MODES,
    MODEL_LIGHTGBM,
    MODELS,
    MODEL_THRESHOLD,
    ZERO_MATCH_POLICY,
    count_data_rows,
    default_params,
    feature_matrix_bytes,
    resolve_fold_mode,
    train,
)
from src.utils import fmt_int, human_bytes, setup_logging  # noqa: E402

LOG_NAME = "train_model"

# Where scripts/extract_pair_features.py writes by default: work_dir/experiments/
# step3_features/. Mirrored here so `--features` can be omitted in the common case.
FEATURES_SUBDIR = Path("experiments") / "step3_features" / "features.tsv"

HOW_TO_BUILD_FEATURES = (
    "Build it first (the feature file is the matcher's input of record):\n"
    "  python scripts/extract_pair_features.py --split train --population all --sample-fraction 1.0\n"
    "Then point this script at the result:\n"
    "  python scripts/train_model.py --features <that output>/features.tsv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V1 match classifier and tune its decision threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--log-level", default="INFO")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--features",
        default=None,
        help=f"candidate feature TSV. Default: <work_dir>/{FEATURES_SUBDIR.as_posix()}",
    )
    data.add_argument("--ground-truth", default=None, help="ground truth TSV; default from config")
    data.add_argument(
        "--output-dir",
        default=None,
        help="where the run's artifacts go. Default: <work_dir>/experiments/v1",
    )
    data.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="read only the first N feature rows - for a smoke test or shakedown, "
        "not for a real run (the metrics are then reported on partial data)",
    )
    data.add_argument("--chunksize", type=int, default=500_000, help="rows per read chunk")
    data.add_argument(
        "--no-matrix",
        action="store_true",
        help="write labels only. Not usable with this script (there would be nothing to "
        "train on); it exists for the labels-only pass through build_label_artifacts()",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model", choices=MODELS, default=MODEL_LIGHTGBM)
    model.add_argument(
        "--score-feature",
        default="name_token_set_ratio",
        help="the --model threshold arm scores this single feature directly",
    )
    model.add_argument("--n-estimators", type=int, default=300, help="boosting rounds per fold")
    model.add_argument("--learning-rate", type=float, default=0.05)
    model.add_argument("--num-leaves", type=int, default=31)
    model.add_argument("--min-data-in-leaf", type=int, default=200)
    model.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="LightGBM threads; 0 = every core. Independent of compute.num_workers, "
        "which sizes the extractor's process pool",
    )
    model.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra LightGBM parameter, repeatable (e.g. --param lambda_l2=5)",
    )
    model.add_argument(
        "--final-model",
        choices=("ensemble", "retrain"),
        default="ensemble",
        help="ensemble = average the per-fold models; retrain = fit one more model on "
        "every row. The out-of-fold sweep is unaffected by either",
    )
    model.add_argument("--seed", type=int, default=42)

    folds = parser.add_argument_group("folds and threshold")
    folds.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    folds.add_argument(
        "--fold-mode",
        choices=FOLD_MODES,
        default="auto",
        help="auto = GroupKFold when scikit-learn is available, else the equivalent "
        "per-entity hash. Both keep every S1 entity's rows in one fold",
    )
    folds.add_argument(
        "--policy",
        choices=(ZERO_MATCH_POLICY, "exclude"),
        default=ZERO_MATCH_POLICY,
        help="zero_match_policy to tune under; score_zero is the challenge's own",
    )
    folds.add_argument("--sweep-points", type=int, default=200)
    folds.add_argument("--refine-points", type=int, default=50)
    folds.add_argument(
        "--plateau-tolerance",
        type=float,
        default=None,
        help="macro F0.5 slack that still counts as the same operating point",
    )
    return parser.parse_args(argv)


def resolve_features_path(config: dict, args: argparse.Namespace) -> Path:
    if args.features:
        return Path(args.features).expanduser().resolve()
    return (Path(config["resolved"]["work_dir"]) / FEATURES_SUBDIR).resolve()


def resolve_output_dir(config: dict, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (Path(config["resolved"]["work_dir"]) / "experiments" / "v1").resolve()


def coerce(value: str):
    """``KEY=VALUE`` from the CLI, in the most specific type that fits."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def build_params(args: argparse.Namespace) -> dict:
    """The LightGBM parameters for this run, from the defaults plus the CLI overrides."""
    params = default_params(seed=args.seed, num_threads=args.num_threads)
    params["learning_rate"] = args.learning_rate
    params["num_leaves"] = args.num_leaves
    params["min_data_in_leaf"] = args.min_data_in_leaf
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        params[key.strip()] = coerce(value)
    return params


def plan_memory(log, features_path: Path, sample_rows: int | None, model: str) -> None:
    """State the storage decision and its cost before doing any of it."""
    if model != MODEL_LIGHTGBM:
        return
    rows_in_file = count_data_rows(features_path)
    rows = min(rows_in_file, sample_rows) if sample_rows else rows_in_file
    log.info(
        "plan: %s rows in the feature file; the matrix is written memory-mapped "
        "(%s), so peak RSS stays near one chunk rather than the whole matrix",
        fmt_int(rows),
        feature_matrix_bytes(rows),
    )
    log.info(
        "      one fold's training copy is ~%s, released before the next fold starts",
        human_bytes(int(rows * len(FEATURE_COLUMNS) * 4 * 4 / 5)),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir}
    )
    log = setup_logging(LOG_NAME, log_dir=config["resolved"]["log_dir"], level=args.log_level)

    features_path = resolve_features_path(config, args)
    output_dir = resolve_output_dir(config, args)
    params = build_params(args)

    log.info("=" * 78)
    log.info("train_model: V1 matcher (%s)", args.model)
    log.info(describe_environment(config))
    log.info("=" * 78)
    log.info("features      : %s", features_path)
    log.info("output dir    : %s", output_dir)
    log.info(
        "folds         : %s (%s%s)",
        args.folds,
        args.fold_mode,
        "" if args.fold_mode != "auto" else f" -> {resolve_fold_mode(args.fold_mode)}",
    )
    log.info("threshold on  : macro F0.5, policy=%s", args.policy)
    if args.model == MODEL_LIGHTGBM:
        log.info(
            "lightgbm      : %s rounds, lr=%s, leaves=%s, min_data_in_leaf=%s, threads=%s",
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
            args.min_data_in_leaf,
            args.num_threads or "all cores",
        )
        log.info(
            "                CPU by default - for 27 features a well-threaded CPU build "
            "usually beats the GPU histogram path; benchmark before switching device"
        )
    else:
        log.info("threshold arm : fits nothing; scores %r directly", args.score_feature)
    if args.sample_rows:
        log.info(
            "SAMPLE ROWS   : %s - a smoke test / shakedown, NOT a real run. Metrics are "
            "reported on this prefix of the file and marked is_smoke",
            fmt_int(args.sample_rows),
        )
    log.info("=" * 78)

    if not features_path.is_file():
        log.error("feature file not found: %s", features_path)
        log.error("%s", HOW_TO_BUILD_FEATURES)
        return 2

    plan_memory(log, features_path, args.sample_rows, args.model)

    try:
        bundle = train(
            config,
            features_path,
            output_dir,
            ground_truth_path=args.ground_truth,
            model=args.model,
            score_feature=args.score_feature,
            folds=args.folds,
            fold_mode=args.fold_mode,
            seed=args.seed,
            chunksize=args.chunksize,
            sample_rows=args.sample_rows,
            write_matrix=not args.no_matrix,
            n_estimators=args.n_estimators,
            params=params,
            num_threads=args.num_threads,
            sweep_points=args.sweep_points,
            refine_points=args.refine_points,
            plateau_tolerance=args.plateau_tolerance,
            policy=args.policy,
            final_model=args.final_model,
            log=log,
        )
    except ImportError as exc:
        # The actionable text already says what to install or which arm to use.
        log.error("%s", exc)
        return 4
    except ValueError as exc:
        log.error("training failed: %s", exc)
        return 5

    log.info("%s", bundle.summary())
    log.info("done: metrics in %s", output_dir / "v1_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
