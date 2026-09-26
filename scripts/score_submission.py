#!/usr/bin/env python
"""Validate a ``matching_results.tsv`` and (train split) score it against the ground truth.

    python scripts/score_submission.py outputs/submission/matching_results.tsv --split test
    python scripts/score_submission.py outputs/submission/train_matching_results.tsv --split train

Validation always runs (every S1 exactly once, in S1 order; singletons an exact
``""``; valid, deduplicated ids). Scoring needs a ground truth, so it runs for the
train split only, and reports macro F0.5 under the challenge's ``score_zero`` rule
over all entities and over the val split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_config, load_ground_truth  # noqa: E402
from src.evaluation import split_mask_for  # noqa: E402
from src.submission import (  # noqa: E402
    SubmissionError,
    load_s1_universe,
    score_submission,
    validate_submission,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--test-data-root", default=None)
    args = parser.parse_args(argv)

    config = load_config(
        args.config, overrides={"data_root": args.data_root, "test_data_root": args.test_data_root}
    )
    try:
        universe = load_s1_universe(config, args.split)
        result = {"validation": validate_submission(args.submission, universe, config)}
    except (SubmissionError, FileNotFoundError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    if args.split == "train":
        ground_truth = load_ground_truth(config, log=None)
        result["score_all"] = score_submission(args.submission, ground_truth, config=config)
        result["score_val"] = score_submission(
            args.submission, ground_truth, s1_mask=split_mask_for(ground_truth, config, "val"), config=config
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
