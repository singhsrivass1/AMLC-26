#!/usr/bin/env python
"""Download the dense-blocker encoder once, to a local directory.

The challenge forbids external API calls at runtime, and HPC compute nodes usually
have no internet anyway, so the dense blocker loads its model strictly from local
files (``blocking.dense.local_files_only: true``). Run this **once**, on a machine
that is allowed to reach the Hugging Face hub (the cluster login node, or a laptop
followed by an rsync), then point the config at the directory it wrote::

    python scripts/fetch_dense_model.py --output /scratch/$USER/models/bge-m3
    # configs/config.yaml -> blocking.dense.model_name_or_path: /scratch/$USER/models/bge-m3

Only the files the sentence-transformers dense path needs are fetched (~2.3 GB for
bge-m3). The ONNX export (another ~2.3 GB), the ColBERT/sparse heads and the README
images are skipped.

Model facts recorded for the challenge's licence / parameter-count constraints:
``BAAI/bge-m3`` - MIT licence, XLM-RoBERTa-large backbone, ~568M parameters,
1024-dim dense embeddings, 100+ languages (Devanagari, Kannada, Bengali included).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-m3"

# Everything the dense path does not load.
IGNORE_PATTERNS = [
    "onnx/*",
    "imgs/*",
    "*.jpg",
    "*.webp",
    "colbert_linear.pt",
    "sparse_linear.pt",
    ".DS_Store",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the dense-blocker encoder to a local directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="hub model id")
    parser.add_argument("--revision", default=None, help="pin a hub revision (commit sha) for reproducibility")
    parser.add_argument("--output", required=True, help="local directory to write the model into")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after downloading, load the model offline and encode one string",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 2

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.model} -> {output}")
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(output),
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"done: {path}")

    if args.verify:
        # Exercise exactly the load path the blocker uses, with the network off.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.blocking import SentenceTransformerEncoder

        encoder = SentenceTransformerEncoder(
            str(output), device="cpu", max_length=64, batch_size=8, local_files_only=True
        )
        vector = encoder.encode(["राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketing private limited"])
        print(f"verified offline load: dim={encoder.dim}, cross-script cosine={float(vector[0] @ vector[1]):.4f}")

    print(f"set blocking.dense.model_name_or_path: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
