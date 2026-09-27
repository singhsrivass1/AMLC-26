#!/usr/bin/env python
"""Dense (bge-m3) candidate generation on GPU: exact cosine top-K per S1 entity.

    # one GPU node, every visible GPU used, test split
    python scripts/generate_dense_candidates.py --split test
    # explicit devices / a quick timing run first
    python scripts/generate_dense_candidates.py --split test --devices cuda:0,cuda:1
    python scripts/generate_dense_candidates.py --split test --limit-s1 200000 --limit-targets 1000000

What it does
------------
1. Loads ``name_norm`` for S1 and for S2+S3 from the prepared tables, and
   **deduplicates identical texts** on each side (S2/S3 hold ~8.1M distinct names
   in 10.3M rows, S1 ~1.5M in 2.2M): a text is encoded and searched once.
2. **Encodes** every distinct text with BAAI/bge-m3 (CLS pooling + L2 norm - the
   model's own dense head), from local files only, fp16 on GPU, texts sorted by
   length so a batch pads to its own longest name, tokenization prefetched on a
   CPU thread so the GPU never waits. Batches are interleaved across all devices.
   Embeddings are cached (float16 ``.npy`` + meta) and reused by a rerun.
3. **Exact search** on GPU: S1 embeddings resident, target embeddings streamed in
   blocks, scores computed one ``q_tile x t_sub`` tile at a time (never the full
   matrix), top-K selected per tile in fp16 with a margin, the survivors **re-scored
   in fp32**, and a running top-K merged per query. The result is exact cosine
   over *all* targets - no approximate index. Queries are split across devices.
4. **Expands** distinct target texts back to records (a name shared by several
   records yields all of them, capped at ``--top-k`` records per S1 - or per S1 and
   target source with ``--scope per-source``, the dense blocker's own semantics:
   top-k from S2 *and* top-k from S3, which keeps one source's near-duplicates from
   crowding out the other's true match) and writes

       {split}_dense_candidates.tsv
           source1_entity_id  matched_entity_id  source  blockers        dense_cosine
           S1-12              S2-9001            S2      source2:dense   0.8731

   - the dense blocker's rows exactly as ``generate_candidates.py`` writes them.
5. **Merges** (``--merge-with``, default: the split's string candidate file when it
   exists) into ``{split}_candidate_pairs_dense_union.tsv``: the union of both
   files, ``blockers`` provenance combined (``source2:exact_name,source2:dense``),
   ``token_df``/``char_jaccard`` kept, ``dense_cosine`` added - i.e. the file
   ``generate_candidates.py`` would have written with the dense blocker enabled,
   ready for ``extract_pair_features.py --candidates candidate_pairs_dense_union``.

IMPORTANT for the matcher: a model trained on string-only candidates has never seen
a non-blank ``dense_cosine`` or ``blocker_dense = 1``, so it will reject dense-only
pairs. Run this script for the train split too, re-extract train features from the
union file and retrain before scoring the test union.

Performance (one A100-class GPU, estimates): encoding ~9.6M distinct short texts
dominates (~15-25 min); the exact search is ~2.5e16 fp16 FLOPs (~5-10 min). Every
phase logs throughput and an ETA, so a timing run with ``--limit-*`` predicts the
full run. More devices divide both phases.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import PAIR_MULTIPLIER  # noqa: E402
from src.data_loader import (  # noqa: E402
    TARGET_SOURCES,
    ChunkWriter,
    candidates_path,
    describe_environment,
    iter_prepared,
    load_config,
)
from src.utils import (  # noqa: E402
    ID_NUMERIC_MODULUS,
    decode_entity_ids,
    encode_entity_ids,
    ensure_dir,
    fmt_int,
    read_json,
    setup_logging,
    write_json,
)

LOG_NAME = "generate_dense_candidates"
S1_COLUMN, TARGET_COLUMN, SOURCE_COLUMN, BLOCKERS_COLUMN = (
    "source1_entity_id",
    "matched_entity_id",
    "source",
    "blockers",
)
COSINE_COLUMN = "dense_cosine"
COSINE_FORMAT = "%.4f"  # the dense blocker's evidence format in generate_candidates.py
EMBEDDING_DIM = 1024
WRITE_ROWS = 2_000_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact bge-m3 cosine top-K candidates on GPU.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model-path", default=None,
                        help="local bge-m3 directory. Default: blocking.dense.model_name_or_path")
    parser.add_argument("--key", default="name_norm", help="prepared text column to embed")
    parser.add_argument("--top-k", type=int, default=10, help="target records kept per S1 (per source with "
                        "--scope per-source)")
    parser.add_argument("--scope", default="combined", choices=["combined", "per-source"],
                        help="combined: top-k over S2+S3 together. per-source: top-k from S2 AND top-k from "
                        "S3 (up to 2k per S1) - the dense blocker's own semantics in generate_candidates.py, "
                        "and better recall when an S1 has true matches in both sources")
    parser.add_argument("--min-score", type=float, default=None,
                        help="optional cosine floor applied after top-K (default: none)")
    parser.add_argument("--margin", type=int, default=8,
                        help="extra distinct texts retrieved per S1 before the exact fp32 re-score")
    parser.add_argument("--devices", default="auto",
                        help="comma list, e.g. cuda:0,cuda:1. auto = every visible GPU, else cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                        help="encoder compute dtype. auto = float16 on GPU, float32 on CPU")
    parser.add_argument("--batch-size", type=int, default=1024, help="texts per encoder batch")
    parser.add_argument("--max-length", type=int, default=64, help="token cap per text (names are short)")
    parser.add_argument("--q-tile", type=int, default=0, help="queries per score tile (0 = auto)")
    parser.add_argument("--t-sub", type=int, default=0, help="targets per score tile (0 = auto)")
    parser.add_argument("--target-block-rows", type=int, default=0,
                        help="target rows resident on a device at once (0 = auto from free memory)")
    parser.add_argument("--output-name", default="dense_candidates", help="dense TSV stem")
    parser.add_argument("--merge-with", default="auto",
                        help="string candidate TSV to union with; auto = the split's "
                        "candidate_pairs file if it exists; none = skip")
    parser.add_argument("--merged-name", default="candidate_pairs_dense_union", help="union TSV stem")
    parser.add_argument("--limit-s1", type=int, default=None, help="first N S1 rows (timing/smoke runs)")
    parser.add_argument("--limit-targets", type=int, default=None, help="first N rows per target source")
    parser.add_argument("--overwrite", action="store_true", help="ignore cached embeddings/search results")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.margin < 0 or args.batch_size < 1 or args.max_length < 2:
        parser.error("--top-k, --batch-size >= 1, --margin >= 0, --max-length >= 2")
    return args


def resolve_devices(spec: str) -> list[str]:
    import torch

    if spec and spec != "auto":
        return [d.strip() for d in spec.split(",") if d.strip()]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def compute_dtype_name(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    # fp16, not bf16: bge-m3's reference inference uses fp16, and fp16 keeps 3 more
    # mantissa bits - which matters for ranking cosines that differ in the 3rd decimal.
    return "float16" if device.startswith("cuda") else "float32"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_texts(config: dict, split: str, source: str, key: str, limit: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(entity_ids, texts)`` of one prepared table, in file order."""
    ids, texts, rows = [], [], 0
    for chunk in iter_prepared(config, split, source, columns=["entity_id", key]):
        if limit is not None and rows + len(chunk) > limit:
            chunk = chunk.iloc[: limit - rows]
        ids.append(chunk["entity_id"].to_numpy(dtype=object))
        texts.append(chunk[key].to_numpy(dtype=object))
        rows += len(chunk)
        if limit is not None and rows >= limit:
            break
    return (np.concatenate(ids) if ids else np.empty(0, dtype=object),
            np.concatenate(texts) if texts else np.empty(0, dtype=object))


def distinct_texts(texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(code_per_row, uniques)``; code -1 for an empty text (never embedded)."""
    non_empty = np.fromiter((isinstance(t, str) and len(t) > 0 for t in texts), dtype=bool, count=len(texts))
    codes = np.full(len(texts), -1, dtype=np.int64)
    if non_empty.any():
        sub_codes, uniques = pd.factorize(texts[non_empty], sort=False)
        codes[non_empty] = sub_codes
    else:
        uniques = np.empty(0, dtype=object)
    return codes, np.asarray(uniques, dtype=object)


def texts_digest(texts: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def build_postings(codes: np.ndarray, entity_codes: np.ndarray, n_texts: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR ``text -> target records``, records ordered by entity code within a text."""
    keep = codes >= 0
    order = np.lexsort((entity_codes[keep], codes[keep]))
    postings = entity_codes[keep][order]
    counts = np.bincount(codes[keep], minlength=n_texts)
    offsets = np.zeros(n_texts + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, postings


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
class BgeM3Encoder:
    """bge-m3 dense head: last hidden state of the CLS token, L2-normalized.

    That is exactly the model's sentence-transformers pipeline (Transformer ->
    CLS Pooling -> Normalize, see its ``modules.json``), driven directly through
    ``transformers`` so batching, dtype and padding are under this script's control.
    """

    def __init__(self, model_path: str, device: str, dtype_name: str, max_length: int) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        try:
            model = AutoModel.from_pretrained(model_path, local_files_only=True, attn_implementation="sdpa")
        except (ValueError, TypeError, ImportError):  # older transformers / no SDPA for this arch
            model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model = model.to(device=self.device, dtype=getattr(torch, dtype_name)).eval()

    def tokenize(self, texts: Sequence[str]) -> dict:
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        if self.device.type == "cuda":
            batch = {k: v.pin_memory() for k, v in batch.items()}
        return batch

    def encode_tokenized(self, batch: dict) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            output = self.model(
                input_ids=batch["input_ids"].to(self.device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(self.device, non_blocking=True),
            )
            cls = output.last_hidden_state[:, 0].float()
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            return cls.to(torch.float16).cpu().numpy()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_tokenized(self.tokenize(texts))


def _encode_worker(payload: dict) -> dict:
    """Encode this worker's batches straight into the shared embedding memmap."""
    log = setup_logging(f"{LOG_NAME}.w{payload['rank']}", level=payload["log_level"])
    encoder = BgeM3Encoder(payload["model_path"], payload["device"], payload["dtype"], payload["max_length"])
    out = np.load(payload["embedding_path"], mmap_mode="r+")
    batches: list[np.ndarray] = payload["batches"]  # memmap row positions, length-sorted
    batch_texts: list[list[str]] = payload["batch_texts"]  # the texts of each batch, aligned
    total = sum(len(b) for b in batches)

    # Tokenization runs on a CPU thread (the fast tokenizer releases the GIL) while
    # the device runs the previous batch; the queue bound keeps pinned memory small.
    prepared: "queue.Queue[Any]" = queue.Queue(maxsize=4)

    def producer() -> None:
        try:
            for rows, texts in zip(batches, batch_texts):
                prepared.put((rows, encoder.tokenize(texts)))
        except BaseException as error:  # surface tokenizer failures in the consumer
            prepared.put(error)
            return
        prepared.put(None)

    threading.Thread(target=producer, daemon=True).start()
    done, started, next_log = 0, time.time(), 0.0
    while True:
        item = prepared.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        rows, tokens = item
        out[rows] = encoder.encode_tokenized(tokens)
        done += len(rows)
        elapsed = time.time() - started
        if elapsed >= next_log or done == total:
            rate = done / max(elapsed, 1e-9)
            log.info("  [%s] encoded %s/%s texts | %.0f texts/s | ETA %.1f min",
                     payload["device"], fmt_int(done), fmt_int(total), rate, (total - done) / max(rate, 1e-9) / 60)
            next_log = elapsed + 30
    out.flush()
    del out
    return {"rank": payload["rank"], "rows": done, "seconds": time.time() - started}


def encode_all(
    texts: np.ndarray, path: Path, meta: dict, devices: list[str], args: argparse.Namespace, log: logging.Logger
) -> np.ndarray:
    """Embeddings for ``texts`` (float16, row-aligned), cached at ``path``."""
    meta_path = path.with_suffix(".meta.json")
    if path.is_file() and meta_path.is_file() and not args.overwrite and read_json(meta_path) == meta:
        log.info("reusing cached embeddings %s (%s texts)", path.name, fmt_int(len(texts)))
        return np.load(path, mmap_mode="r")

    n = len(texts)
    partial = path.with_name(path.name + ".partial")
    np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(n, EMBEDDING_DIM)).flush()
    if n == 0:
        partial.replace(path)
        write_json(meta_path, meta)
        return np.load(path, mmap_mode="r")

    # Length-sorted batches: a batch of short names pads to a short length. Batches
    # are dealt round-robin, so every device gets the same mix of lengths.
    lengths = np.fromiter((len(t) for t in texts), dtype=np.int64, count=n)
    order = np.argsort(lengths, kind="stable")
    batches = [order[i : i + args.batch_size] for i in range(0, n, args.batch_size)]
    payloads = []
    for rank, device in enumerate(devices):
        mine = batches[rank :: len(devices)]
        payloads.append(
            {
                "rank": rank,
                "device": device,
                "dtype": compute_dtype_name(device, args.dtype),
                "model_path": args.model_path,
                "max_length": args.max_length,
                "embedding_path": str(partial),
                "batches": mine,
                "batch_texts": [texts[rows].tolist() for rows in mine],
                "log_level": args.log_level,
            }
        )
    log.info("encoding %s texts on %s (batch %s, max_length %s)",
             fmt_int(n), ", ".join(devices), args.batch_size, args.max_length)
    started = time.time()
    results = _run_workers(_encode_worker, payloads)
    log.info("encoded %s texts in %.1f min (%.0f texts/s over %d device(s))",
             fmt_int(n), (time.time() - started) / 60, n / max(time.time() - started, 1e-9), len(devices))
    for result in results:
        if result["rows"] != sum(len(b) for b in payloads[result["rank"]]["batches"]):
            raise RuntimeError(f"worker {result['rank']} encoded {result['rows']} rows, expected more")
    partial.replace(path)
    write_json(meta_path, meta)
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------------------
# Exact search
# ---------------------------------------------------------------------------
def _search_worker(payload: dict) -> dict:
    """Exact top-``k`` inner product of a query row range against every target."""
    import warnings

    import torch

    # The memmapped embeddings are read-only and only ever read; torch's warning about
    # wrapping a non-writable array is noise here (copying 16 GB blocks to silence it
    # would not be).
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
    log = setup_logging(f"{LOG_NAME}.s{payload['rank']}", level=payload["log_level"])
    device = torch.device(payload["device"])
    on_gpu = device.type == "cuda"
    if on_gpu:
        torch.cuda.set_device(device)
    work = torch.float16 if on_gpu else torch.float32  # CPU has no fast fp16 GEMM
    q0, q1, k = payload["q0"], payload["q1"], payload["k"]
    queries = np.load(payload["query_path"], mmap_mode="r")
    targets = np.load(payload["target_path"], mmap_mode="r")
    n_targets = len(targets)
    q_tile = payload["q_tile"] or (4096 if on_gpu else 512)
    t_sub = payload["t_sub"] or (65_536 if on_gpu else 8192)

    Q = torch.from_numpy(np.ascontiguousarray(queries[q0:q1])).to(device=device, dtype=work)
    nq = len(Q)
    best_scores = torch.full((nq, k), float("-inf"), dtype=torch.float32, device=device)
    best_index = torch.full((nq, k), -1, dtype=torch.int64, device=device)

    block_rows = payload["target_block_rows"]
    if not block_rows:
        if on_gpu:
            free, _ = torch.cuda.mem_get_info(device)
            # Leave room for one score tile, the fp32 re-score gather and the merge.
            tile_bytes = q_tile * t_sub * 2 + q_tile * (k + 1) * EMBEDDING_DIM * 6 + (64 << 20)
            block_rows = max(t_sub, int((free * 0.8 - tile_bytes) // (EMBEDDING_DIM * 2)))
        else:
            block_rows = 1 << 20
    block_rows = min(max(block_rows, 1), max(n_targets, 1))

    started, next_log = time.time(), 0.0
    for t0 in range(0, n_targets, block_rows):
        t1 = min(t0 + block_rows, n_targets)
        T = torch.from_numpy(np.ascontiguousarray(targets[t0:t1])).to(device=device, dtype=work)
        for s0 in range(0, len(T), t_sub):
            Ts = T[s0 : s0 + t_sub]
            kk = min(k, len(Ts))
            for a in range(0, nq, q_tile):
                q = Q[a : a + q_tile]
                # Select in the working dtype (a margin of extra candidates absorbs fp16
                # rounding at the cut), then re-score the survivors exactly in fp32.
                _, local = (q @ Ts.T).topk(kk, dim=1)
                exact = torch.einsum("qd,qkd->qk", q.float(), Ts[local].float())
                scores = torch.cat([best_scores[a : a + q_tile], exact], dim=1)
                index = torch.cat([best_index[a : a + q_tile], local + (t0 + s0)], dim=1)
                top, position = scores.topk(k, dim=1)
                best_scores[a : a + q_tile] = top
                best_index[a : a + q_tile] = index.gather(1, position)
            elapsed = time.time() - started
            done = t0 + s0 + len(Ts)
            if elapsed >= next_log or done == n_targets:
                rate = done / max(elapsed, 1e-9)
                log.info("  [%s] queries %s-%s: %s/%s targets searched | ETA %.1f min",
                         payload["device"], fmt_int(q0), fmt_int(q1), fmt_int(done), fmt_int(n_targets),
                         (n_targets - done) / max(rate, 1e-9) / 60)
                next_log = elapsed + 30
        del T

    out_index = np.load(payload["index_path"], mmap_mode="r+")
    out_scores = np.load(payload["score_path"], mmap_mode="r+")
    out_index[q0:q1] = best_index.cpu().numpy()
    out_scores[q0:q1] = best_scores.cpu().numpy()
    out_index.flush()
    out_scores.flush()
    return {"rank": payload["rank"], "queries": nq, "seconds": time.time() - started}


def search_all(
    query_path: Path, target_path: Path, n_queries: int, k: int, out_dir: Path, meta: dict,
    devices: list[str], args: argparse.Namespace, log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """``(index, scores)``, each ``[n_queries, k]``: exact top-k target rows per query."""
    index_path, score_path = out_dir / "search_index.npy", out_dir / "search_scores.npy"
    meta_path = out_dir / "search.meta.json"
    if index_path.is_file() and score_path.is_file() and meta_path.is_file() and not args.overwrite \
            and read_json(meta_path) == meta:
        log.info("reusing cached search results (%s queries, k=%s)", fmt_int(n_queries), k)
        return np.load(index_path), np.load(score_path)

    for path in (index_path, score_path):
        path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    np.lib.format.open_memmap(index_path, mode="w+", dtype=np.int64, shape=(n_queries, k)).flush()
    np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(n_queries, k)).flush()
    bounds = np.linspace(0, n_queries, len(devices) + 1).astype(np.int64)
    payloads = [
        {
            "rank": rank,
            "device": device,
            "q0": int(bounds[rank]),
            "q1": int(bounds[rank + 1]),
            "k": k,
            "query_path": str(query_path),
            "target_path": str(target_path),
            "index_path": str(index_path),
            "score_path": str(score_path),
            "q_tile": args.q_tile,
            "t_sub": args.t_sub,
            "target_block_rows": args.target_block_rows,
            "log_level": args.log_level,
        }
        for rank, device in enumerate(devices)
        if bounds[rank + 1] > bounds[rank]
    ]
    started = time.time()
    if payloads:
        _run_workers(_search_worker, payloads)
    log.info("exact search: %s queries x %s targets in %.1f min",
             fmt_int(n_queries), fmt_int(len(np.load(target_path, mmap_mode="r"))), (time.time() - started) / 60)
    write_json(meta_path, meta)
    return np.load(index_path), np.load(score_path)


def _run_workers(function, payloads: list[dict]) -> list[dict]:
    """One process per device; in-process when there is only one (easier to debug)."""
    if len(payloads) == 1:
        return [function(payloads[0])]
    import multiprocessing

    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(function, payload) for payload in payloads]
        return [future.result() for future in futures]


# ---------------------------------------------------------------------------
# Expansion to records
# ---------------------------------------------------------------------------
def expand_to_records(
    s1_codes: np.ndarray, index: np.ndarray, scores: np.ndarray,
    offsets: np.ndarray, postings: np.ndarray, top_k: int, min_score: Optional[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(s1_row, target_entity_code, cosine)``, sorted by (s1_row, entity code).

    Per distinct S1 text the target texts are taken best-first and expanded to their
    records until ``top_k`` records are reached (records of one text in entity-code
    order); every S1 row then receives its text's records.
    """
    n_q, k = index.shape
    valid = index >= 0
    if min_score is not None:
        valid &= scores >= np.float32(min_score)
    safe = np.where(valid, index, 0)
    counts = np.where(valid, np.minimum(offsets[safe + 1] - offsets[safe], top_k), 0)
    before = np.cumsum(counts, axis=1) - counts
    take = np.clip(top_k - before, 0, counts)  # records taken from each ranked text

    flat_take = take.ravel()
    per_query = take.sum(axis=1)
    text_of = np.repeat(safe.ravel(), flat_take)
    cosine_of = np.repeat(scores.ravel(), flat_take)
    starts = np.repeat(np.cumsum(flat_take) - flat_take, flat_take)
    within = np.arange(int(flat_take.sum()), dtype=np.int64) - starts
    code_of = postings[offsets[text_of] + within]
    # Rows are grouped by query (row-major ravel), best text first within a query.

    # Rows per distinct query -> rows per S1 record.
    query_start = np.zeros(n_q + 1, dtype=np.int64)
    np.cumsum(per_query, out=query_start[1:])
    rows = np.flatnonzero(s1_codes >= 0)
    rows = rows[per_query[s1_codes[rows]] > 0]
    n_rows = per_query[s1_codes[rows]]
    record_s1 = np.repeat(rows, n_rows)
    record_start = np.repeat(query_start[s1_codes[rows]], n_rows)
    record_within = np.arange(int(n_rows.sum()), dtype=np.int64) - np.repeat(np.cumsum(n_rows) - n_rows, n_rows)
    source_row = record_start + record_within
    record_code, record_cosine = code_of[source_row], cosine_of[source_row]
    order = np.lexsort((record_code, record_s1))
    return record_s1[order], record_code[order], record_cosine[order]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def dense_frame(s1_ids: np.ndarray, s1_rows: np.ndarray, codes: np.ndarray, cosine: np.ndarray,
                extra_blank: Sequence[str] = ()) -> pd.DataFrame:
    source_numbers = codes // ID_NUMERIC_MODULUS
    frame = pd.DataFrame(
        {
            S1_COLUMN: s1_ids[s1_rows],
            TARGET_COLUMN: decode_entity_ids(codes),
            SOURCE_COLUMN: np.where(source_numbers == 2, "S2", "S3").astype(object),
            BLOCKERS_COLUMN: np.where(source_numbers == 2, "source2:dense", "source3:dense").astype(object),
        }
    )
    for column in extra_blank:
        frame[column] = ""
    frame[COSINE_COLUMN] = np.char.mod(COSINE_FORMAT, cosine.astype(np.float64))
    return frame


def write_dense(path: Path, s1_ids, s1_rows, codes, cosine) -> int:
    rows = 0
    partial = path.with_name(path.name + ".partial")
    with ChunkWriter(partial) as writer:
        for a in range(0, len(codes), WRITE_ROWS):
            rows += writer.append(dense_frame(s1_ids, s1_rows[a : a + WRITE_ROWS], codes[a : a + WRITE_ROWS],
                                              cosine[a : a + WRITE_ROWS]))
        if rows == 0:
            writer.append_header([S1_COLUMN, TARGET_COLUMN, SOURCE_COLUMN, BLOCKERS_COLUMN, COSINE_COLUMN])
    partial.replace(path)
    return rows


def merge_union(
    string_path: Path, out_path: Path, s1_ids: np.ndarray,
    dense_s1: np.ndarray, dense_code: np.ndarray, dense_cosine: np.ndarray,
    chunksize: int, log: logging.Logger,
) -> dict:
    """Stream the string candidates and union the dense rows in, S1 by S1.

    Both inputs are in S1 file order (the string generator walks the prepared S1
    table; the dense rows are sorted by S1 row), so one pass suffices: an S1 group
    of the string file is joined with its dense rows, and S1 entities that only
    dense proposed are emitted between groups. Output rows are sorted by
    (S1 row, entity code) - the order ``union_blockers`` writes.
    """
    from scripts.extract_pair_features import iter_entity_batches

    with open(string_path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    for column in (S1_COLUMN, TARGET_COLUMN, SOURCE_COLUMN, BLOCKERS_COLUMN):
        if column not in header:
            raise ValueError(f"{string_path} lacks column {column!r}")
    if COSINE_COLUMN in header:
        raise ValueError(f"{string_path} already carries {COSINE_COLUMN!r}; refusing to merge dense twice")
    evidence = [c for c in header if c not in (S1_COLUMN, TARGET_COLUMN, SOURCE_COLUMN, BLOCKERS_COLUMN)]
    columns = [*header, COSINE_COLUMN]

    s1_index = pd.Index(s1_ids)
    dense_key = dense_s1 * PAIR_MULTIPLIER + dense_code  # sorted: dense rows are sorted by (s1, code)
    stats = {"string_rows": 0, "dense_rows": int(len(dense_key)), "overlap": 0, "dense_only": 0, "rows_out": 0}
    next_s1 = 0
    partial = out_path.with_name(out_path.name + ".partial")

    def dense_slice(lo_s1: int, hi_s1: int) -> tuple[int, int]:
        return (int(np.searchsorted(dense_s1, lo_s1, side="left")),
                int(np.searchsorted(dense_s1, hi_s1, side="left")))

    with ChunkWriter(partial) as writer:

        def emit_dense_only(lo_s1: int, hi_s1: int) -> None:
            a, b = dense_slice(lo_s1, hi_s1)
            for start in range(a, b, WRITE_ROWS):
                stop = min(start + WRITE_ROWS, b)
                frame = dense_frame(s1_ids, dense_s1[start:stop], dense_code[start:stop],
                                    dense_cosine[start:stop], extra_blank=evidence)
                stats["dense_only"] += len(frame)
                stats["rows_out"] += writer.append(frame[columns])

        for batch in iter_entity_batches(string_path, chunksize):
            batch = batch.copy()
            positions = s1_index.get_indexer(batch[S1_COLUMN])
            if (positions < 0).any():
                raise ValueError(f"{string_path} names S1 ids absent from the prepared S1 table, "
                                 f"e.g. {batch[S1_COLUMN][positions < 0].head(3).tolist()}")
            if positions[0] < next_s1 or (np.diff(positions) < 0).any():
                raise ValueError(f"{string_path} is not in S1 file order; cannot stream-merge")
            lo, hi = int(positions[0]), int(positions[-1]) + 1
            emit_dense_only(next_s1, lo)
            stats["string_rows"] += len(batch)

            a, b = dense_slice(lo, hi)
            codes = encode_entity_ids(batch[TARGET_COLUMN])
            keys = positions * PAIR_MULTIPLIER + codes
            window = dense_key[a:b]
            if len(window):
                where_clipped = np.minimum(np.searchsorted(window, keys), len(window) - 1)
                hit = window[where_clipped] == keys
            else:
                where_clipped = np.zeros(len(keys), dtype=np.int64)
                hit = np.zeros(len(keys), dtype=bool)
            cosine = np.full(len(batch), "", dtype=object)
            if hit.any():
                cosine[hit] = np.char.mod(COSINE_FORMAT, dense_cosine[a:b][where_clipped[hit]].astype(np.float64))
                labels = np.where(codes[hit] // ID_NUMERIC_MODULUS == 2, ",source2:dense", ",source3:dense")
                blockers = batch[BLOCKERS_COLUMN].to_numpy(dtype=object).copy()
                blockers[hit] = blockers[hit] + labels
                batch[BLOCKERS_COLUMN] = blockers
            batch[COSINE_COLUMN] = cosine
            stats["overlap"] += int(hit.sum())

            used = np.zeros(b - a, dtype=bool)
            used[where_clipped[hit]] = True
            extra = np.flatnonzero(~used) + a
            parts = [batch[columns]]
            sort_s1, sort_code = [positions], [codes]
            if extra.size:
                parts.append(dense_frame(s1_ids, dense_s1[extra], dense_code[extra], dense_cosine[extra],
                                         extra_blank=evidence)[columns])
                sort_s1.append(dense_s1[extra])
                sort_code.append(dense_code[extra])
                stats["dense_only"] += int(extra.size)
            merged = pd.concat(parts, ignore_index=True)
            order = np.lexsort((np.concatenate(sort_code), np.concatenate(sort_s1)))
            stats["rows_out"] += writer.append(merged.iloc[order])
            next_s1 = hi
        emit_dense_only(next_s1, len(s1_ids))
        if stats["rows_out"] == 0:
            writer.append_header(columns)
    partial.replace(out_path)
    log.info("union: %s string rows + %s dense rows -> %s rows (%s overlap, %s dense-only)",
             fmt_int(stats["string_rows"]), fmt_int(stats["dense_rows"]), fmt_int(stats["rows_out"]),
             fmt_int(stats["overlap"]), fmt_int(stats["dense_only"]))
    return stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir})
    log = setup_logging(LOG_NAME, log_dir=config["resolved"]["log_dir"],
                        level=getattr(logging, args.log_level.upper(), logging.INFO))
    args.model_path = args.model_path or (config.get("blocking", {}).get("dense", {}) or {}).get("model_name_or_path")
    if not args.model_path or not Path(args.model_path).is_dir():
        log.error("--model-path must be a local bge-m3 directory (got %r); fetch it with "
                  "scripts/fetch_dense_model.py on a node with internet", args.model_path)
        return 2
    devices = resolve_devices(args.devices)
    cache = ensure_dir(Path(config["resolved"]["work_dir"]) / "dense" / args.split / args.scope)
    limited = args.limit_s1 is not None or args.limit_targets is not None
    if limited:
        cache = ensure_dir(cache / f"limited_s1{args.limit_s1}_t{args.limit_targets}")

    log.info("=" * 78)
    log.info("generate_dense_candidates: exact bge-m3 cosine top-%s per S1 (split=%s)", args.top_k, args.split)
    log.info(describe_environment(config))
    log.info("model=%s devices=%s cache=%s", args.model_path, ",".join(devices), cache)
    if any(d.startswith("cuda") for d in devices):
        import torch

        for device in devices:
            log.info("  %s: %s", device, torch.cuda.get_device_name(torch.device(device)))
    log.info("=" * 78)
    timings: dict[str, float] = {}
    t = time.time()

    # -- inputs ---------------------------------------------------------------------
    t = time.time()
    s1_ids, s1_texts = load_texts(config, args.split, "source1", args.key, args.limit_s1)
    s1_codes, s1_unique = distinct_texts(s1_texts)
    groups = [("targets", list(TARGET_SOURCES))] if args.scope == "combined" else         [(source, [source]) for source in TARGET_SOURCES]
    timings["load_seconds"] = time.time() - t
    log.info("S1: %s rows, %s distinct texts | scope=%s", fmt_int(len(s1_ids)), fmt_int(len(s1_unique)), args.scope)
    if len(s1_unique) == 0:
        log.error("no non-empty S1 texts to search")
        return 2

    base_meta = {"model": str(Path(args.model_path).resolve()), "max_length": args.max_length,
                 "dtype": [compute_dtype_name(d, args.dtype) for d in devices][0], "key": args.key}
    t = time.time()
    s1_path = cache / "s1_embeddings.npy"
    s1_meta = {**base_meta, "n": len(s1_unique), "texts": texts_digest(s1_unique)}
    encode_all(s1_unique, s1_path, s1_meta, devices, args, log)
    timings["encode_seconds"] = time.time() - t
    timings["search_seconds"] = 0.0

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    group_stats = {}
    for name, sources in groups:
        # -- one target group: load, encode, exact search, expand to records --------
        t = time.time()
        loaded = [load_texts(config, args.split, source, args.key, args.limit_targets) for source in sources]
        target_ids = np.concatenate([ids for ids, _ in loaded])
        target_codes, target_unique = distinct_texts(np.concatenate([texts for _, texts in loaded]))
        offsets, postings = build_postings(target_codes, encode_entity_ids(pd.Series(target_ids, dtype=object)),
                                           len(target_unique))
        timings["load_seconds"] += time.time() - t
        log.info("[%s] %s rows, %s distinct texts", name, fmt_int(len(target_ids)), fmt_int(len(target_unique)))
        group_stats[name] = {"rows": int(len(target_ids)), "distinct_texts": int(len(target_unique))}
        if len(target_unique) == 0:
            log.warning("[%s] no non-empty target texts; skipped", name)
            continue

        t = time.time()
        group_dir = ensure_dir(cache / name)
        target_path = group_dir / "target_embeddings.npy"
        target_meta = {**base_meta, "n": len(target_unique), "texts": texts_digest(target_unique)}
        encode_all(target_unique, target_path, target_meta, devices, args, log)
        timings["encode_seconds"] += time.time() - t

        t = time.time()
        k = min(args.top_k + args.margin, len(target_unique))
        index, scores = search_all(
            s1_path, target_path, len(s1_unique), k, group_dir,
            {"s1": s1_meta, "targets": target_meta, "k": k}, devices, args, log,
        )
        timings["search_seconds"] += time.time() - t
        parts.append(expand_to_records(s1_codes, index, scores, offsets, postings, args.top_k, args.min_score))

    # -- combine groups + write ------------------------------------------------------
    t = time.time()
    if parts:
        dense_s1 = np.concatenate([p[0] for p in parts])
        dense_code = np.concatenate([p[1] for p in parts])
        dense_cosine = np.concatenate([p[2] for p in parts])
        order = np.lexsort((dense_code, dense_s1))  # (S1 row, entity code): the union's order
        dense_s1, dense_code, dense_cosine = dense_s1[order], dense_code[order], dense_cosine[order]
    else:
        dense_s1 = dense_code = np.empty(0, dtype=np.int64)
        dense_cosine = np.empty(0, dtype=np.float32)
    dense_path = candidates_path(config, args.output_name, split=args.split, legacy_fallback=False)
    dense_rows = write_dense(dense_path, s1_ids, dense_s1, dense_code, dense_cosine)
    timings["write_seconds"] = time.time() - t
    log.info("dense candidates: %s rows for %s S1 entities -> %s",
             fmt_int(dense_rows), fmt_int(len(np.unique(dense_s1))), dense_path)

    # -- union with the string candidates --------------------------------------------
    union_stats = None
    string_path = None
    if args.merge_with != "none":
        string_path = (candidates_path(config, "candidate_pairs", split=args.split)
                       if args.merge_with == "auto" else Path(args.merge_with))
        if string_path.is_file():
            if limited:
                log.warning("merging a limited (--limit-*) dense run: the union covers only the "
                            "limited S1/target subset's dense rows")
            t = time.time()
            union_path = candidates_path(config, args.merged_name, split=args.split, legacy_fallback=False)
            union_stats = merge_union(string_path, union_path, s1_ids, dense_s1, dense_code, dense_cosine,
                                      int(config.get("io", {}).get("chunksize", 500_000)), log)
            union_stats["path"] = str(union_path)
            timings["merge_seconds"] = time.time() - t
        elif args.merge_with != "auto":
            log.error("--merge-with %s does not exist", string_path)
            return 2
        else:
            log.info("no string candidate file at %s; skipping the union", string_path)

    cosine_summary = (
        {f"p{q}": float(np.percentile(dense_cosine, q)) for q in (1, 10, 50, 90, 99)} if len(dense_cosine) else {}
    )
    report = {
        "split": args.split,
        "model": args.model_path,
        "devices": devices,
        "top_k": args.top_k,
        "scope": args.scope,
        "min_score": args.min_score,
        "limited": limited,
        "s1_rows": int(len(s1_ids)),
        "s1_distinct_texts": int(len(s1_unique)),
        "target_groups": group_stats,
        "dense_rows": int(dense_rows),
        "dense_path": str(dense_path),
        "dense_cosine": cosine_summary,
        "union": union_stats,
        "string_candidates": str(string_path) if string_path else None,
        "timings_seconds": {k2: round(v, 1) for k2, v in timings.items()},
        "total_minutes": round(sum(timings.values()) / 60, 2),
    }
    write_json(dense_path.with_name(dense_path.stem + "_report.json"), report)
    log.info("done in %.1f min: %s", report["total_minutes"], report["timings_seconds"])
    if union_stats:
        log.info("next: python scripts/extract_pair_features.py --split %s --candidates %s --sample-fraction 1.0",
                 args.split, args.merged_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
