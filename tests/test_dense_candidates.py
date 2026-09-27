"""scripts/generate_dense_candidates.py: exact search, record expansion, the union, the encoder.

The GPU is not needed: the search runs the same tiled code on CPU (float32), with
tiles and blocks far smaller than the data so every merge path is exercised, and is
checked against a brute-force NumPy top-k. The multi-device path is exercised with
two CPU worker processes. The encoder test compares this script's CLS+normalize
encoder with sentence-transformers' own bge-m3 pipeline and needs the local model
(``ER_DENSE_MODEL``); it is skipped otherwise.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import generate_dense_candidates as gdc  # noqa: E402
from src.data_loader import read_tsv  # noqa: E402
from src.utils import encode_entity_id  # noqa: E402

LOG = logging.getLogger("test_dense_candidates")
LOG.setLevel(logging.CRITICAL)


def _args(**overrides) -> argparse.Namespace:
    base = {"q_tile": 7, "t_sub": 11, "target_block_rows": 23, "overwrite": True, "log_level": "CRITICAL"}
    return argparse.Namespace(**{**base, **overrides})


def _unit(rng, n, d=gdc.EMBEDDING_DIM) -> np.ndarray:
    x = rng.standard_normal((n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


@pytest.mark.parametrize("devices", [["cpu"], ["cpu", "cpu"]])
def test_tiled_search_is_exact_against_brute_force(tmp_path, devices):
    rng = np.random.default_rng(7)
    queries, targets = _unit(rng, 40), _unit(rng, 97)
    np.save(tmp_path / "q.npy", queries)
    np.save(tmp_path / "t.npy", targets)
    k = 6
    index, scores = gdc.search_all(tmp_path / "q.npy", tmp_path / "t.npy", len(queries), k, tmp_path,
                                   {"test": 1}, devices, _args(), LOG)

    reference = queries.astype(np.float32) @ targets.astype(np.float32).T
    expected = np.argsort(-reference, axis=1, kind="stable")[:, :k]
    assert np.array_equal(index, expected), "top-k differs from brute force"
    np.testing.assert_allclose(scores, np.take_along_axis(reference, expected, axis=1), atol=1e-5)


def test_search_with_fewer_targets_than_k_leaves_missing_slots_empty(tmp_path):
    rng = np.random.default_rng(1)
    np.save(tmp_path / "q.npy", _unit(rng, 3))
    np.save(tmp_path / "t.npy", _unit(rng, 2))
    index, scores = gdc.search_all(tmp_path / "q.npy", tmp_path / "t.npy", 3, 2, tmp_path, {}, ["cpu"], _args(), LOG)
    assert sorted(index[0].tolist()) == [0, 1]
    assert np.all(np.isfinite(scores))


def test_expansion_caps_records_orders_by_entity_code_and_applies_min_score():
    # distinct target texts: t0 has 3 records, t1 has 1, t2 has 2
    target_codes = np.array([0, 1, 0, 2, 0, 2])
    entity = np.array([encode_entity_id(f"S2-{n}") for n in (30, 40, 10, 50, 20, 5)])
    offsets, postings = gdc.build_postings(target_codes, entity, 3)
    assert postings[offsets[0]:offsets[1]].tolist() == sorted(entity[target_codes == 0].tolist())

    s1_codes = np.array([0, -1, 1, 0])  # rows 0 and 3 share a text; row 1 has an empty name
    index = np.array([[0, 2, 1], [2, 1, -1]])
    scores = np.array([[0.9, 0.8, 0.7], [0.95, 0.4, -np.inf]], dtype=np.float32)
    rows, codes, cosine = gdc.expand_to_records(s1_codes, index, scores, offsets, postings, top_k=4, min_score=None)
    decoded = [(int(r), f"S2-{c % 10**10}") for r, c in zip(rows, codes)]
    # row 0: t0's 3 records then 1 of t2 (cap 4); sorted by entity code within the S1
    assert decoded[:4] == [(0, "S2-5"), (0, "S2-10"), (0, "S2-20"), (0, "S2-30")]
    assert decoded[4:7] == [(2, "S2-5"), (2, "S2-40"), (2, "S2-50")]
    assert [c for r, c in decoded if r == 3] == [c for r, c in decoded if r == 0], (
        "an S1 sharing a text gets the same candidates"
    )
    assert 1 not in rows, "an empty name has no candidates"
    assert cosine[decoded.index((0, "S2-5"))] == pytest.approx(0.8)

    rows, _, cosine = gdc.expand_to_records(s1_codes, index, scores, offsets, postings, top_k=4, min_score=0.85)
    assert set(rows.tolist()) == {0, 2, 3} and cosine.min() >= 0.85


def test_union_combines_provenance_and_interleaves_dense_only_entities(tmp_path):
    string = tmp_path / "string.tsv"
    string.write_text(
        "source1_entity_id\tmatched_entity_id\tsource\tblockers\ttoken_df\tchar_jaccard\n"
        "S1-2\tS2-7\tS2\tsource2:exact_name,source2:char_ngram\t\t1.0000\n"
        "S1-2\tS3-1\tS3\tsource3:token\t4\t\n"
        "S1-4\tS2-9\tS2\tsource2:token\t2\t\n",
        encoding="utf-8",
    )
    s1_ids = np.array([f"S1-{i}" for i in range(1, 6)], dtype=object)
    code = encode_entity_id
    dense = sorted(
        [(0, code("S2-3"), 0.91), (1, code("S2-7"), 0.99), (1, code("S2-8"), 0.81), (2, code("S3-4"), 0.77),
         (4, code("S2-1"), 0.70)]
    )
    out = tmp_path / "union.tsv"
    stats = gdc.merge_union(string, out, s1_ids, np.array([d[0] for d in dense]), np.array([d[1] for d in dense]),
                            np.array([d[2] for d in dense], dtype=np.float32), chunksize=2, log=LOG)
    table = read_tsv(out)
    assert list(table.columns) == ["source1_entity_id", "matched_entity_id", "source", "blockers",
                                   "token_df", "char_jaccard", "dense_cosine"]
    assert table.values.tolist() == [
        ["S1-1", "S2-3", "S2", "source2:dense", "", "", "0.9100"],
        ["S1-2", "S2-7", "S2", "source2:exact_name,source2:char_ngram,source2:dense", "", "1.0000", "0.9900"],
        ["S1-2", "S2-8", "S2", "source2:dense", "", "", "0.8100"],
        ["S1-2", "S3-1", "S3", "source3:token", "4", "", ""],
        ["S1-3", "S3-4", "S3", "source3:dense", "", "", "0.7700"],
        ["S1-4", "S2-9", "S2", "source2:token", "2", "", ""],
        ["S1-5", "S2-1", "S2", "source2:dense", "", "", "0.7000"],
    ]
    assert stats == {"string_rows": 3, "dense_rows": 5, "overlap": 1, "dense_only": 4, "rows_out": 7}

    with pytest.raises(ValueError, match="already carries"):
        gdc.merge_union(out, tmp_path / "again.tsv", s1_ids, np.array([0]), np.array([code("S2-3")]),
                        np.array([0.5], dtype=np.float32), chunksize=2, log=LOG)


@pytest.mark.skipif(not os.environ.get("ER_DENSE_MODEL"), reason="needs ER_DENSE_MODEL (local bge-m3)")
def test_encoder_matches_the_sentence_transformers_bge_m3_pipeline():
    from sentence_transformers import SentenceTransformer

    model = os.environ["ER_DENSE_MODEL"]
    texts = ["ram marketing private limited", "राम मार्केटिंग प्राइवेट लिमिटेड", "ಕೃಷ್ಣ ಹೋಟೆಲ್",
             "acme holdings llc", 'joe s "famous" pizza', "a"]
    ours = gdc.BgeM3Encoder(model, "cpu", "float32", 64).encode(texts).astype(np.float32)
    reference = SentenceTransformer(model, device="cpu", local_files_only=True)
    reference.max_seq_length = 64
    theirs = reference.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    cosine = (ours * theirs).sum(axis=1)
    assert cosine.min() > 0.9995, cosine  # equal up to float16 storage
    np.testing.assert_allclose(np.linalg.norm(ours, axis=1), 1.0, atol=2e-3)
