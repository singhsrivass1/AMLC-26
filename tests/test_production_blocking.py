"""Fixture tests for the production blocking interface (``src/blocking.py``).

The provisional production blocker is a **union** of three candidate generators::

    exact(name_norm)  UNION  token(name_norm, df<=1000, rarest 1)
                             UNION  char(name_key, df<=1000, rarest 5, J>=0.3)

Nothing here touches the dataset. Every fixture is a handful of hand-written rows
built in memory, chosen so that each blocker has a case only it can retrieve:

* token-only - ``acme zeta`` / ``acme omega`` share the eligible token ``acme`` but
  their trigram Jaccard is 0.18, below the 0.3 cut-off, and their ``name_norm``
  differs, so exact and char both decline;
* char-only - ``quickmart`` / ``quick mart`` have an identical ``name_key`` (so
  Jaccard 1.0) but different ``name_norm`` and no shared token, so exact and token
  both decline;
* exact - identical ``name_norm``, which necessarily also satisfies both other
  blockers, so it is the provenance that is asserted, not the retrieval;
* eligibility-before-ranking - an S1 whose rarest token is absent from the target
  corpus entirely. Ranking before eligibility would spend the entity's single
  ``rarest_k`` slot on a key that cannot be looked up and silently return nothing;
* the df cap and the rarest-K rule - checked against a plain-python reference of
  the rule as *specified*, not against another implementation of it;
* the codecs - cross-checked against the calibration scripts, which are the source
  of truth for the semantics (``scripts/calibrate_{token,char}_blocker.py``), and
  the token and char indexes are cross-checked structurally against the calibration
  index at the same ``(df_cap, rarest_k)`` cell.

Runs standalone (``python tests/test_production_blocking.py``) and under pytest.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.blocking import (  # noqa: E402
    BLOCKER_CHAR_NGRAM,
    BLOCKER_DENSE,
    BLOCKER_EXACT_NAME,
    BLOCKER_TOKEN,
    INDEX_BUILDERS,
    INDEX_LOADERS,
    LEXICAL_BLOCKERS,
    UNION_BLOCKERS,
    CharNgramIndex,
    ExactNameIndex,
    TokenIndex,
    build_index,
    decode_trigram_code,
    evidence_columns_for,
    index_dir_for,
    load_index,
    pack_pairs,
    resolve_blocker_settings,
    tokenize,
    trigram_codes,
    union_blockers,
)
from src.data_loader import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    SOURCE_PREFIX,
    load_config,
    prepared_path,
    read_tsv,
)
from src.normalization import strip_separators  # noqa: E402
from src.utils import encode_entity_id, encode_entity_ids  # noqa: E402

# The fixtures are deliberately tiny; the loggers below stay quiet so the only
# output is the pass/fail line per test.
LOG = logging.getLogger("test_production_blocking")
LOG.addHandler(logging.NullHandler())
LOG.setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
_SCRATCH = Path(tempfile.mkdtemp(prefix="test_production_blocking_"))
atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)


def key_of(name: str) -> str:
    """``name_key`` for an already-normalized name, via the real normalizer."""
    return str(strip_separators(pd.Series([name], dtype="object")).iloc[0])


def frame(source: str, rows: list[tuple[int, str]]) -> pd.DataFrame:
    """A prepared-table-shaped chunk: ``entity_id``, ``name_norm``, ``name_key``.

    ``name_key`` is derived with the real normalizer rather than written by hand, so
    a fixture cannot accidentally assert on a ``name_key`` the pipeline would never
    produce.
    """
    prefix = SOURCE_PREFIX[source]
    names = [name for _, name in rows]
    return pd.DataFrame(
        {
            "entity_id": [f"{prefix}-{numeric}" for numeric, _ in rows],
            "name_norm": names,
            "name_key": strip_separators(pd.Series(names, dtype="object")).astype(object),
        }
    )


def build(index_class, frames: list[pd.DataFrame], source: str, **settings):
    """Build an index from in-memory frames.

    ``ExactNameIndex`` needs one pass and takes an iterator; the multi-key blockers
    read the table twice and take a chunk *factory*, so the same frames can be
    streamed again for the document-frequency pass.
    """
    chunks = iter(frames) if index_class is ExactNameIndex else (lambda: iter(frames))
    return index_class.build(
        chunks,
        source=source,
        prefix=SOURCE_PREFIX[source],
        log=LOG,
        **settings,
    )


def kept_pairs(index) -> set[tuple[int, int]]:
    """The index's ``(key_code, entity_code)`` pairs, read out of its CSR arrays."""
    out = set()
    for position in range(index.n_keys):
        key = int(index.keys[position])
        for posting in index.postings[index.postings_offsets[position] : index.postings_offsets[position + 1]]:
            out.add((key, int(posting)))
    return out


def reference_rarest_keep(codes, owners, dfs, df_cap: int, rarest_k: int) -> set[tuple[int, int]]:
    """The rarest-K eligible-key rule, written out in plain python.

    Eligibility first (``0 < df <= df_cap``), then order by ``(df, code)`` and keep
    the first ``rarest_k``. This is the specification; if the vectorized
    ``_rarest_keep_mask`` disagrees with it, the vectorized one is wrong.
    """
    per_owner: dict[int, list[tuple[int, int]]] = {}
    for code, owner, df in zip(codes, owners, dfs):
        if 0 < int(df) <= df_cap:
            per_owner.setdefault(int(owner), []).append((int(df), int(code)))
    return {
        (code, owner)
        for owner, pairs in per_owner.items()
        for _, code in sorted(pairs)[:rarest_k]
    }


def reference_rarest_keep_by_entity(index, frame, df_cap: int, rarest_k: int, encoder=None):
    """:func:`reference_rarest_keep` mapped onto the index's own key/entity codes.

    ``encoder`` turns the key-column values into ``(codes, owners)``; it defaults to
    the token vocabulary's **read-only** lookup, so building the expectation cannot
    itself change the vocabulary it is checking.
    """
    texts = np.asarray(frame[index.key_field], dtype=object)
    if encoder is None:
        codes, owners, _ = index.vocabulary.lookup_texts(texts)
    else:
        codes, owners = encoder(texts)
    dfs = index.df.lookup(codes)
    entity_codes = encode_entity_ids(frame["entity_id"])
    return {
        (key, int(entity_codes[owner]))
        for key, owner in reference_rarest_keep(codes, owners, dfs, df_cap, rarest_k)
    }


def char_encoder(texts: np.ndarray):
    """``(codes, owners)`` for ``name_key`` values, using the real codec."""
    from src.blocking import _trigram_codes_for_list

    return _trigram_codes_for_list(texts)


def entity_code(entity_id: str) -> int:
    """The packed int64 code for one entity id."""
    return encode_entity_id(entity_id)


# ---------------------------------------------------------------------------
# codecs: agreement with the calibration scripts, which define the semantics
# ---------------------------------------------------------------------------
def test_tokenize_matches_calibration():
    from scripts.calibrate_token_blocker import tokenize as calibration_tokenize

    for text in ["", "acme", "acme holdings", "  acme   holdings  ", "a\tb\nc", "ACME Holdings"]:
        assert tokenize(text) == calibration_tokenize(text), text


def test_trigram_codes_match_calibration():
    from scripts.calibrate_char_blocker import trigram_codes as calibration_trigram_codes

    for text in ["", "a", "ab", "abc", "acmeholdings", "oréeleé", "Ω≈ç√", "blue sky exports"]:
        ours = trigram_codes(text)
        theirs = calibration_trigram_codes(text)
        assert np.array_equal(ours, theirs), text
        # injective, and ascending: the lookup skips string re-verification because
        # of the first property and binary search because of the second
        assert np.array_equal(ours, np.unique(ours)), text


def test_trigram_codes_short_and_empty_strings_have_none():
    for text in ["", "a", "ab"]:
        assert len(trigram_codes(text)) == 0, text
    assert len(trigram_codes("abc")) == 1


def test_trigram_codes_round_trip_through_decode():
    for text in ["abc", "acmeholdings", "Ω≈ç"]:
        for code in trigram_codes(text):
            decoded = decode_trigram_code(int(code))
            assert len(decoded) == 3, decoded
            assert code in trigram_codes(decoded)


def test_trigram_jaccard_matches_calibration():
    from scripts.analyze_name_differences import _trigram_jaccard as calibration_jaccard
    from src.blocking import _trigram_jaccard

    pairs = [
        ("acme holdings", "acme holdings"),
        ("acme holdings", "acme holding"),
        ("acmeholdings", "acmeomega"),
        ("a", "b"),
        ("ab", "abc"),
        ("", "abc"),
        ("", ""),
        ("quickmart", "quick mart"),
        ("oréeleé", "oreelee"),
        ("blue sky exports", "bluesky exports"),
        ("a", ""),
    ]
    for left, right in pairs:
        ours, theirs = _trigram_jaccard(left, right), calibration_jaccard(left, right)
        assert ours == theirs, (left, right, ours, theirs)
        assert calibration_jaccard(right, left) == ours, (left, right)


# ---------------------------------------------------------------------------
# exact blocker: unchanged behaviour
# ---------------------------------------------------------------------------
def test_exact_blocker_still_retrieves_identical_names():
    target = frame("source2", [(1, "acme holdings"), (2, "acme holdings"), (3, "acme omega")])
    index = build(ExactNameIndex, [target], "source2", key_field="name_norm")
    query = frame("source1", [(10, "acme holdings"), (11, "acme omega"), (12, "acme")])

    packed, evidence = index.query(query["name_norm"])
    assert evidence == {}
    s1_positions, codes = _unpack(packed)
    got = {(int(s), int(c)) for s, c in zip(s1_positions, codes)}
    assert got == {
        (0, entity_code("S2-1")),
        (0, entity_code("S2-2")),
        (1, entity_code("S2-3")),
    }, got
    assert _targets(index, query) == {"S2-1", "S2-2", "S2-3"}


def test_exact_index_round_trips_through_disk():
    target = frame("source2", [(1, "acme holdings"), (2, "acme omega")])
    index = build(ExactNameIndex, [target], "source2", key_field="name_norm")
    directory = _SCRATCH / "exact_round_trip"
    index.save(directory)

    reloaded = ExactNameIndex.load(directory, log=LOG)
    assert reloaded.key_field == "name_norm"
    query = frame("source1", [(10, "acme holdings")])
    assert np.array_equal(index.query(query["name_norm"])[0], reloaded.query(query["name_norm"])[0])


# ---------------------------------------------------------------------------
# token blocker
# ---------------------------------------------------------------------------
def test_token_eligibility_is_applied_before_ranking():
    """The one case that distinguishes "eligible then ranked" from "ranked then eligible".

    ``delta`` is absent from the target corpus, so its df is 0. An entity whose
    rarest key is that absent token has to fall through to its next-rarest
    *eligible* token; ranking first would keep ``delta`` and drop ``alpha``, and the
    entity would retrieve nothing.
    """
    # df("alpha") == 2 and df("solo") == 3, so "alpha" is S2-1/S2-2's rarest token and
    # both rows keep it - the pair must be retrievable for the test to mean anything.
    target = frame("source2", [(1, "alpha solo"), (2, "alpha solo"), (3, "solo trading")])
    index = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=1000, rarest_k=1)
    assert index.df.lookup(index.vocabulary.lookup_texts(np.array(["missing"], dtype=object))[0])[0] == 0

    query = frame("source1", [(10, "alpha missing")])
    packed, _ = index.query(query["name_norm"])
    s1_positions, codes = _unpack(packed)
    got = {(int(s), int(c)) for s, c in zip(s1_positions, codes)}
    assert got == {(0, entity_code("S2-1")), (0, entity_code("S2-2"))}, got


def test_token_df_cap_excludes_common_tokens():
    target = frame(
        "source2",
        [(1, "common thing"), (2, "common stuff"), (3, "common item"), (4, "rare thing")],
    )
    query = frame("source1", [(10, "common thing")])

    loose = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=1000, rarest_k=2)
    tight = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=2, rarest_k=2)

    assert _targets(loose, query) == {"S2-1", "S2-2", "S2-3", "S2-4"}
    # df("common") == 3 > cap 2, so only "thing" (df 2) survives
    assert _targets(tight, query) == {"S2-1", "S2-4"}


def test_token_df_is_counted_on_the_target_corpus_not_source1():
    """A token that is common in S1 but rare in the target corpus stays eligible.

    Only the target table is ever streamed into the index build, so S1's own
    frequency cannot inflate a df and disqualify a key - which is the silent failure
    this fixture rules out.
    """
    target = frame("source2", [(1, "solo shop")])
    index = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=2, rarest_k=1)
    query = frame("source1", [(10 + i, "solo shop") for i in range(5)])
    assert _targets(index, query) == {"S2-1"}


def test_token_query_does_not_grow_the_vocabulary():
    """An S1 token the target corpus never had stays absent rather than being interned."""
    target = frame("source2", [(1, "alpha beta")])
    index = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=1000, rarest_k=1)
    before = len(index.vocabulary)

    index.query(frame("source1", [(10, "brand new token")])["name_norm"])
    assert len(index.vocabulary) == before


def test_token_blocker_is_boolean_with_no_verification():
    """Sharing one eligible token is the decision - no similarity test follows it.

    ``acme zeta`` and ``acme omega`` share only ``acme``; their trigram Jaccard is
    well under 0.3. The token blocker must still propose the pair, and the char
    blocker must decline it.
    """
    target = frame("source2", [(1, "acme omega")])
    query = frame("source1", [(10, "acme zeta")])

    token = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=1000, rarest_k=1)
    assert _targets(token, query) == {"S2-1"}

    char = build(
        CharNgramIndex, [target], "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.3
    )
    assert _targets(char, query) == set()


def test_token_index_matches_the_plain_python_rule():
    target = frame(
        "source2",
        [
            (1, "acme holdings"),
            (2, "acme holdings"),
            (3, "acme omega"),
            (4, "zeta holdings"),
            (5, "zeta omega"),
        ],
    )
    index = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=1000, rarest_k=1)
    assert kept_pairs(index) == reference_rarest_keep_by_entity(index, target, 1000, 1)

    # A tighter K must be a strict subset, and building at the tight cell directly
    # must equal the loose build filtered down - which is the property that lets
    # production build at one frozen cell instead of building a grid.
    tighter = build(TokenIndex, [target], "source2", key_field="name_norm", df_cap=1000, rarest_k=0 + 2)
    assert kept_pairs(tighter) == reference_rarest_keep_by_entity(index, target, 1000, 2)


def test_token_index_matches_the_calibration_index_at_the_frozen_cell():
    """Structural regression against the calibration implementation.

    The calibration builds one loose index and filters it down per cell; production
    builds the frozen cell directly. For the two to be interchangeable the
    ``(keys, key_df, postings)`` triple has to agree exactly at the cell.
    """
    from scripts.calibrate_token_blocker import (
        _TokenVocabulary,
        build_token_index,
        count_token_df,
    )

    frames = [
        frame(
            "source2",
            [
                (1, "acme holdings"),
                (2, "acme holdings"),
                (3, "acme omega"),
                (4, "zeta holdings"),
                (5, "omega zeta"),
            ],
        )
    ]
    ours = build(TokenIndex, frames, "source2", key_field="name_norm", df_cap=1000, rarest_k=1)

    vocabulary = _TokenVocabulary("source2")
    df = count_token_df(iter(frames), "name_norm", vocabulary, LOG, "test")
    loose = build_token_index(
        iter(frames),
        df,
        vocabulary,
        "entity_id",
        "name_norm",
        max_df_cap=10_000,
        max_rank=10,
        source="source2",
        log=LOG,
        label="test",
    )
    for cap, rarest_k in ((1000, 1),):
        theirs = loose.filtered(cap, rarest_k)
        assert np.array_equal(ours.keys, theirs.keys), (cap, rarest_k)
        assert np.array_equal(ours.key_df, theirs.key_df), (cap, rarest_k)
        assert np.array_equal(ours.postings, theirs.postings), (cap, rarest_k)
        assert np.array_equal(ours.postings_offsets, theirs.postings_offsets), (cap, rarest_k)


def test_token_index_is_deterministic_and_round_trips():
    frames = [frame("source2", [(1, "acme holdings"), (2, "acme omega"), (3, "zeta holdings")])]
    first = build(TokenIndex, frames, "source2", key_field="name_norm", df_cap=1000, rarest_k=1)
    second = build(TokenIndex, frames, "source2", key_field="name_norm", df_cap=1000, rarest_k=1)
    _assert_same_index(first, second)

    directory = _SCRATCH / "token_round_trip"
    first.save(directory)
    reloaded = TokenIndex.load(directory, log=LOG)
    _assert_same_index(first, reloaded)
    assert reloaded.key_field == "name_norm"
    assert reloaded.df_cap == 1000 and reloaded.rarest_k == 1


# ---------------------------------------------------------------------------
# char blocker
# ---------------------------------------------------------------------------
def test_char_blocker_keys_on_name_key():
    """``quickmart`` vs ``quick mart``: identical ``name_key``, different ``name_norm``.

    This is the pair that forces the char blocker onto ``name_key`` - on ``name_norm``
    the two strings share no trigram run and the pair would be lost.
    """
    target = frame("source2", [(1, "quick mart")])
    index = build(
        CharNgramIndex, [target], "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.3
    )
    assert index.key_field == "name_key"
    assert target["name_norm"][0] != "quickmart", "fixture must differ in name_norm"

    query = frame("source1", [(10, "quickmart")])
    packed, evidence = index.query(query["name_key"])
    s1_positions, codes = _unpack(packed)
    assert {(int(s), int(c)) for s, c in zip(s1_positions, codes)} == {(0, entity_code("S2-1"))}
    assert evidence["char_jaccard"][0] == 1.0

    # ...and the exact blocker must NOT propose it.
    exact = build(ExactNameIndex, [target], "source2", key_field="name_norm")
    assert exact.query(query["name_norm"])[0].size == 0


def test_char_jaccard_threshold_is_inclusive():
    """A pair is kept exactly when its Jaccard reaches the threshold."""
    from src.blocking import _trigram_jaccard

    target = frame("source2", [(1, "acme omega"), (2, "quick mart")])
    query_rows = [(10, "acme zeta"), (11, "quickmart")]
    query = frame("source1", query_rows)

    for threshold in (0.0 + 1e-9, 0.3, 0.5, 0.99):
        index = build(
            CharNgramIndex,
            [target],
            "source2",
            key_field="name_key",
            df_cap=1000,
            rarest_k=5,
            jaccard=threshold,
        )
        got = {(int(s), int(c)) for s, c in zip(*_unpack(index.query(query["name_key"])[0]))}
        expected = set()
        for row, (_, name) in enumerate(query_rows):
            for numeric, target_name in [(1, "acme omega"), (2, "quick mart")]:
                similarity = _trigram_jaccard(key_of(name), key_of(target_name))
                if similarity >= threshold:
                    expected.add((row, entity_code(f"S2-{numeric}")))
        assert got == expected, (threshold, got, expected)


def test_char_verification_uses_the_same_jaccard_as_the_calibration():
    from scripts.analyze_name_differences import _trigram_jaccard as calibration_jaccard

    target = frame("source2", [(1, "acme omega"), (2, "quick mart"), (3, "acme holdings")])
    query = frame("source1", [(10, "acme zeta"), (11, "quickmart"), (12, "acme holding")])
    index = build(
        CharNgramIndex, [target], "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.1
    )
    packed, evidence = index.query(query["name_key"])
    s1_positions, codes = _unpack(packed)
    from src.utils import decode_entity_id

    for row, code, similarity in zip(s1_positions, codes, evidence["char_jaccard"]):
        left = key_of(query["name_norm"][int(row)])
        right = key_of(target["name_norm"][_row_of(decode_entity_id(int(code)))])
        assert similarity == calibration_jaccard(left, right), (left, right)


def test_char_rarest_k_is_five():
    target = frame("source2", [(1, "abcdefghij"), (2, "klmnopqrst")])
    index = build(
        CharNgramIndex, [target], "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.3
    )
    expected = reference_rarest_keep_by_entity(index, target, 1000, 5, encoder=char_encoder)
    assert kept_pairs(index) == expected
    # every entity contributes exactly five keys: eight unique trigrams, five kept
    for numeric in (1, 2):
        assert sum(1 for _, entity in expected if entity == entity_code(f"S2-{numeric}")) == 5


def test_char_short_names_contribute_no_keys():
    target = frame("source2", [(1, "ab"), (2, "abc")])
    index = build(
        CharNgramIndex, [target], "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.3
    )
    assert kept_pairs(index) == reference_rarest_keep_by_entity(index, target, 1000, 5, encoder=char_encoder)
    # "ab" has no trigram at all; "abc" has exactly one
    assert all(entity == entity_code("S2-2") for _, entity in kept_pairs(index))


def test_char_index_matches_the_calibration_index_at_the_frozen_cell():
    from scripts.calibrate_char_blocker import (
        _TrigramDf,
        build_trigram_index,
        count_trigram_df,
    )

    frames = [
        frame(
            "source2",
            [
                (1, "acme holdings"),
                (2, "acme holdings"),
                (3, "acme omega"),
                (4, "quick mart"),
                (5, "quickmart"),
            ],
        )
    ]
    ours = build(
        CharNgramIndex, frames, "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.3
    )
    df = count_trigram_df(iter(frames), "name_key", LOG, "test")
    assert isinstance(df, _TrigramDf)
    loose = build_trigram_index(
        iter(frames),
        df,
        "entity_id",
        "name_key",
        max_df_cap=10_000,
        max_rank=10,
        source="source2",
        log=LOG,
        label="test",
    )
    theirs = loose.filtered(1000, 5)
    assert np.array_equal(ours.keys, theirs.keys)
    assert np.array_equal(ours.key_df, theirs.key_df)
    assert np.array_equal(ours.postings, theirs.postings)
    assert np.array_equal(ours.postings_offsets, theirs.postings_offsets)


def test_char_verification_is_deterministic_across_worker_counts():
    """``compute.num_workers`` is a performance knob, never a semantic one."""
    target = frame("source2", [(numeric, f"acme holdings {numeric}") for numeric in range(1, 9)])
    query = frame("source1", [(10, "acme holdings")])
    query = pd.concat([query] * 40, ignore_index=True)
    query["entity_id"] = [f"S1-{100 + i}" for i in range(len(query))]

    directory = _SCRATCH / "char_workers"
    serial = build(
        CharNgramIndex, [target], "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.3
    )
    serial.save(directory)

    sequential = load_index_at(directory, workers=1).query(query["name_key"])
    parallel = load_index_at(directory, workers=2).query(query["name_key"])
    assert np.array_equal(sequential[0], parallel[0])
    assert np.array_equal(sequential[1]["char_jaccard"], parallel[1]["char_jaccard"])


def test_char_index_round_trips_through_disk():
    target = frame("source2", [(1, "acme holdings"), (2, "quick mart")])
    index = build(
        CharNgramIndex, [target], "source2", key_field="name_key", df_cap=1000, rarest_k=5, jaccard=0.3
    )
    directory = _SCRATCH / "char_round_trip"
    index.save(directory)
    reloaded = load_index_at(directory)
    assert reloaded.key_field == "name_key"
    assert reloaded.jaccard_threshold == 0.3
    assert reloaded.name_key_at_row(0) == key_of("acme holdings")
    for row in range(reloaded.n_rows):
        assert reloaded.name_key_at_row(row) == target["name_key"][row]

    query = frame("source1", [(10, "quickmart")])
    assert np.array_equal(index.query(query["name_key"])[0], reloaded.query(query["name_key"])[0])


# ---------------------------------------------------------------------------
# union, provenance, evidence
# ---------------------------------------------------------------------------
def test_union_is_a_union_not_an_intersection():
    """A pair only one blocker proposed must survive; an intersection would drop it."""
    only_exact = pack_pairs(np.array([0]), np.array([entity_code("S2-1")]))
    only_char = pack_pairs(np.array([1]), np.array([entity_code("S2-2")]))
    shared = pack_pairs(np.array([2]), np.array([entity_code("S2-3")]))

    s1_positions, codes, provenance, evidence = union_blockers(
        {"source2:exact_name": only_exact, "source2:char_ngram": np.concatenate([only_char, shared])}
    )
    assert len(s1_positions) == 3
    assert set(provenance) == {"source2:exact_name", "source2:char_ngram"}
    assert evidence == {}


def test_union_deduplicates_and_records_multi_blocker_provenance():
    pair = pack_pairs(np.array([0]), np.array([entity_code("S2-1")]))
    other = pack_pairs(np.array([1]), np.array([entity_code("S2-2")]))
    s1_positions, codes, provenance, _ = union_blockers(
        {
            "source3:char_ngram": np.concatenate([pair, other]),
            "source2:exact_name": pair,
            "source2:token": pair.copy(),
        }
    )
    assert len(s1_positions) == 2
    rows = {
        (int(s), int(c)): str(p) for s, c, p in zip(s1_positions, codes, provenance)
    }
    shared_key = (0, entity_code("S2-1"))
    assert rows[shared_key] == "source2:exact_name,source2:token,source3:char_ngram", rows[shared_key]
    assert rows[(1, entity_code("S2-2"))] == "source3:char_ngram"
    # deterministic order: sorted by (s1_position, entity_code)
    assert list(zip(s1_positions, codes)) == sorted(zip(s1_positions, codes))


def test_union_evidence_is_the_minimum_and_nan_where_absent():
    pair_a = pack_pairs(np.array([0]), np.array([entity_code("S2-1")]))
    pair_b = pack_pairs(np.array([0]), np.array([entity_code("S2-2")]))
    s1_positions, codes, provenance, evidence = union_blockers(
        {"source2:token": np.concatenate([pair_a, pair_b])},
        blocker_evidence={
            "source2:token": {"token_df": np.array([7.0, 3.0])},
            "source2:char_ngram": {"char_jaccard": np.array([0.9, 0.4])},
        },
    )
    assert len(s1_positions) == 2
    token_df = dict(zip(codes, evidence["token_df"]))
    assert token_df[entity_code("S2-1")] == 7.0
    assert token_df[entity_code("S2-2")] == 3.0
    # the char blocker proposed no pair at all, so it contributes no column; the
    # writer fills the whole column blank via format_evidence(None, ...)
    assert "char_jaccard" not in evidence
    assert all(str(p) == "source2:token" for p in provenance)


def test_union_evidence_takes_the_minimum_across_blockers():
    pair = pack_pairs(np.array([0]), np.array([entity_code("S2-1")]))
    _, codes, _, evidence = union_blockers(
        {"source2:token": pair, "source3:token": pair.copy()},
        blocker_evidence={
            "source2:token": {"token_df": np.array([9.0])},
            "source3:token": {"token_df": np.array([4.0])},
        },
    )
    assert evidence["token_df"][0] == 4.0
    assert codes.tolist() == [entity_code("S2-1"), entity_code("S2-1")][: len(codes)]


def test_union_accepts_empty_blockers():
    empty = np.empty(0, dtype=np.int64)
    s1_positions, codes, provenance, evidence = union_blockers(
        {"source2:exact_name": empty, "source2:token": empty}
    )
    assert len(s1_positions) == 0 and len(codes) == 0 and len(provenance) == 0
    assert evidence == {}


def test_union_rejects_misaligned_evidence():
    pair = pack_pairs(np.array([0]), np.array([entity_code("S2-1")]))
    try:
        union_blockers(
            {"source2:token": pair},
            blocker_evidence={"source2:token": {"token_df": np.array([1.0, 2.0])}},
        )
    except ValueError as error:
        assert "aligned" in str(error) or "values" in str(error)
    else:
        raise AssertionError("misaligned evidence must raise")


# ---------------------------------------------------------------------------
# registry and settings
# ---------------------------------------------------------------------------
def test_registry_covers_the_union_including_dense():
    assert LEXICAL_BLOCKERS == (BLOCKER_EXACT_NAME, BLOCKER_TOKEN, BLOCKER_CHAR_NGRAM)
    # dense is last, so the lexical provenance strings are unchanged by its addition
    assert UNION_BLOCKERS == LEXICAL_BLOCKERS + (BLOCKER_DENSE,)
    for blocker in UNION_BLOCKERS:
        assert blocker in INDEX_BUILDERS, blocker
        assert blocker in INDEX_LOADERS, blocker


def test_evidence_columns_follow_the_enabled_blockers():
    assert evidence_columns_for([BLOCKER_EXACT_NAME]) == []
    assert evidence_columns_for([BLOCKER_EXACT_NAME, BLOCKER_TOKEN, BLOCKER_CHAR_NGRAM]) == [
        "token_df",
        "char_jaccard",
    ]
    # registry order, not the order the caller happened to list them in
    assert evidence_columns_for([BLOCKER_CHAR_NGRAM, BLOCKER_TOKEN]) == ["token_df", "char_jaccard"]


def test_blocker_settings_default_to_the_frozen_cell():
    assert resolve_blocker_settings({}, BLOCKER_TOKEN) == {"df_cap": 1000, "rarest_k": 1}
    assert resolve_blocker_settings({}, BLOCKER_CHAR_NGRAM) == {
        "df_cap": 1000,
        "rarest_k": 5,
        "jaccard": 0.3,
    }
    assert resolve_blocker_settings({}, BLOCKER_EXACT_NAME) == {}


def test_blocker_settings_reject_typos_and_bad_values():
    for section, message in [
        ({"rarest_k": 0}, "positive integer"),
        ({"df_cap": -1}, "positive integer"),
        ({"df_cap": True}, "positive integer"),
        ({"jaccard": 0.0}, "(0, 1]"),
        ({"jaccard": 1.5}, "(0, 1]"),
        ({"raerest_k": 1}, "unknown setting"),
    ]:
        try:
            resolve_blocker_settings({"blocking": {BLOCKER_CHAR_NGRAM: section}}, BLOCKER_CHAR_NGRAM)
        except ValueError as error:
            assert message in str(error), (section, str(error))
        else:
            raise AssertionError(f"{section} should have been rejected")


def test_key_fields_default_per_blocker():
    from src.blocking import _key_field_for

    config = {"blocking": {BLOCKER_EXACT_NAME: {"enabled": True, "key": "name_norm"}}}
    assert _key_field_for(config, BLOCKER_EXACT_NAME) == "name_norm"
    # a config that enables char without naming a column must get name_key, because
    # name_norm would be a different blocker than the one that was calibrated
    assert _key_field_for(config, BLOCKER_CHAR_NGRAM) == "name_key"
    assert _key_field_for(config, BLOCKER_TOKEN) == "name_norm"


# ---------------------------------------------------------------------------
# end-to-end: the real script, a mixed-key union, on a tiny prepared fixture
# ---------------------------------------------------------------------------
SOURCE2_ROWS = [
    (1, "acme holdings"),
    (2, "acme holdings"),
    (3, "zephyr omega"),
    (4, "quick mart"),
    (5, "omega trading"),
]
SOURCE3_ROWS = [
    (1, "acme holdings"),
    (2, "quasar trading"),
    (3, "core logistics"),
]
SOURCE1_ROWS = [
    (1, "acme holdings"),
    (2, "zephyr quasar"),
    (3, "quickmart"),
    (4, "quasar trading"),
    (5, "corelogistics"),
]


def _write_end_to_end_fixture(root: Path) -> Path:
    """A complete miniature dataset: prepared tables, a config, and the indexes.

    Every resolved path is redirected into ``root``. ``config.yaml`` names
    ``prepared_dir``/``index_dir``/``candidates_dir`` explicitly, and an explicit
    path is resolved against the repository root rather than against ``work_dir`` -
    so overriding only ``work_dir`` would read and write the real ``outputs/`` tree,
    and a stale index from an earlier run would be loaded instead of rebuilt.
    """
    root.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("paths", {})
    config["paths"].update(
        {
            "work_dir": str(root / "work"),
            "prepared_dir": str(root / "work" / "prepared"),
            "index_dir": str(root / "work" / "indexes"),
            "candidates_dir": str(root / "work" / "candidates"),
            "log_dir": str(root / "logs"),
        }
    )
    # small chunks so the streaming path is exercised across chunk boundaries
    config.setdefault("io", {})["chunksize"] = 3
    config.setdefault("compute", {})["num_workers"] = 1

    config_path = root / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    loaded = load_config(config_path)
    for source, rows in (("source1", SOURCE1_ROWS), ("source2", SOURCE2_ROWS), ("source3", SOURCE3_ROWS)):
        table = frame(source, rows)
        path = prepared_path(loaded, "train", source)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False)

    for source in ("source2", "source3"):
        # The lexical set: the config leaves dense disabled, and the dense blocker has
        # its own tests (tests/test_dense_blocker.py) with a download-free encoder.
        for blocker in LEXICAL_BLOCKERS:
            build_index(loaded, "train", source, blocker, log=LOG)
    return config_path


_FIXTURE_ROOT = _SCRATCH / "end_to_end"
_FIXTURE_CONFIG = _write_end_to_end_fixture(_FIXTURE_ROOT)


def _run_generate(extra: list[str], name: str) -> tuple[pd.DataFrame, dict]:
    import json

    from scripts import generate_candidates

    argv = [
        "--config", str(_FIXTURE_CONFIG),
        "--log-level", "CRITICAL",
        "--name", name,
        *extra,
    ]
    code = generate_candidates.main(argv)
    assert code == 0, code
    loaded = load_config(_FIXTURE_CONFIG)
    from src.data_loader import candidates_path

    output = candidates_path(loaded, name)
    table = read_tsv(output)
    with open(output.with_name(output.stem + "_stats.json"), encoding="utf-8") as handle:
        stats = json.load(handle)
    return table, stats


def test_end_to_end_mixed_key_union():
    table, stats = _run_generate([], "mixed")
    assert list(table.columns) == [
        "source1_entity_id",
        "matched_entity_id",
        "source",
        "blockers",
        "token_df",
        "char_jaccard",
    ]
    assert stats["blockers"] == ["exact_name", "token", "char_ngram"]
    assert stats["s1_key_fields"] == ["name_key", "name_norm"]
    assert stats["evidence_columns"] == ["token_df", "char_jaccard"]

    rows = {
        (row.source1_entity_id, row.matched_entity_id): row
        for row in table.itertuples(index=False)
    }

    # exact only reaches identical name_norm; token and char necessarily agree
    assert "source2:char_ngram" in rows[("S1-1", "S2-1")].blockers
    assert "source2:exact_name" in rows[("S1-1", "S2-1")].blockers
    assert "source2:token" in rows[("S1-1", "S2-1")].blockers
    assert rows[("S1-1", "S2-1")].source == "S2"
    assert rows[("S1-1", "S2-1")].char_jaccard == "1.0000"
    assert rows[("S1-1", "S2-1")].token_df == "2"

    # token only: shares the eligible token "zephyr" (df 1), no exact key, and a
    # trigram Jaccard of 0.2 - below the char threshold
    assert rows[("S1-2", "S2-3")].blockers == "source2:token"
    assert rows[("S1-2", "S2-3")].char_jaccard == ""
    assert rows[("S1-2", "S2-3")].token_df == "1"

    # char only: identical name_key, different name_norm, no shared token
    assert rows[("S1-3", "S2-4")].blockers == "source2:char_ngram"
    assert rows[("S1-3", "S2-4")].char_jaccard == "1.0000"
    assert rows[("S1-3", "S2-4")].token_df == ""

    # source3 is not source2: the same three blockers fire through its own index
    assert set(rows[("S1-4", "S3-2")].blockers.split(",")) == {
        "source3:exact_name",
        "source3:token",
        "source3:char_ngram",
    }
    assert rows[("S1-4", "S3-2")].source == "S3"
    assert rows[("S1-5", "S3-3")].blockers == "source3:char_ngram"

    # the source label always agrees with the target id, and no S1 is dropped
    assert (table["source"] == table["matched_entity_id"].str.slice(0, 2)).all()
    assert set(table["source1_entity_id"]) == {f"S1-{n}" for n, _ in SOURCE1_ROWS}
    # no candidate is emitted twice
    assert not table.duplicated(["source1_entity_id", "matched_entity_id"]).any()
    # every pair is internally consistent: the union never invents a blocker
    label_for = {"S2": "source2", "S3": "source3"}
    for row in table.itertuples(index=False):
        assert row.blockers, row
        for name in row.blockers.split(","):
            assert name.startswith(label_for[row.source] + ":"), row


def test_end_to_end_exact_only_keeps_the_original_schema():
    table, stats = _run_generate(["--blockers", "exact_name"], "exact_only")
    assert list(table.columns) == [
        "source1_entity_id",
        "matched_entity_id",
        "source",
        "blockers",
    ]
    assert stats["evidence_columns"] == []
    assert set(table["blockers"]) == {"source2:exact_name", "source3:exact_name"}
    assert set(table["source1_entity_id"]) == {"S1-1", "S1-4"}


def test_end_to_end_blocker_order_does_not_depend_on_the_cli():
    forward, _ = _run_generate(["--blockers", "char_ngram,token,exact_name"], "cli_forward")
    reverse, _ = _run_generate(["--blockers", "exact_name,char_ngram,token"], "cli_reverse")
    forward = forward.sort_values(["source1_entity_id", "matched_entity_id"]).reset_index(drop=True)
    reverse = reverse.sort_values(["source1_entity_id", "matched_entity_id"]).reset_index(drop=True)
    assert forward.equals(reverse)


def test_end_to_end_is_deterministic():
    first, stats_first = _run_generate([], "determinism_a")
    second, stats_second = _run_generate([], "determinism_b")
    assert first.equals(second)
    for key in ("candidate_pairs", "pairs_by_source", "pairs_by_blocker_before_union"):
        assert stats_first[key] == stats_second[key], key


def test_end_to_end_worker_count_does_not_change_the_output():
    """``--workers`` is a performance knob end to end, not just inside one index."""
    serial, _ = _run_generate(["--workers", "1"], "workers_1")
    parallel, _ = _run_generate(["--workers", "3"], "workers_3")
    serial = serial.sort_values(["source1_entity_id", "matched_entity_id"]).reset_index(drop=True)
    parallel = parallel.sort_values(["source1_entity_id", "matched_entity_id"]).reset_index(drop=True)
    assert serial.equals(parallel)


def test_end_to_end_candidate_cap_still_applies():
    table, _ = _run_generate(["--max-candidates", "1"], "capped")
    per_s1 = table.groupby("source1_entity_id").size()
    assert int(per_s1.max()) <= 1
    uncapped, _ = _run_generate([], "uncapped")
    assert int(uncapped.groupby("source1_entity_id").size().max()) > 1


def test_end_to_end_limit_s1_still_applies():
    table, stats = _run_generate(["--limit-s1", "2"], "limited")
    assert stats["s1_entities_processed"] == 2
    assert set(table["source1_entity_id"]) <= {"S1-1", "S1-2"}


def test_end_to_end_rejects_an_unknown_blocker():
    from scripts import generate_candidates

    code = generate_candidates.main(
        ["--config", str(_FIXTURE_CONFIG), "--log-level", "CRITICAL", "--blockers", "nonsense"]
    )
    assert code == 2


def test_generate_candidates_help_still_lists_the_original_flags():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_candidates.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--config", "--data-root", "--work-dir", "--split", "--sources", "--blockers",
                 "--limit-s1", "--chunksize", "--max-candidates", "--workers", "--name", "--log-level"):
        assert flag in result.stdout, flag


# ---------------------------------------------------------------------------
# shared assertions
# ---------------------------------------------------------------------------
def _unpack(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from src.blocking import unpack_pairs

    if len(packed) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return unpack_pairs(packed)


def _targets(index, query_frame: pd.DataFrame) -> set[str]:
    from src.utils import decode_entity_ids

    packed, _ = index.query(query_frame[index.key_field])
    if len(packed) == 0:
        return set()
    _, codes = _unpack(packed)
    return set(decode_entity_ids(codes).tolist())


def _row_of(entity_id: str) -> int:
    return int(entity_id.split("-")[1]) - 1


def _assert_same_index(left, right) -> None:
    for attribute in ("keys", "key_df", "postings", "postings_offsets"):
        assert np.array_equal(getattr(left, attribute), getattr(right, attribute)), attribute
    assert left.key_field == right.key_field
    assert left.df_cap == right.df_cap and left.rarest_k == right.rarest_k


def load_index_at(directory: Path, workers: int = 1):
    """Load a persisted index of any implemented blocker, by directory."""
    from src.blocking import _load_index_at

    return _load_index_at(directory, BLOCKER_CHAR_NGRAM, log=LOG, workers=workers)


# ---------------------------------------------------------------------------
# standalone runner (no pytest required)
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        else:
            print(f"[PASS] {test.__name__}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} tests failed")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
