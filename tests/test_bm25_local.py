import math

import pytest

from searchengine.data.toy import make_toy_collection
from searchengine.eval.metrics import evaluate
from searchengine.retrieval.bm25_local import LocalBM25
from searchengine.text import Analyzer
from searchengine.types import Document

DOCS = [
    Document("d1", "the quick brown fox jumps over the lazy dog"),
    Document("d2", "a quick brown dog"),
    Document("d3", "foxes and dogs are animals; the fox is quick"),
    Document("d4", "nothing relevant here at all"),
]


def test_analyzer_matches_es_english_pipeline():
    a = Analyzer()
    assert a("The runner's shoes are RUNNING") == ["runner", "shoe", "run"]
    assert a("the and of") == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("the runner’s shoes", ["runner", "shoes"]),  # curly possessive stripped, no junk "s"
        ("don’t stop", ["don’t", "stop"]),  # in-word apostrophe kept, as Lucene does
        ("costs 3.5 dollars, 1,000 total", ["costs", "3.5", "dollars", "1,000", "total"]),
        ("the U.S. army", ["u.s", "army"]),  # trailing period is not between word chars
        ("end of sentence. Next", ["end", "sentence", "next"]),  # period + space splits
        ("a,b", ["b"]),  # comma joins digits only ("a" is a stopword)
        ("e.g. foo_bar v1.2", ["e.g", "foo_bar", "v1.2"]),
        ("No.1 Court high.4", ["1", "court", "high", "4"]),  # letter.digit splits ("no" is a stopword)
        ("Prevnar 13® Docs™", ["prevnar", "13", "®", "docs", "™"]),  # pictographs are tokens
        ("Residence:_____ here", ["residence", "here"]),  # underscore-only words dropped
    ],
)
def test_tokenizer_v3_uax29_rules(text, expected):
    assert Analyzer(stemmer=None)(text) == expected


@pytest.mark.parametrize(
    "word,lucene_stem",
    # Observed from Elasticsearch's `_analyze` on MS MARCO (scripts/check_analyzer.py output).
    [("technologies", "technolog"), ("assembly", "assembl"), ("possibly", "possibl"),
     ("psychology", "psycholog"), ("anthropology", "anthropolog"), ("us", "us"), ("s", "s"),
     ("u.s", "u."), ("running", "run"), ("news", "new")],
)
def test_stemmer_matches_lucene_porter(word, lucene_stem):
    assert Analyzer(stopwords=None)(word) == [lucene_stem]


def test_short_words_not_stemmed_and_no_empty_terms():
    for a in (Analyzer(), Analyzer(stemmer="porter", tokenizer="v2")):
        assert a("us as s es news") == ["us", "s", "es", "new"]  # "as" is a stopword
        assert "" not in a("the runner’s 1990 s era")


def test_lucene_length_quantization():
    from searchengine.retrieval.bm25_local import lucene_quantize_length as q

    # Exact below 40, then 3 mantissa bits: values verified against SmallFloat's codec.
    assert [q(n) for n in (0, 23, 24, 39, 40, 41, 55, 57, 87, 88, 100, 1000)] == [
        0, 23, 24, 39, 40, 40, 54, 56, 84, 88, 96, 984]
    # Monotone and never rounds up.
    vals = [q(n) for n in range(5000)]
    assert all(a <= b for a, b in zip(vals, vals[1:])) and all(v <= n for n, v in enumerate(vals))


def test_lucene_norms_flag_changes_long_docs_only():
    long_doc = Document("long", " ".join(["fox"] + ["filler%d" % i for i in range(60)]))
    docs = DOCS + [long_doc]
    exact = {h.doc_id: h.score for h in LocalBM25.build(docs, lucene_norms=False).search("fox", 10)}
    quant = {h.doc_id: h.score for h in LocalBM25.build(docs, lucene_norms=True).search("fox", 10)}
    assert quant["d1"] == pytest.approx(exact["d1"], rel=1e-6)  # short doc: identical
    assert quant["long"] > exact["long"]  # length rounded down -> smaller penalty


def test_v1_analyzer_still_loadable_for_old_indexes():
    old = Analyzer.from_config({"stopwords": "lucene_english", "stemmer": "porter"})
    assert old.tokenizer == "v1"
    # v1's behaviour, preserved exactly so an index built with it is queried consistently:
    # the curly possessive splits off an "s", which Snowball Porter stems to "".
    assert old("the runner’s") == ["runner", ""]


def test_score_matches_formula():
    k1, b = 1.2, 0.75
    bm = LocalBM25.build(DOCS, k1=k1, b=b)
    an = Analyzer()
    toks = [an(d.text) for d in DOCS]
    N = len(DOCS)
    avgdl = sum(map(len, toks)) / N
    q = an("quick fox")

    def expected(j):
        s = 0.0
        for t in q:
            df = sum(1 for tt in toks if t in tt)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            tf = toks[j].count(t)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(toks[j]) / avgdl))
        return s

    hits = {h.doc_id: h.score for h in bm.search("quick fox", k=10)}
    for j, d in enumerate(DOCS):
        e = expected(j)
        if e == 0:
            assert d.id not in hits
        else:
            assert hits[d.id] == pytest.approx(e, rel=1e-5)


def test_repeated_query_terms_add_up():
    bm = LocalBM25.build(DOCS)
    one = {h.doc_id: h.score for h in bm.search("fox", 10)}
    two = {h.doc_id: h.score for h in bm.search("fox fox", 10)}
    for d in one:
        assert two[d] == pytest.approx(2 * one[d], rel=1e-6)


def test_empty_and_oov_queries():
    bm = LocalBM25.build(DOCS)
    assert bm.search("the of and", 10) == []
    assert bm.search("zzzunknown", 10) == []
    assert bm.batch_search({"a": "the", "b": "fox"}, 10)["a"] == {}


def test_batch_equals_single_and_topk_is_sorted():
    docs, queries, _ = make_toy_collection(n_docs=500, n_queries=30)
    bm = LocalBM25.build(docs)
    batch = bm.batch_search(queries, k=20, batch_size=7)
    for qid, text in queries.items():
        single = bm.search(text, 20)
        assert [h.doc_id for h in single] == sorted(batch[qid], key=lambda d: (batch[qid][d], d), reverse=True)
        scores = [h.score for h in single]
        assert scores == sorted(scores, reverse=True)


def test_set_params_equals_fresh_build(tmp_path):
    docs, queries, _ = make_toy_collection(n_docs=300, n_queries=10)
    bm = LocalBM25.build(docs, k1=0.9, b=0.4)
    bm.set_params(1.5, 0.9)
    fresh = LocalBM25.build(docs, k1=1.5, b=0.9)
    for text in list(queries.values())[:5]:
        assert [(h.doc_id, round(h.score, 5)) for h in bm.search(text, 10)] == [
            (h.doc_id, round(h.score, 5)) for h in fresh.search(text, 10)
        ]
    bm.save(tmp_path / "idx")
    loaded = LocalBM25.load(tmp_path / "idx")
    assert (loaded.k1, loaded.b) == (1.5, 0.9)
    q = next(iter(queries.values()))
    assert [h.doc_id for h in loaded.search(q, 10)] == [h.doc_id for h in bm.search(q, 10)]


def test_bm25_beats_random_on_toy_collection():
    docs, queries, qrels = make_toy_collection()
    run = LocalBM25.build(docs).batch_search(queries, k=100)
    res = evaluate(run, qrels, ["ndcg@10", "recall@100"])
    assert res.mean["ndcg@10"] > 0.5
