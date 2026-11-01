"""Elasticsearch tests.

Unit tests use a fake client to check request bodies and response parsing.
The integration test runs only when a real cluster answers at ES_URL:
    docker compose up -d && pytest -m es
"""

import os

import pytest

from searchengine.data.toy import make_toy_collection
from searchengine.index.es import SEARCH_FIELD, index_definition
from searchengine.retrieval.bm25_es import ElasticsearchBM25


class FakeES:
    def __init__(self):
        self.calls = []

    def msearch(self, searches):
        self.calls.append(searches)
        out = []
        for header, body in zip(searches[0::2], searches[1::2]):
            q = body["query"]["match"][SEARCH_FIELD]["query"]
            out.append({"hits": {"hits": [{"_id": f"{q}-1", "_score": 2.0}, {"_id": f"{q}-2", "_score": 1.0}]}})
        return {"responses": out}

    def search(self, index, **body):
        return {"hits": {"hits": [{"_id": "x", "_score": 3.5}]}}


def test_index_definition_single_shard_default_similarity():
    d = index_definition(0.82, 0.68)
    assert d["settings"]["number_of_shards"] == 1
    assert d["settings"]["similarity"]["default"] == {"type": "BM25", "k1": 0.82, "b": 0.68}
    assert d["mappings"]["properties"][SEARCH_FIELD]["analyzer"] == "english"
    assert SEARCH_FIELD in d["mappings"]["_source"]["excludes"]


def test_msearch_batching_and_parsing():
    fake = FakeES()
    r = ElasticsearchBM25(fake, "idx")
    queries = {f"q{i}": f"text{i}" for i in range(5)}
    run = r.batch_search(queries, k=10, batch_size=2)
    assert len(fake.calls) == 3  # ceil(5 / 2)
    header, body = fake.calls[0][0], fake.calls[0][1]
    assert header == {"index": "idx"}
    assert body["size"] == 10 and body["_source"] is False and body["track_total_hits"] is False
    assert run["q3"] == {"text3-1": 2.0, "text3-2": 1.0}


def test_single_search_and_empty_query():
    r = ElasticsearchBM25(FakeES(), "idx")
    assert r.search("hello", 5)[0].doc_id == "x"
    assert r.search("   ", 5) == []


def _es_available():
    try:
        from searchengine.index.es import make_client

        return make_client(os.environ.get("ES_URL", "http://127.0.0.1:9200"), timeout=2).ping()
    except Exception:
        return False


@pytest.mark.es
@pytest.mark.skipif(not _es_available(), reason="no Elasticsearch at ES_URL")
def test_es_matches_local_bm25_on_toy():
    from searchengine.eval.metrics import evaluate
    from searchengine.index.es import bulk_index, create_index, make_client
    from searchengine.retrieval.bm25_local import LocalBM25

    docs, queries, qrels = make_toy_collection(n_docs=1000, n_queries=40)
    es = make_client(os.environ.get("ES_URL", "http://127.0.0.1:9200"))
    name = "pytest-toy"
    create_index(es, name, 0.9, 0.4, overwrite=True)
    bulk_index(es, name, docs)
    try:
        es_run = ElasticsearchBM25(es, name).batch_search(queries, k=100)
        local_run = LocalBM25.build(docs, 0.9, 0.4).batch_search(queries, k=100)
        m_es = evaluate(es_run, qrels, ["ndcg@10"]).mean["ndcg@10"]
        m_local = evaluate(local_run, qrels, ["ndcg@10"]).mean["ndcg@10"]
        assert abs(m_es - m_local) < 0.005
        # Score-level parity: same analyzer, IDF, avgdl, length quantization and (k1+1).
        local = LocalBM25.build(docs, 0.9, 0.4)
        for qid, text in list(queries.items())[:20]:
            ids = list(es_run[qid])[:10]
            ours = local.score_docs(text, ids)
            for d, s in zip(ids, ours):
                assert s == pytest.approx(es_run[qid][d], rel=1e-4)
    finally:
        es.indices.delete(index=name)
