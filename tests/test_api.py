import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from searchengine.api import create_app  # noqa: E402
from searchengine.docstore import DocStore  # noqa: E402
from searchengine.pipeline import SearchPipeline  # noqa: E402
from searchengine.retrieval.bm25_local import LocalBM25  # noqa: E402


@pytest.fixture
def client(toy_collection, tmp_path, reverse_length_reranker):
    coll, docs, queries, _ = toy_collection
    pipe = SearchPipeline(LocalBM25.build(docs), DocStore.open(coll, tmp_path), reverse_length_reranker)
    with TestClient(create_app(pipe, {"collection": "toy", "backend": "local"})) as c:
        c.queries = queries
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["retriever"] == "bm25-local"
    assert body["reranker"]["name"] == "fake:reverse-length"


def test_search_reranked(client):
    q = next(iter(client.queries.values()))
    r = client.get("/search", params={"q": q, "k": 5, "candidates": 40})
    assert r.status_code == 200
    body = r.json()
    assert body["reranked"] is True and body["candidates"] == 40 and len(body["results"]) == 5
    first = body["results"][0]
    assert set(first) == {"rank", "id", "title", "snippet", "score", "first_stage_score", "first_stage_rank",
                          "rerank_score", "relevance"}
    assert first["rank"] == 1 and 1 <= first["first_stage_rank"] <= 40
    assert body["retriever"] == "bm25-local" and body["filtered_out"] == 0
    assert set(body["timings"]) == {"retrieve_ms", "fetch_ms", "rerank_ms", "total_ms"}


def test_search_without_rerank(client):
    q = next(iter(client.queries.values()))
    body = client.get("/search", params={"q": q, "k": 3, "rerank": "false"}).json()
    assert body["reranked"] is False
    assert [r["first_stage_rank"] for r in body["results"]] == [1, 2, 3]
    assert all(r["rerank_score"] is None and r["relevance"] is None for r in body["results"])


@pytest.mark.parametrize(
    "params",
    [{"q": ""}, {"q": "   "}, {"q": "x", "k": 0}, {"q": "x", "k": 101}, {"q": "x", "k": 20, "candidates": 10}],
)
def test_validation(client, params):
    assert client.get("/search", params=params).status_code == 422


def test_query_matching_nothing_returns_empty(client):
    body = client.get("/search", params={"q": "zzzz qqqq"}).json()
    assert body["results"] == []


def test_build_pipeline_keeps_dml_away_from_torch(monkeypatch):
    import searchengine.factory as f

    seen = {}
    monkeypatch.setattr(f, "Collection", type("C", (), {"from_name": staticmethod(lambda n, d: None)}))
    monkeypatch.setattr(f, "build_retriever", lambda backend, coll, s, **kw: seen.setdefault("retriever", kw["device"]))
    monkeypatch.setattr(f, "build_reranker", lambda model, device, **kw: seen.setdefault("reranker", device))
    monkeypatch.setattr(f, "build_docs", lambda *a, **kw: None)
    f.build_pipeline("x", backend="dense", model="m", device="dml")
    assert seen == {"retriever": "auto", "reranker": "dml"}


def test_relevance_is_sigmoid_and_min_relevance_filters(client):
    import math

    q = next(iter(client.queries.values()))
    body = client.get("/search", params={"q": q, "k": 5, "candidates": 40}).json()
    for r in body["results"]:  # fake reranker scores are -len(passage): very negative logits
        assert r["relevance"] == pytest.approx(1 / (1 + math.exp(-r["rerank_score"])), abs=1e-4)
    strict = client.get("/search", params={"q": q, "k": 5, "candidates": 40, "min_relevance": 0.5}).json()
    assert strict["results"] == [] and strict["filtered_out"] == 5
    assert client.get("/search", params={"q": q, "min_relevance": 1.5}).status_code == 422


def test_sigmoid_is_overflow_safe():
    from searchengine.api import sigmoid

    assert sigmoid(1000) == 1.0 and sigmoid(-1000) == 0.0 and sigmoid(0) == 0.5
