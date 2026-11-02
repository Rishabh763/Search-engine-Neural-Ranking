import pytest

from searchengine.docstore import DocStore
from searchengine.eval.metrics import rank_documents
from searchengine.eval.sweep import run_at_depth
from searchengine.pipeline import rerank_run
from searchengine.retrieval.base import Retriever
from searchengine.retrieval.bm25_local import LocalBM25
from searchengine.retrieval.hybrid import RRFHybrid, rrf_fuse
from searchengine.types import Hit


def test_run_at_depth_equals_reranking_at_that_depth(toy_collection, tmp_path, reverse_length_reranker):
    coll, docs, queries, _ = toy_collection
    store = DocStore.open(coll, tmp_path)
    first = LocalBM25.build(docs).batch_search(queries, k=100)
    deep, _ = rerank_run(first, queries, store, reverse_length_reranker, depth=50, progress=False)
    for d in (1, 5, 20, 50):
        direct, _ = rerank_run(first, queries, store, reverse_length_reranker, depth=d, progress=False)
        cached = run_at_depth(first, deep, d, max_depth=50)
        for q in queries:
            assert rank_documents(cached[q]) == rank_documents(direct[q])
    with pytest.raises(ValueError):
        run_at_depth(first, deep, 60, max_depth=50)


def test_rrf_formula():
    a = {"x": 9.0, "y": 5.0, "z": 1.0}  # ranks x1 y2 z3
    b = {"y": 0.9, "w": 0.8}  # ranks y1 w2
    f = rrf_fuse([a, b], k=60)
    assert f["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert f["x"] == pytest.approx(1 / 61)
    assert f["w"] == pytest.approx(1 / 62)
    assert rank_documents(f)[0] == "y"  # agreement beats a single first place


class Fixed(Retriever):
    def __init__(self, name, ranking):
        self.name = name
        self.ranking = ranking

    def search(self, query, k=1000):
        return [Hit(float(len(self.ranking) - i), d) for i, d in enumerate(self.ranking[:k])]


def test_hybrid_single_and_batch_agree():
    h = RRFHybrid([Fixed("a", ["d1", "d2", "d3"]), Fixed("b", ["d3", "d4"])], k_rrf=60, depth=10)
    assert h.name == "rrf(a+b)"
    single = [x.doc_id for x in h.search("q", k=4)]
    batch = rank_documents(h.batch_search({"q1": "q"}, k=4)["q1"])
    assert single == batch
    assert single[0] == "d3" and set(single) == {"d1", "d2", "d3", "d4"}
    assert len(h.search("q", k=2)) == 2
    with pytest.raises(AttributeError):
        h.get_documents(["d1"])  # neither component serves text


def test_hybrid_online_depth_limits_component_fetch():
    calls = []

    class Spy(Fixed):
        def search(self, query, k=1000):
            calls.append(k)
            return super().search(query, k)

    h = RRFHybrid([Spy("a", [f"d{i}" for i in range(500)]), Spy("b", ["d1"])], depth=1000, online_depth=200)
    h.search("q", k=100)
    assert calls == [200, 200]  # not 1000
    calls.clear()
    h.search("q", k=300)
    assert calls == [300, 300]  # never below k
