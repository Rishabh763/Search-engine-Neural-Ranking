import pytest

from searchengine.docstore import DocStore
from searchengine.eval.metrics import evaluate, rank_documents
from searchengine.pipeline import SearchPipeline, rerank_run
from searchengine.retrieval.bm25_local import LocalBM25

from conftest import OracleReranker


@pytest.fixture
def setup(toy_collection, tmp_path):
    coll, docs, queries, qrels = toy_collection
    return coll, docs, queries, qrels, DocStore.open(coll, tmp_path), LocalBM25.build(docs)


def test_docstore_random_access_and_cache(toy_collection, tmp_path):
    coll, docs, _, _ = toy_collection
    store = DocStore.open(coll, tmp_path)
    want = [docs[5], docs[599], docs[0]]
    got = store.get_documents([d.id for d in want] + ["missing-id"])
    assert set(got) == {d.id for d in want}
    for d in want:
        assert got[d.id] == d
    store.close()
    # Second open loads the cached offsets instead of rescanning.
    again = DocStore.open(coll, tmp_path)
    assert again.ids == store.ids and (again.offsets == store.offsets).all()


def test_search_pipeline_reranks_candidates_and_times_stages(setup, reverse_length_reranker):
    coll, docs, queries, _, store, bm25 = setup
    pipe = SearchPipeline(bm25, store, reverse_length_reranker)
    q = next(iter(queries.values()))
    results, t = pipe.search(q, k=5, candidates=50)
    assert len(results) == 5
    assert reverse_length_reranker.calls == [(q, 50)]
    lens = [len(r.text) for r in results]
    assert lens == sorted(lens)  # fake reranker: shorter first
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert all(r.rerank_score is not None and r.score == r.rerank_score for r in results)
    assert t.total_ms >= t.retrieve_ms + t.rerank_ms - 0.1

    plain, t2 = pipe.search(q, k=5, rerank=False)
    bm25_ids = [h.doc_id for h in bm25.search(q, 5)]
    assert [r.doc_id for r in plain] == bm25_ids
    assert all(r.rerank_score is None for r in plain) and t2.rerank_ms < 1


def test_pipeline_without_reranker_ignores_rerank_flag(setup):
    coll, docs, queries, _, store, bm25 = setup
    pipe = SearchPipeline(bm25, store, None)
    results, _ = pipe.search(next(iter(queries.values())), k=3, rerank=True)
    assert len(results) == 3 and results[0].rerank_score is None


def test_rerank_run_keeps_tail_below_head(setup, reverse_length_reranker):
    coll, docs, queries, qrels, store, bm25 = setup
    first = bm25.batch_search(queries, k=100)
    run, lat = rerank_run(first, queries, store, reverse_length_reranker, depth=10, progress=False)
    assert len(lat) == len(queries)
    for q in queries:
        before, after = rank_documents(first[q]), rank_documents(run[q])
        assert set(before) == set(after)  # nothing lost: recall@100 unchanged
        assert set(after[:10]) == set(before[:10])  # head = same candidates, reordered
        assert after[10:] == before[10:]  # tail order untouched
    m_first = evaluate(first, qrels, ["recall@100"]).mean["recall@100"]
    m_rr = evaluate(run, qrels, ["recall@100"]).mean["recall@100"]
    assert m_first == pytest.approx(m_rr)


def test_oracle_reranker_never_hurts(setup):
    coll, docs, queries, qrels, store, bm25 = setup
    first = bm25.batch_search(queries, k=100)
    run, _ = rerank_run(first, queries, store, OracleReranker(qrels, queries, docs), depth=50, progress=False)
    before = evaluate(first, qrels, ["ndcg@10"]).mean["ndcg@10"]
    after = evaluate(run, qrels, ["ndcg@10"]).mean["ndcg@10"]
    assert after >= before
