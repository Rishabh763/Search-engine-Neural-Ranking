"""Near-duplicate collapsing, on the shapes of copies seen in MS MARCO results."""

import pytest

from searchengine.dedup import collapse, containment, shingles
from searchengine.pipeline import SearchPipeline
from searchengine.retrieval.base import Retriever
from searchengine.types import Document, Hit

FULL = ("Vaccines work by stimulating our immune system to produce antibodies (substances produced by the body "
        "to fight disease) without actually infecting us with the disease. They trigger the immune system to "
        "produce its own antibodies, as though the body has been infected with a disease. This is called active "
        "immunity.")
TRUNCATED = FULL[: FULL.index(" This is called")]  # same passage, last sentence cut
FLATTENED = ("vaccines work by stimulating our immune system to produce antibodies substances produced by the body "
             "to fight disease without actually infecting us with the disease they trigger the immune system to "
             "produce its own antibodies as though the body has been infected with a disease this is called active "
             "immunity")
NO_SPACE = FULL.replace("disease. This", "disease.This")
OTHER = ("Vaccines are like a training course for the immune system. They prepare the body to fight disease "
         "without exposing it to disease symptoms.")
UNRELATED = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."


def test_shingles_ignore_case_punctuation_and_spacing():
    assert shingles(FULL) == shingles(FLATTENED)
    assert containment(shingles(FULL), shingles(NO_SPACE)) >= 0.95
    assert shingles("") == frozenset()
    assert shingles("Two words") == frozenset([("two", "words")])


def test_containment_catches_truncated_copies_that_jaccard_misses():
    a, b = shingles(FULL), shingles(TRUNCATED)
    jaccard = len(a & b) / len(a | b)
    assert containment(a, b) == 1.0  # every shingle of the shorter copy is in the full one
    assert jaccard < 0.9  # length difference alone would hide the duplicate from a Jaccard test
    assert containment(shingles(FULL), shingles(OTHER)) < 0.3  # same topic, different passage
    assert containment(shingles(FULL), shingles(UNRELATED)) == 0.0
    assert containment(frozenset(), a) == 0.0


def test_collapse_keeps_the_best_ranked_copy_and_groups_the_rest():
    texts = [FULL, OTHER, TRUNCATED, FLATTENED, NO_SPACE, UNRELATED]
    c = collapse(texts)
    assert c.kept == [0, 1, 5]
    assert c.duplicates == {0: [2, 3, 4]}
    assert c.n_dropped == 3 and c.examined == 6


def test_collapse_stops_at_k_and_threshold_is_respected():
    texts = [FULL, TRUNCATED, OTHER, UNRELATED, FLATTENED]
    c = collapse(texts, k=2)
    assert c.kept == [0, 2] and c.examined == 3  # FLATTENED is never looked at
    assert collapse([FULL, OTHER]).kept == [0, 1]  # same topic, different passage: kept apart at 0.8
    assert collapse([FULL, OTHER], threshold=0.1).kept == [0]  # a loose threshold merges topics, not copies
    assert collapse([FULL, TRUNCATED], threshold=1.01).kept == [0, 1]  # above 1 nothing is a duplicate


class _ListRetriever(Retriever):
    name = "list"

    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k=1000):
        return [Hit(float(len(self.ids) - i), d) for i, d in enumerate(self.ids[:k])]


class _Docs:
    def __init__(self, texts):
        self.texts = texts

    def get_documents(self, ids):
        return {d: Document(d, self.texts[d]) for d in ids if d in self.texts}


@pytest.fixture
def dup_pipeline():
    texts = {"a": FULL, "b": TRUNCATED, "c": OTHER, "d": FLATTENED, "e": UNRELATED, "f": "Paris is in France."}
    return SearchPipeline(_ListRetriever(list(texts)), _Docs(texts), None)


def test_pipeline_dedup_fills_k_with_distinct_passages(dup_pipeline):
    plain, t0 = dup_pipeline.search("vaccines", k=3, candidates=6, rerank=False)
    assert [r.doc_id for r in plain] == ["a", "b", "c"] and t0.dedup_ms == 0.0
    res, t = dup_pipeline.search("vaccines", k=3, candidates=6, rerank=False, dedup=True)
    assert [r.doc_id for r in res] == ["a", "c", "e"]  # b and d are copies of a
    assert res[0].duplicates == ["b", "d"] and res[1].duplicates == [] and res[2].duplicates == []
    assert [r.rank for r in res] == [1, 2, 3]
    assert [r.first_stage_rank for r in res] == [1, 3, 5]
    assert t.dedup_ms >= 0 and t.total_ms >= t.dedup_ms


def test_pipeline_dedup_after_reranking(dup_pipeline, reverse_length_reranker):
    dup_pipeline.reranker = reverse_length_reranker  # shortest passage first: f, e, c, b, d, a
    res, _ = dup_pipeline.search("vaccines", k=5, candidates=6, rerank=True, dedup=True)
    # the truncated copy b now ranks best among the copies, so it represents them; only
    # 4 distinct passages exist among the 6 candidates, so k=5 returns 4
    assert [r.doc_id for r in res] == ["f", "e", "c", "b"]
    assert res[3].duplicates == ["d", "a"]  # in rank order


def test_api_reports_collapsed_duplicates(dup_pipeline):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from searchengine.api import create_app

    with TestClient(create_app(dup_pipeline, {"collection": "t", "backend": "list"})) as client:
        on = client.get("/search", params={"q": "vaccines", "k": 3, "candidates": 6, "rerank": False}).json()
        off = client.get("/search", params={"q": "vaccines", "k": 3, "candidates": 6, "rerank": False,
                                            "dedup": False}).json()
    assert on["dedup"] is True and on["collapsed"] == 2  # on by default
    assert [r["id"] for r in on["results"]] == ["a", "c", "e"]
    assert on["results"][0]["duplicates"] == ["b", "d"]
    assert "dedup_ms" in on["timings"]
    assert off["dedup"] is False and off["collapsed"] == 0
    assert [r["id"] for r in off["results"]] == ["a", "b", "c"]
