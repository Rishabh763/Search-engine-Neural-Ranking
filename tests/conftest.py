"""Shared fixtures for Phase 2 tests: a toy collection on disk, and stub rerankers."""

import numpy as np
import pytest

from searchengine.data.collection import Collection, write_collection
from searchengine.data.toy import make_toy_collection
from searchengine.ranking.base import Reranker


@pytest.fixture(scope="session")
def toy_collection(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "toy"
    docs, queries, qrels = make_toy_collection(n_docs=600, n_topics=20, n_queries=30)
    write_collection(root, docs, {"toy": (queries, qrels)})
    return Collection(root), docs, queries, qrels


class ReverseLengthReranker(Reranker):
    """Deterministic stand-in for a model: shorter passages score higher."""

    name = "fake:reverse-length"

    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, len(passages)))
        return np.array([-len(p) for p in passages], dtype=np.float32)


class OracleReranker(Reranker):
    """Scores passages by their qrels label: reranking with it must not hurt nDCG."""

    name = "fake:oracle"

    def __init__(self, qrels, queries, docs):
        text_to_id = {d.text: d.id for d in docs}
        self.rels = {queries[q]: qrels[q] for q in qrels}
        self.text_to_id = text_to_id

    def score(self, query, passages):
        rels = self.rels.get(query, {})
        return np.array([rels.get(self.text_to_id.get(p, ""), 0) for p in passages], dtype=np.float32)


@pytest.fixture
def reverse_length_reranker():
    return ReverseLengthReranker()
