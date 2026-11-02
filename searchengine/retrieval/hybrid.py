r"""Hybrid retrieval by Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))      (rank starts at 1; absent = no term)

Why ranks and not scores: BM25 scores are unbounded and their scale changes per
query (query length, IDF); cosine similarities live in [-1, 1]. A weighted sum of
raw scores needs per-query normalization and tuned weights; RRF needs neither.
k (default 60) damps the head: with k = 60, rank 1 vs rank 2 is 1/61 vs 1/62, so
a document both retrievers like (e.g. ranks 3 and 5) beats one that only a single
retriever puts first. That rewards agreement, which is what makes fusion robust.

The point of fusing BM25 with dense retrieval is *recall*: they fail on different
queries (exact rare terms vs paraphrases), so the union of their top-k covers more
relevant passages than either alone. Reranking then fixes the order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from searchengine.eval.metrics import rank_documents
from searchengine.retrieval.base import Retriever
from searchengine.types import Hit, Queries, Run


def rrf_fuse(rankings: list[dict[str, float]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for scores in rankings:
        for rank, doc in enumerate(rank_documents(scores), start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


class RRFHybrid(Retriever):
    def __init__(self, retrievers: list[Retriever], k_rrf: int = 60, depth: int = 1000, online_depth: int = 200):
        self.retrievers = retrievers
        self.k_rrf = k_rrf
        self.depth = depth  # per-retriever depth for offline evaluation (recall@1000 needs 1000)
        # Live queries need far fewer: measured 78 ms at depth 1000 vs ~9 + 37 ms for the parts.
        # A doc ranked below 200 by a retriever adds < 1/(60+200) to its fused score anyway.
        self.online_depth = online_depth
        self.name = "rrf(" + "+".join(r.name for r in retrievers) + ")"
        # Components are I/O-bound (Elasticsearch) or release the GIL (torch, FAISS),
        # so running them concurrently cuts single-query latency to ~max instead of sum.
        self._pool = ThreadPoolExecutor(max_workers=len(retrievers))

    def search(self, query: str, k: int = 1000) -> list[Hit]:
        depth = max(k, self.online_depth)
        futures = [self._pool.submit(r.search, query, depth) for r in self.retrievers]
        rankings = [{h.doc_id: h.score for h in f.result()} for f in futures]
        fused = rrf_fuse(rankings, self.k_rrf)
        return [Hit(fused[d], d) for d in rank_documents(fused)[:k]]

    def batch_search(self, queries: Queries, k: int = 1000) -> Run:
        depth = max(k, self.depth)
        runs = [r.batch_search(queries, depth) for r in self.retrievers]
        out: Run = {}
        for q in queries:
            fused = rrf_fuse([run.get(q, {}) for run in runs], self.k_rrf)
            out[q] = {d: fused[d] for d in rank_documents(fused)[:k]}
        return out

    def get_documents(self, doc_ids):
        """Delegate passage text to a component that can serve it (Elasticsearch)."""
        for r in self.retrievers:
            if hasattr(r, "get_documents"):
                return r.get_documents(doc_ids)
        raise AttributeError("no component retriever serves documents")
