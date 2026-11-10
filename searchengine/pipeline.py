"""Retrieve-then-rerank: the two-stage search pipeline.

    query --> first-stage retriever (BM25, dense, or hybrid) --> top `candidates`
          --> fetch passage text --> reranker scores each (query, passage) pair
          --> [optional] collapse near-duplicates (searchengine.dedup)
          --> top k by reranker score

Every stage is timed separately, because "the search takes 120 ms" is useless
for optimization while "retrieve 7 ms, fetch 2 ms, rerank 110 ms" says exactly
where to look.

`rerank_run` applies the same logic offline to a saved first-stage run, for
evaluation: the top `depth` candidates are reordered by the reranker, and the
rest keep their first-stage order *below* them. Keeping the tail means
recall@1000 is unchanged and metrics isolate the effect of reranking.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from searchengine.dedup import DEFAULT_THRESHOLD, collapse
from searchengine.docstore import DocumentSource
from searchengine.eval.metrics import rank_documents
from searchengine.ranking.base import Reranker
from searchengine.retrieval.base import Retriever
from searchengine.types import Document, Queries, Run


def passage_text(d: Document) -> str:
    return f"{d.title} {d.text}" if d.title else d.text


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    score: float
    first_stage_score: float
    first_stage_rank: int
    rerank_score: float | None
    title: str
    text: str
    duplicates: list[str] = field(default_factory=list)  # ids of near-duplicates collapsed into this result


@dataclass
class Timings:
    retrieve_ms: float
    fetch_ms: float
    rerank_ms: float
    total_ms: float
    dedup_ms: float = 0.0


class SearchPipeline:
    def __init__(self, retriever: Retriever, docs: DocumentSource, reranker: Reranker | None = None):
        self.retriever = retriever
        self.docs = docs
        self.reranker = reranker

    def search(self, query: str, k: int = 10, candidates: int = 100, rerank: bool = True,
               dedup: bool = False, dedup_threshold: float = DEFAULT_THRESHOLD) -> tuple[list[SearchResult], Timings]:
        """`dedup`: collapse near-duplicate passages after ranking, so the top k are k
        distinct passages (fewer only if the candidates run out). Without reranking,
        `candidates` passages are still retrieved so there is something to refill from."""
        rerank = rerank and self.reranker is not None
        t0 = time.perf_counter()
        hits = self.retriever.search(query, k=candidates if (rerank or dedup) else k)
        t1 = time.perf_counter()
        ids = [h.doc_id for h in hits]
        docs = self.docs.get_documents(ids)
        hits = [h for h in hits if h.doc_id in docs]  # guard against index/doc-store drift
        t2 = time.perf_counter()

        ce: np.ndarray | None = None
        order = list(range(len(hits)))
        if rerank and hits:
            ce = self.reranker.score(query, [passage_text(docs[h.doc_id]) for h in hits])
            # stable sort: ties keep first-stage order
            order = sorted(order, key=lambda i: -float(ce[i]))
        t3 = time.perf_counter()

        dups: dict[int, list[int]] = {}
        if dedup and order:
            c = collapse([passage_text(docs[hits[i].doc_id]) for i in order], k=k, threshold=dedup_threshold)
            dups = {order[a]: [order[b] for b in bs] for a, bs in c.duplicates.items()}
            order = [order[a] for a in c.kept]
        t4 = time.perf_counter()

        results = []
        for rank, i in enumerate(order[:k], start=1):
            h, d = hits[i], docs[hits[i].doc_id]
            rs = float(ce[i]) if ce is not None else None
            results.append(
                SearchResult(rank, h.doc_id, rs if rs is not None else h.score, h.score, i + 1, rs, d.title, d.text,
                             [hits[j].doc_id for j in dups.get(i, [])])
            )
        ms = lambda a, b: round((b - a) * 1000, 2)  # noqa: E731
        return results, Timings(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t4), ms(t3, t4))


def rerank_run(
    run: Run,
    queries: Queries,
    docs: DocumentSource,
    reranker: Reranker,
    depth: int,
    progress: bool = True,
) -> tuple[Run, list[float]]:
    """Rerank the top `depth` of each query's first-stage ranking.

    Returns the new run and per-query rerank latency in ms (text fetch + scoring).
    """
    from tqdm import tqdm

    out: Run = {}
    latencies: list[float] = []
    items = [(q, run.get(q, {})) for q in queries]
    for qid, scores in tqdm(items, desc=f"rerank@{depth}", unit="q", disable=not progress):
        ranked = rank_documents(scores)
        head, tail = ranked[:depth], ranked[depth:]
        t0 = time.perf_counter()
        texts = docs.get_documents(head)
        head = [d for d in head if d in texts]
        ce = reranker.score(queries[qid], [passage_text(texts[d]) for d in head]) if head else np.zeros(0)
        latencies.append((time.perf_counter() - t0) * 1000)
        new = {d: float(s) for d, s in zip(head, ce)}
        # Tail stays below every reranked doc, in its original order.
        floor = (min(new.values()) if new else 0.0) - 1.0
        for i, d in enumerate(tail):
            new[d] = floor - i
        out[qid] = new
    return out, latencies


def timings_dict(t: Timings) -> dict:
    return asdict(t)
