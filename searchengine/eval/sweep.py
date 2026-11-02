"""Quality at every reranking depth from ONE reranked run.

Reranking the top-d candidates means: take the first-stage top-d, order them by
reranker score, keep the rest below in first-stage order. If we already reranked
the top-D (D >= d), every one of the top-d candidates already has its reranker
score, so the depth-d ranking can be assembled from cached scores without running
the model again. A whole quality-vs-depth curve costs zero extra model calls;
only latency needs timing (it scales ~linearly with d).
"""

from __future__ import annotations

from searchengine.eval.metrics import rank_documents
from searchengine.types import Run


def run_at_depth(first: Run, reranked: Run, depth: int, max_depth: int) -> Run:
    if depth > max_depth:
        raise ValueError(f"depth {depth} > reranked depth {max_depth}: those candidates have no reranker score")
    out: Run = {}
    for q, scores in first.items():
        ranked = rank_documents(scores)
        head, tail = ranked[:depth], ranked[depth:]
        cached = reranked.get(q, {})
        missing = [d for d in head if d not in cached]
        if missing:
            raise KeyError(f"query {q}: {len(missing)} of the top-{depth} have no cached reranker score")
        new = {d: cached[d] for d in head}
        floor = (min(new.values()) if new else 0.0) - 1.0
        for i, d in enumerate(tail):
            new[d] = floor - i
        out[q] = new
    return out
