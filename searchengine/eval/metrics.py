r"""Ranking metrics, implemented from their definitions and checked against trec_eval.

Notation for one query: ranked list d_1, d_2, ...; graded label rel(d) from qrels
(unjudged = 0); R = number of documents with rel >= t (t = relevance threshold).

    RR@k       = 1 / min{ i <= k : rel(d_i) >= t }        (0 if none in top k)
    Recall@k   = |{ i <= k : rel(d_i) >= t }| / R
    P@k        = |{ i <= k : rel(d_i) >= t }| / k          (divides by k even if fewer retrieved)
    AP@k       = (1/R) * sum_{i <= k, rel(d_i) >= t} P@i
    DCG@k      = sum_{i=1..k} g(rel(d_i)) / log2(i + 1)
    nDCG@k     = DCG@k / IDCG@k                            (IDCG: DCG of the ideal ordering of all judged docs)

Gain g: "linear" g(r) = r (what trec_eval's ndcg_cut uses, so it is the default
for comparability with published TREC DL numbers), or "exponential"
g(r) = 2^r - 1 (Burges et al.; rewards highly relevant docs more steeply).

Conventions, chosen to match `trec_eval -c`:
- Ranking: sort by score descending; ties broken by doc id descending.
- Query set: every query in qrels that has at least one relevant document.
  A query missing from the run scores 0 (it is not silently dropped).
- nDCG uses raw graded labels; the threshold t only affects binary metrics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from searchengine.types import Qrels, Run

_METRIC_RE = re.compile(r"^(mrr|recall|precision|p|ndcg|map)@(\d+)$")


def rank_documents(scores: dict[str, float]) -> list[str]:
    """Doc ids by descending score, ties by descending doc id (trec_eval order)."""
    return [d for d, _ in sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)]


def parse_metric(metric: str) -> tuple[str, int]:
    m = _METRIC_RE.match(metric.lower().strip())
    if not m:
        raise ValueError(f"Bad metric {metric!r}; expected e.g. mrr@10, recall@100, ndcg@10, precision@5, map@1000")
    name, k = m.group(1), int(m.group(2))
    if k <= 0:
        raise ValueError("k must be positive")
    return ("precision" if name == "p" else name), k


# ---------- per-query metric functions (ranked list, qrels for that query) ----------


def reciprocal_rank(ranked: list[str], rels: dict[str, int], k: int, t: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) >= t:
            return 1.0 / (i + 1)
    return 0.0


def recall(ranked: list[str], rels: dict[str, int], k: int, t: int) -> float:
    R = sum(1 for r in rels.values() if r >= t)
    if R == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rels.get(d, 0) >= t) / R


def precision(ranked: list[str], rels: dict[str, int], k: int, t: int) -> float:
    return sum(1 for d in ranked[:k] if rels.get(d, 0) >= t) / k


def average_precision(ranked: list[str], rels: dict[str, int], k: int, t: int) -> float:
    R = sum(1 for r in rels.values() if r >= t)
    if R == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if rels.get(d, 0) >= t:
            hits += 1
            total += hits / i
    return total / R


def _gain(r: int, kind: str) -> float:
    if r <= 0:
        return 0.0
    return float(r) if kind == "linear" else float(2**r - 1)


def ndcg(ranked: list[str], rels: dict[str, int], k: int, gain: str = "linear") -> float:
    dcg = sum(_gain(rels.get(d, 0), gain) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(_gain(r, gain) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------- aggregate ----------


@dataclass
class EvalResult:
    mean: dict[str, float]
    per_query: dict[str, dict[str, float]] = field(repr=False)
    n_queries: int
    n_missing_from_run: int
    rel_threshold: int
    ndcg_gain: str

    def format(self) -> str:
        cells = "  ".join(f"{m}={v:.4f}" for m, v in self.mean.items())
        extra = f"  ({self.n_missing_from_run} queries absent from run, scored 0)" if self.n_missing_from_run else ""
        return f"n={self.n_queries}  {cells}{extra}"


def evaluate(
    run: Run,
    qrels: Qrels,
    metrics: list[str] | tuple[str, ...],
    rel_threshold: int = 1,
    ndcg_gain: str = "linear",
) -> EvalResult:
    if ndcg_gain not in ("linear", "exponential"):
        raise ValueError("ndcg_gain must be 'linear' or 'exponential'")
    parsed = [(m, *parse_metric(m)) for m in metrics]

    # Queries that can be scored: at least one relevant doc at the threshold.
    # (nDCG only needs rel > 0, but using one query set for all metrics keeps
    # the per-query vectors aligned for significance testing.)
    qids = sorted(q for q, rels in qrels.items() if any(r >= rel_threshold for r in rels.values()))

    per_query: dict[str, dict[str, float]] = {m: {} for m, _, _ in parsed}
    missing = 0
    for q in qids:
        rels = qrels[q]
        if q not in run:
            missing += 1
        ranked = rank_documents(run.get(q, {}))
        for label, name, k in parsed:
            if name == "mrr":
                v = reciprocal_rank(ranked, rels, k, rel_threshold)
            elif name == "recall":
                v = recall(ranked, rels, k, rel_threshold)
            elif name == "precision":
                v = precision(ranked, rels, k, rel_threshold)
            elif name == "map":
                v = average_precision(ranked, rels, k, rel_threshold)
            else:
                v = ndcg(ranked, rels, k, ndcg_gain)
            per_query[label][q] = v

    mean = {m: (sum(per_query[m].values()) / len(qids) if qids else 0.0) for m, _, _ in parsed}
    return EvalResult(mean, per_query, len(qids), missing, rel_threshold, ndcg_gain)
