"""
metrics.py

Dummy stub for IR evaluation metrics: Precision@k, Recall@k, MRR, NDCG.
"""

import math
from typing import Dict, List


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: List[str], relevance_map: Dict[str, float], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevance_map.get(doc_id, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved: List[str], relevance_map: Dict[str, float], k: int) -> float:
    """TODO: verify against a reference implementation before real use."""
    dcg = dcg_at_k(retrieved, relevance_map, k)
    ideal_order = sorted(relevance_map, key=lambda d: relevance_map[d], reverse=True)
    idcg = dcg_at_k(ideal_order, relevance_map, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


if __name__ == "__main__":
    retrieved = ["doc_2", "doc_1", "doc_3"]
    relevant = ["doc_1", "doc_3"]
    relevance_map = {"doc_1": 1.0, "doc_2": 0.0, "doc_3": 1.0}

    print("[stub] P@3:", precision_at_k(retrieved, relevant, 3))
    print("[stub] R@3:", recall_at_k(retrieved, relevant, 3))
    print("[stub] MRR:", mean_reciprocal_rank(retrieved, relevant))
    print("[stub] NDCG@3:", ndcg_at_k(retrieved, relevance_map, 3))
