"""
evaluate_neural.py

Dummy stub for evaluating the neural reranking pipeline against a
labeled query/relevance judgment set.
"""

import argparse
import json
from typing import Dict, List

from evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from ranking.reranker import Reranker

DEFAULT_QRELS_PATH = "data/sample/qrels.json"


def load_qrels(path: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[stub] Qrels file not found: {path}. Using dummy qrels.")
        return [
            {"query": "dense embeddings", "relevant_doc_ids": ["doc_3"]},
        ]


def evaluate(reranker: Reranker, qrels: List[Dict], candidate_pool: List[Dict], k: int = 10):
    p_scores, r_scores, mrr_scores, ndcg_scores = [], [], [], []

    for item in qrels:
        query = item["query"]
        relevant = item["relevant_doc_ids"]
        relevance_map = {doc_id: 1.0 for doc_id in relevant}

        reranked = reranker.rerank(query, candidate_pool, top_k=k)
        retrieved_ids = [c["doc_id"] for c in reranked]

        p_scores.append(precision_at_k(retrieved_ids, relevant, k))
        r_scores.append(recall_at_k(retrieved_ids, relevant, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, relevant))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevance_map, k))

    n = len(qrels) or 1
    print(f"[stub] Avg P@{k}: {sum(p_scores) / n:.4f}")
    print(f"[stub] Avg R@{k}: {sum(r_scores) / n:.4f}")
    print(f"[stub] Avg MRR: {sum(mrr_scores) / n:.4f}")
    print(f"[stub] Avg NDCG@{k}: {sum(ndcg_scores) / n:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural reranking.")
    parser.add_argument("--qrels", type=str, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dummy_candidates = [
        {"doc_id": "doc_1", "text": "search engines index documents", "score": 1.2},
        {"doc_id": "doc_2", "text": "bm25 ranks documents by relevance", "score": 2.1},
        {"doc_id": "doc_3", "text": "neural rerankers use dense embeddings", "score": 0.8},
    ]

    reranker = Reranker()
    qrels = load_qrels(args.qrels)
    evaluate(reranker, qrels, dummy_candidates, args.k)


if __name__ == "__main__":
    main()
