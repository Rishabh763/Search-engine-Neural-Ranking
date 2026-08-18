"""
evaluate_bm25.py

Dummy stub for evaluating the BM25 retrieval pipeline against a
labeled query/relevance judgment set.
"""

import argparse
import json
from typing import Dict, List

from evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from retrieval.candidate_retriever import CandidateRetriever

DEFAULT_QRELS_PATH = "data/sample/qrels.json"


def load_qrels(path: str) -> List[Dict]:
    """
    Expected format:
    [{"query": "...", "relevant_doc_ids": ["doc_1", "doc_3"]}, ...]
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[stub] Qrels file not found: {path}. Using dummy qrels.")
        return [
            {"query": "search engine ranking", "relevant_doc_ids": ["doc_1", "doc_2"]},
        ]


def evaluate(retriever: CandidateRetriever, qrels: List[Dict], k: int = 10):
    p_scores, r_scores, mrr_scores = [], [], []

    for item in qrels:
        query = item["query"]
        relevant = item["relevant_doc_ids"]

        candidates = retriever.retrieve(query, top_k=k)
        retrieved_ids = [c["doc_id"] for c in candidates]

        p_scores.append(precision_at_k(retrieved_ids, relevant, k))
        r_scores.append(recall_at_k(retrieved_ids, relevant, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, relevant))

    n = len(qrels) or 1
    print(f"[stub] Avg P@{k}: {sum(p_scores) / n:.4f}")
    print(f"[stub] Avg R@{k}: {sum(r_scores) / n:.4f}")
    print(f"[stub] Avg MRR: {sum(mrr_scores) / n:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BM25 retrieval.")
    parser.add_argument("--qrels", type=str, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dummy_corpus = [
        ["search", "engine", "index", "documents"],
        ["bm25", "ranks", "documents", "by", "relevance"],
        ["neural", "rerankers", "use", "dense", "embeddings"],
    ]
    dummy_ids = ["doc_1", "doc_2", "doc_3"]
    retriever = CandidateRetriever(dummy_corpus, dummy_ids)

    qrels = load_qrels(args.qrels)
    evaluate(retriever, qrels, args.k)


if __name__ == "__main__":
    main()
