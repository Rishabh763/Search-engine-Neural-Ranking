"""
neural_ranker.py

Dummy stub for a neural (dense) ranking model that scores
query-document pairs using embeddings from embedder.py.
"""

import math
from typing import Dict, List

from ranking.embedder import Embedder


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class NeuralRanker:
    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder or Embedder()

    def score(self, query: str, doc_text: str) -> float:
        """TODO: replace cosine similarity with a trained cross-encoder / bi-encoder scorer."""
        query_vec = self.embedder.embed(query)
        doc_vec = self.embedder.embed(doc_text)
        return cosine_similarity(query_vec, doc_vec)

    def rank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        candidates: [{"doc_id": ..., "text": ...}, ...]
        Returns candidates sorted by neural score, truncated to top_k.
        """
        scored = []
        for cand in candidates:
            s = self.score(query, cand.get("text", ""))
            scored.append({**cand, "neural_score": s})

        scored.sort(key=lambda x: x["neural_score"], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    ranker = NeuralRanker()
    dummy_candidates = [
        {"doc_id": "doc_1", "text": "search engines index documents"},
        {"doc_id": "doc_2", "text": "bm25 ranks documents by relevance"},
        {"doc_id": "doc_3", "text": "neural rerankers use dense embeddings"},
    ]
    ranked = ranker.rank("dense embeddings for reranking", dummy_candidates)
    print("[stub] Neural ranking results:", ranked)
