"""
reranker.py

Dummy stub for combining/orchestrating BM25 candidate retrieval
with neural reranking into a single pipeline step.
"""

from typing import Dict, List

from ranking.neural_ranker import NeuralRanker


class Reranker:
    def __init__(self, neural_ranker: NeuralRanker = None, alpha: float = 0.5):
        """
        alpha: weight for blending bm25 score and neural score
               final_score = alpha * neural_score + (1 - alpha) * bm25_score
        """
        self.neural_ranker = neural_ranker or NeuralRanker()
        self.alpha = alpha

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        candidates: [{"doc_id": ..., "text": ..., "score": <bm25_score>}, ...]
        Returns reranked candidates with blended final_score.
        """
        results = []
        for cand in candidates:
            neural_score = self.neural_ranker.score(query, cand.get("text", ""))
            bm25_score = cand.get("score", 0.0)
            final_score = self.alpha * neural_score + (1 - self.alpha) * bm25_score
            results.append({**cand, "neural_score": neural_score, "final_score": final_score})

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    reranker = Reranker()
    dummy_candidates = [
        {"doc_id": "doc_1", "text": "search engines index documents", "score": 1.2},
        {"doc_id": "doc_2", "text": "bm25 ranks documents by relevance", "score": 2.1},
        {"doc_id": "doc_3", "text": "neural rerankers use dense embeddings", "score": 0.8},
    ]
    reranked = reranker.rerank("dense embeddings", dummy_candidates)
    print("[stub] Reranked results:", reranked)
