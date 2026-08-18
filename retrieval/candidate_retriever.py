"""
candidate_retriever.py

Dummy stub that ties query_processor + bm25 together to fetch a
candidate set of documents for downstream (neural) reranking.
"""

from typing import Dict, List

from retrieval.bm25 import BM25
from retrieval.query_processor import QueryProcessor


class CandidateRetriever:
    def __init__(self, corpus: List[List[str]], doc_ids: List[str]):
        """
        corpus: tokenized documents aligned by index with doc_ids
        doc_ids: external doc_id for each corpus entry
        """
        self.doc_ids = doc_ids
        self.bm25 = BM25(corpus)
        self.query_processor = QueryProcessor()

    def retrieve(self, raw_query: str, top_k: int = 100) -> List[Dict]:
        """Return top_k candidate docs as [{"doc_id": ..., "score": ...}, ...]."""
        query_tokens = self.query_processor.process(raw_query)
        ranked = self.bm25.rank(query_tokens, top_k=top_k)

        return [
            {"doc_id": self.doc_ids[idx], "score": score}
            for idx, score in ranked
        ]


if __name__ == "__main__":
    dummy_corpus = [
        ["search", "engine", "index", "documents"],
        ["bm25", "ranks", "documents", "by", "relevance"],
        ["neural", "rerankers", "use", "dense", "embeddings"],
    ]
    dummy_ids = ["doc_1", "doc_2", "doc_3"]

    retriever = CandidateRetriever(dummy_corpus, dummy_ids)
    candidates = retriever.retrieve("relevant documents", top_k=3)
    print("[stub] Candidates:", candidates)
