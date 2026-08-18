"""
bm25.py

Dummy stub implementation of the BM25 ranking function.
Replace internals with a real implementation (or wrap rank_bm25 /
Elasticsearch / Pyserini) once the index is in place.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple


class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        """
        corpus: list of tokenized documents, e.g. [["term1", "term2"], ...]
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lens = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if corpus else 0.0
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus]
        self.idf: Dict[str, float] = {}
        self._compute_idf()

    def _compute_idf(self):
        """TODO: compute proper IDF (BM25 variant) over vocabulary."""
        df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1

        n_docs = len(self.corpus)
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query: List[str], doc_index: int) -> float:
        """Score a single document against a tokenized query."""
        score = 0.0
        doc_len = self.doc_lens[doc_index]
        freqs = self.doc_freqs[doc_index]

        for term in query:
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (tf * (self.k1 + 1)) / denom

        return score

    def rank(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """Return top_k (doc_index, score) pairs for the query."""
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    # Minimal smoke test with dummy data
    dummy_corpus = [
        ["search", "engine", "index", "documents"],
        ["bm25", "ranks", "documents", "by", "relevance"],
        ["neural", "rerankers", "use", "dense", "embeddings"],
    ]
    bm25 = BM25(dummy_corpus)
    results = bm25.rank(["documents", "relevance"], top_k=3)
    print("[stub] BM25 smoke test results:", results)
