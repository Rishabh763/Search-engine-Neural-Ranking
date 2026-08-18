"""
embedder.py

Dummy stub for producing dense embeddings for queries/documents.
Replace with a real model (SentenceTransformers, OpenAI/Anthropic
embeddings API, custom encoder, etc.).
"""

import hashlib
from typing import List

EMBEDDING_DIM = 128


class Embedder:
    def __init__(self, model_name: str = "dummy-embedder-v0", dim: int = EMBEDDING_DIM):
        self.model_name = model_name
        self.dim = dim

    def _deterministic_pseudo_embedding(self, text: str) -> List[float]:
        """
        Placeholder: hash-based pseudo-embedding so the pipeline is
        runnable end-to-end without a real model. NOT semantically
        meaningful — replace before using for actual ranking.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat/trim the digest bytes to fill the embedding dimension
        values = [(digest[i % len(digest)] / 255.0) for i in range(self.dim)]
        return values

    def embed(self, text: str) -> List[float]:
        """TODO: replace with real model inference."""
        return self._deterministic_pseudo_embedding(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


if __name__ == "__main__":
    embedder = Embedder()
    vec = embedder.embed("search engine ranking")
    print(f"[stub] Embedding dim: {len(vec)}, first 5 values: {vec[:5]}")
