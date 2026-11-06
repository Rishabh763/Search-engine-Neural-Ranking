r"""Bi-encoder: maps queries and passages *independently* to vectors in R^d.

    s(q, d) = <E(q), E(d)>        (cosine similarity when vectors are L2-normalized)

Independence is the whole point: every passage is encoded once, offline, and a
query costs one short forward pass plus a nearest-neighbour search. The price is
that query and passage never attend to each other (unlike the cross-encoder), so
it is less precise, and it compresses a passage into one vector, so it can blur
exact matches on rare terms. BM25 fails differently, which is why hybrid fusion helps.

Training (for reference, we use pretrained models): contrastive InfoNCE with
in-batch negatives,
    L = -log( exp(s(q, d+)/tau) / sum_j exp(s(q, d_j)/tau) ).

Some models expect instruction prefixes; using the wrong one (or none) silently
costs several points of recall, so known models carry their prefixes here.
"""

from __future__ import annotations

from pathlib import PurePath

import numpy as np

DEFAULT_DENSE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"

KNOWN_MODELS: dict[str, dict] = {
    # 6 layers, 384-d, trained on MS MARCO: fastest sensible default for CPU.
    "sentence-transformers/msmarco-MiniLM-L6-cos-v5": {"query_prefix": "", "passage_prefix": ""},
    "sentence-transformers/msmarco-MiniLM-L12-cos-v5": {"query_prefix": "", "passage_prefix": ""},
    "sentence-transformers/all-MiniLM-L6-v2": {"query_prefix": "", "passage_prefix": ""},
    # Stronger, 12 layers (2x the encoding cost); needs its query instruction.
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "intfloat/e5-small-v2": {"query_prefix": "query: ", "passage_prefix": "passage: "},
}


def default_dense_model() -> str:
    """The bi-encoder every script uses unless told otherwise: DENSE_MODEL env, else the built-in default.
    One source of truth, so encoding, indexing, evaluation and serving always agree."""
    import os

    return os.environ.get("DENSE_MODEL") or DEFAULT_DENSE_MODEL


def model_slug(model_name: str) -> str:
    return PurePath(model_name).name


class BiEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        device: str = "auto",
        batch_size: int = 128,
        max_length: int | None = 256,
        fp16: bool | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        from searchengine.ranking.cross_encoder import resolve_device

        known = KNOWN_MODELS.get(model_name, {})
        self.model_name = model_name
        self.query_prefix = known.get("query_prefix", "") if query_prefix is None else query_prefix
        self.passage_prefix = known.get("passage_prefix", "") if passage_prefix is None else passage_prefix
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.fp16 = (self.device == "cuda") if fp16 is None else fp16
        if self.fp16:
            self.model.half()
        if max_length:
            # MS MARCO passages average ~80 tokens; the cap bounds worst-case cost.
            self.model.max_seq_length = min(max_length, self.model.max_seq_length or max_length)
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        # sentence-transformers renamed this method; prefer the new name, fall back to the old.
        get = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(get())

    def config(self) -> dict:
        return {
            "model": self.model_name,
            "dim": self.dim,
            "normalized": True,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "max_length": self.model.max_seq_length,
        }

    def _encode(self, texts: list[str], progress: bool) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(self, queries: list[str], progress: bool = False) -> np.ndarray:
        return self._encode([self.query_prefix + q for q in queries], progress)

    def encode_passages(self, passages: list[str], progress: bool = False) -> np.ndarray:
        return self._encode([self.passage_prefix + p for p in passages], progress)
