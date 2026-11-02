"""Dense first-stage retriever: bi-encoder query embedding + FAISS search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from searchengine.retrieval.base import Retriever
from searchengine.types import Hit, Queries, Run


class DenseRetriever(Retriever):
    def __init__(self, encoder, index, doc_ids: list[str], name: str, nprobe: int | None = 32, ef_search: int | None = 128):
        self.encoder = encoder
        self.index = index
        self.doc_ids = doc_ids
        self.name = name
        self.nprobe = nprobe
        self.ef_search = ef_search

    @classmethod
    def load(
        cls,
        root: Path,
        kind: str = "flat",
        device: str = "auto",
        nprobe: int | None = 32,
        ef_search: int | None = 128,
        encoder=None,
    ) -> "DenseRetriever":
        import faiss

        from searchengine.dense.encoder import BiEncoder
        from searchengine.dense.index import EmbeddingStore

        root = Path(root)
        store = EmbeddingStore(root)
        path = root / f"{kind}.faiss"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run scripts/build_dense_index.py --type {kind}")
        enc_cfg = store.meta["encoder"]
        encoder = encoder or BiEncoder(
            enc_cfg["model"], device=device, max_length=enc_cfg.get("max_length"),
            query_prefix=enc_cfg.get("query_prefix"), passage_prefix=enc_cfg.get("passage_prefix"),
        )
        name = f"dense-{kind}:{root.name}"
        return cls(encoder, faiss.read_index(str(path)), store.ids, name, nprobe, ef_search)

    def _search_vectors(self, Q: np.ndarray, k: int) -> list[list[Hit]]:
        from searchengine.dense.index import set_search_params

        set_search_params(self.index, self.nprobe, self.ef_search, k)
        D, I = self.index.search(np.ascontiguousarray(Q, dtype=np.float32), k)
        return [[Hit(float(s), self.doc_ids[i]) for s, i in zip(drow, irow) if i >= 0] for drow, irow in zip(D, I)]

    def search(self, query: str, k: int = 1000) -> list[Hit]:
        if not query.strip():
            return []
        return self._search_vectors(self.encoder.encode_queries([query]), k)[0]

    def batch_search(self, queries: Queries, k: int = 1000, batch_size: int = 256) -> Run:
        qids = list(queries)
        Q = self.encoder.encode_queries([queries[q] for q in qids])
        run: Run = {}
        for start in range(0, len(qids), batch_size):
            for qid, hits in zip(qids[start : start + batch_size], self._search_vectors(Q[start : start + batch_size], k)):
                run[qid] = {h.doc_id: h.score for h in hits}
        return run

    def info(self) -> dict:
        return {"name": self.name, "ntotal": int(self.index.ntotal), "nprobe": self.nprobe, "ef_search": self.ef_search,
                "encoder": json.loads(json.dumps(self.encoder.config()))}
