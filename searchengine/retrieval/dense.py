"""Dense first-stage retriever: bi-encoder query embedding + FAISS search,
optionally refined with exact scores (see searchengine.dense.index.refine_topk)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from searchengine.retrieval.base import Retriever
from searchengine.types import Hit, Queries, Run


class DenseRetriever(Retriever):
    """`refine` = k' > 0: fetch max(k, k') ANN candidates, rescore them exactly from the
    stored vectors, keep the top k. Needs `vectors` (a VectorRows over the embedding shards).
    Refining exactly k candidates only reorders them: it raises recall@10 but not
    recall@k; k' > k also recovers true neighbours that PQ ranked below k."""

    def __init__(self, encoder, index, doc_ids: list[str], name: str, nprobe: int | None = 32, ef_search: int | None = 128,
                 refine: int = 0, vectors=None):
        if refine and vectors is None:
            raise ValueError("refine > 0 needs the stored vectors (VectorRows)")
        self.encoder = encoder
        self.index = index
        self.doc_ids = doc_ids
        self.name = name
        self.nprobe = nprobe
        self.ef_search = ef_search
        self.refine = int(refine or 0)
        self.vectors = vectors

    @classmethod
    def load(
        cls,
        root: Path,
        kind: str = "flat",
        device: str = "auto",
        nprobe: int | None = 32,
        ef_search: int | None = 128,
        encoder=None,
        refine: int = 0,
    ) -> "DenseRetriever":
        import faiss

        from searchengine.dense.encoder import BiEncoder
        from searchengine.dense.index import EmbeddingStore, VectorRows

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
        name = f"dense-{kind}:{root.name}" + (f"+refine{refine}" if refine else "")
        vectors = VectorRows(store) if refine else None
        return cls(encoder, faiss.read_index(str(path)), store.ids, name, nprobe, ef_search, refine, vectors)

    def _search_vectors(self, Q: np.ndarray, k: int) -> list[list[Hit]]:
        from searchengine.dense.index import refine_topk, set_search_params

        Q = np.ascontiguousarray(Q, dtype=np.float32)
        depth = max(k, self.refine) if self.refine else k
        set_search_params(self.index, self.nprobe, self.ef_search, depth)
        D, I = self.index.search(Q, depth)
        if self.refine:
            D, I = refine_topk(Q, I, self.vectors, k)
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
                "refine": self.refine,
                "encoder": json.loads(json.dumps(self.encoder.config()))}


class DenseExactTorch(Retriever):
    """Exact (brute-force) dense search with PyTorch, corpus held on the GPU in fp16.

    8.8M x 384 fp16 = 6.8 GB fits a 16 GB T4, and a query batch against the corpus is
    one matrix multiply: exact results at GPU speed, no index approximation. Used on
    Kaggle for full-corpus evaluation and hard-negative mining, and as the ground
    truth that ANN indexes are measured against. Scores are blockwise to bound memory:
    (queries x block) fp16 scores, then a running top-k merge.
    """

    def __init__(self, encoder, emb, doc_ids: list[str], name: str, block: int = 1_000_000):
        self.encoder = encoder
        self.emb = emb  # torch tensor [N, d], fp16 on GPU (fp32 on CPU)
        self.doc_ids = doc_ids
        self.name = name
        self.block = block

    @classmethod
    def load(cls, root: Path, device: str = "auto", encoder=None, block: int = 1_000_000) -> "DenseExactTorch":
        import torch

        from searchengine.dense.encoder import BiEncoder
        from searchengine.dense.index import EmbeddingStore
        from searchengine.ranking.cross_encoder import resolve_device

        root = Path(root)
        store = EmbeddingStore(root)
        dev = resolve_device(device)
        dtype = torch.float16 if dev == "cuda" else torch.float32  # CPU matmul in fp16 is slow
        # Fill one preallocated tensor shard by shard: torch.cat of GPU shards would briefly
        # hold two copies (2 x 6.8 GB on a 15 GB T4 for the full corpus).
        emb = torch.empty((store.n, store.dim), dtype=dtype, device=dev)
        off = 0
        for sh in store.meta["shards"]:
            part = torch.from_numpy(np.load(root / sh["file"]))
            emb[off : off + len(part)] = part.to(device=dev, dtype=dtype)
            off += len(part)
            del part
        enc_cfg = store.meta["encoder"]
        encoder = encoder or BiEncoder(
            enc_cfg["model"], device=dev, max_length=enc_cfg.get("max_length"),
            query_prefix=enc_cfg.get("query_prefix"), passage_prefix=enc_cfg.get("passage_prefix"),
        )
        return cls(encoder, emb, store.ids, f"dense-exact:{root.name}", block)

    def search_vectors(self, Q: np.ndarray, k: int, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k rows and scores for query vectors Q [nq, d]."""
        import torch

        n = self.emb.shape[0]
        k = min(k, n)
        all_s, all_i = [], []
        for qs in range(0, len(Q), batch):
            q = torch.from_numpy(np.ascontiguousarray(Q[qs : qs + batch])).to(self.emb.device, self.emb.dtype)
            best_s = best_i = None
            for bs in range(0, n, self.block):
                S = q @ self.emb[bs : bs + self.block].T  # [b, block]
                s_blk, i_blk = torch.topk(S.float(), min(k, S.shape[1]), dim=1)
                i_blk = i_blk + bs
                if best_s is None:
                    best_s, best_i = s_blk, i_blk
                else:
                    cat_s, cat_i = torch.cat([best_s, s_blk], 1), torch.cat([best_i, i_blk], 1)
                    best_s, pos = torch.topk(cat_s, k, dim=1)
                    best_i = torch.gather(cat_i, 1, pos)
            all_s.append(best_s.cpu().numpy())
            all_i.append(best_i.cpu().numpy())
        return np.concatenate(all_s), np.concatenate(all_i)

    def search(self, query: str, k: int = 1000) -> list[Hit]:
        if not query.strip():
            return []
        s, i = self.search_vectors(self.encoder.encode_queries([query]), k)
        return [Hit(float(a), self.doc_ids[b]) for a, b in zip(s[0], i[0])]

    def batch_search(self, queries: Queries, k: int = 1000) -> Run:
        qids = list(queries)
        S, I = self.search_vectors(self.encoder.encode_queries([queries[q] for q in qids]), k)
        return {q: {self.doc_ids[j]: float(v) for v, j in zip(srow, irow)} for q, srow, irow in zip(qids, S, I)}
