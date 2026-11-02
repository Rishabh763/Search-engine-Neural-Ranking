r"""Embedding storage (sharded, resumable) and FAISS index construction.

Layout, per collection and model:

    indexes/<collection>/dense/<model-slug>/
        meta.json          encoder config, shard list, throughput, machine
        ids.json           row -> doc id (corpus order)
        emb_00000.npy ...  float16 shards of `shard_size` rows (half the disk of fp32;
                           scores are computed in fp32 after loading)
        flat.faiss / hnsw.faiss / ivfpq.faiss  (+ .json with build params and ANN recall)

Index types, all on inner product (= cosine, vectors are normalized):

  flat   Exact search: N x d multiply-adds per query. Ground truth for the others.
  hnsw   Graph search (Hierarchical Navigable Small World): greedy walk through a
         multi-layer proximity graph, ~O(log N) hops. Fast and high recall, but
         stores full vectors plus M links per node: more memory than flat.
  ivfpq  Inverted file + product quantization. k-means splits the space into
         `nlist` cells; a query visits only the `nprobe` nearest cells (IVF). Each
         vector is stored as m bytes: split into m sub-vectors, each replaced by
         the id of the nearest of 256 learned centroids (PQ). 384-d fp32 (1536 B)
         -> 48 B, a 32x reduction, at a measured cost in recall.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def dense_dir(index_dir: Path, collection: str, slug: str) -> Path:
    return Path(index_dir) / collection / "dense" / slug


class EmbeddingStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        with open(self.root / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self._ids: list[str] | None = None

    @property
    def ids(self) -> list[str]:
        if self._ids is None:
            with open(self.root / "ids.json", encoding="utf-8") as f:
                self._ids = json.load(f)
        return self._ids

    @property
    def dim(self) -> int:
        return int(self.meta["encoder"]["dim"])

    @property
    def n(self) -> int:
        return sum(s["n"] for s in self.meta["shards"])

    def shards(self):
        """Yield (row_offset, float32 array) per shard, in row order."""
        off = 0
        for sh in self.meta["shards"]:
            arr = np.load(self.root / sh["file"]).astype(np.float32)
            yield off, arr
            off += len(arr)

    def sample(self, size: int, seed: int = 0) -> np.ndarray:
        """Uniform sample of rows (for IVF/PQ training) without loading everything."""
        rng = np.random.default_rng(seed)
        frac = min(1.0, size / max(self.n, 1))
        parts = [arr[rng.random(len(arr)) < frac] for _, arr in self.shards()]
        return np.concatenate(parts) if parts else np.zeros((0, self.dim), np.float32)


def exact_topk(store: EmbeddingStore, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by streaming over shards: memory O(shard), not O(corpus)."""
    nq = len(Q)
    best_s = np.full((nq, 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, 0), dtype=np.int64)
    for off, E in store.shards():
        S = Q @ E.T
        kk = min(k, S.shape[1])
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        best_s = np.concatenate([best_s, np.take_along_axis(S, part, 1)], 1)
        best_i = np.concatenate([best_i, part + off], 1)
        if best_s.shape[1] > k:
            keep = np.argpartition(-best_s, k - 1, axis=1)[:, :k]
            best_s, best_i = np.take_along_axis(best_s, keep, 1), np.take_along_axis(best_i, keep, 1)
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_s, order, 1), np.take_along_axis(best_i, order, 1)


def gather_rows(store: EmbeddingStore, rows: np.ndarray) -> np.ndarray:
    """Fetch specific rows (any shape of indices) as float32, one pass over the shards."""
    flat = np.asarray(rows).ravel()
    out = np.zeros((len(flat), store.dim), dtype=np.float32)
    for off, arr in store.shards():
        mask = (flat >= off) & (flat < off + len(arr))
        if mask.any():
            out[mask] = arr[flat[mask] - off]
    return out.reshape(*np.asarray(rows).shape, store.dim)


def recall_vs_exact(store: EmbeddingStore, Q: np.ndarray, exact_scores: np.ndarray, approx_ids: np.ndarray, k: int,
                    tol: float = 1e-5) -> float:
    """Tie-aware ANN recall@k: the fraction of approximate top-k hits whose *true* score
    reaches the exact k-th best score. Plain id overlap would punish equally good answers
    when scores tie; and PQ's returned distances are themselves approximate, so true
    scores are recomputed from the stored vectors."""
    ids = approx_ids[:, :k]
    valid = ids >= 0
    vecs = gather_rows(store, np.where(valid, ids, 0))
    true = np.einsum("qd,qkd->qk", Q, vecs)
    kth = exact_scores[:, k - 1 : k]
    return float(np.mean((true >= kth - tol) & valid))


def default_nlist(n: int) -> int:
    # Rule of thumb ~4*sqrt(N) cells; at least 39 training points per centroid (FAISS warns below).
    return int(max(1, min(4 * int(np.sqrt(n)), n // 39)))


def build_faiss_index(
    store: EmbeddingStore,
    kind: str,
    nlist: int | None = None,
    pq_m: int = 48,
    nbits: int = 8,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    train_size: int = 200_000,
):
    import faiss

    d = store.dim
    ip = faiss.METRIC_INNER_PRODUCT
    params: dict = {"type": kind}
    t0 = time.perf_counter()
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, ip)
        index.hnsw.efConstruction = ef_construction
        params.update(hnsw_m=hnsw_m, ef_construction=ef_construction)
    elif kind == "ivfpq":
        if d % pq_m:
            raise ValueError(f"pq_m={pq_m} must divide dim={d}")
        nlist = nlist or default_nlist(store.n)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, nbits, ip)
        train = store.sample(max(train_size, 39 * nlist))
        index.train(train)
        params.update(nlist=nlist, pq_m=pq_m, nbits=nbits, train_size=len(train), bytes_per_vector=pq_m * nbits // 8)
    else:
        raise ValueError("kind must be flat, hnsw or ivfpq")
    for _, arr in store.shards():
        index.add(arr)
    params["build_seconds"] = round(time.perf_counter() - t0, 1)
    params["n"] = int(index.ntotal)
    return index, params


def set_search_params(index, nprobe: int | None = None, ef_search: int | None = None, k: int | None = None) -> None:
    import faiss

    base = faiss.extract_index_ivf(index) if hasattr(faiss, "extract_index_ivf") and "IVF" in type(index).__name__ else None
    if base is not None and nprobe:
        base.nprobe = nprobe
    if hasattr(index, "hnsw") and ef_search:
        index.hnsw.efSearch = max(ef_search, k or 0)  # efSearch < k cannot return k results reliably
