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

Refinement (re-ranking ANN candidates with exact scores): PQ distances are
approximate, so the true top 10 is often *in* the IVF-PQ top 1000 but in the wrong
order. `refine_topk` rescores the top k' candidates with their stored vectors
(memory-mapped fp16 shards via `VectorRows`, so RAM holds only the compressed index
and whatever pages the OS caches) and keeps the exact top k. Cost: k' dot products
plus k' row reads per query, instead of N.
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


def exact_topk(store: EmbeddingStore, Q: np.ndarray, k: int, q_batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by streaming over shards: memory O(shard + q_batch x shard), not O(corpus).

    Shards are the outer loop, so each shard is read from disk once however many
    queries there are; queries are scored in chunks to bound the score matrix."""
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    nq = len(Q)
    k_eff = min(k, store.n)
    best_s = np.full((nq, k_eff), -np.inf, dtype=np.float32)
    best_i = np.full((nq, k_eff), -1, dtype=np.int64)
    for off, E in store.shards():
        for a in range(0, nq, q_batch):
            S = Q[a : a + q_batch] @ E.T
            kk = min(k_eff, S.shape[1])
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            cat_s = np.concatenate([best_s[a : a + q_batch], np.take_along_axis(S, part, 1)], 1)
            cat_i = np.concatenate([best_i[a : a + q_batch], part + off], 1)
            keep = np.argpartition(-cat_s, k_eff - 1, axis=1)[:, :k_eff]
            best_s[a : a + q_batch] = np.take_along_axis(cat_s, keep, 1)
            best_i[a : a + q_batch] = np.take_along_axis(cat_i, keep, 1)
    order = np.argsort(-best_s, axis=1, kind="stable")
    return np.take_along_axis(best_s, order, 1), np.take_along_axis(best_i, order, 1)


class VectorRows:
    """Random access to stored vectors by row id, without loading the corpus.

    Each fp16 shard is opened as a read-only memory map; `get` sorts the requested
    rows so reads within a shard go in file order, then returns float32. On the full
    MS MARCO corpus this is 6.8 GB on disk and ~0 bytes of process memory up front;
    repeated queries are served from the OS page cache."""

    def __init__(self, store: EmbeddingStore):
        self.dim = store.dim
        self._shards = [np.load(store.root / sh["file"], mmap_mode="r") for sh in store.meta["shards"]]
        self._starts = np.cumsum([0] + [len(a) for a in self._shards]).astype(np.int64)
        self.n = int(self._starts[-1])

    def get(self, rows) -> np.ndarray:
        """rows: int array of any shape (-1 = missing, returned as a zero vector)."""
        rows = np.asarray(rows, dtype=np.int64)
        flat = rows.ravel()
        out = np.zeros((len(flat), self.dim), dtype=np.float32)
        pos = np.nonzero(flat >= 0)[0]
        if len(pos):
            r = flat[pos]
            if r.max() >= self.n:
                raise IndexError(f"row {int(r.max())} out of range for {self.n} vectors")
            order = np.argsort(r, kind="stable")
            r, pos = r[order], pos[order]
            bounds = np.searchsorted(r, self._starts)  # slice of r falling in each shard
            for sh, arr in enumerate(self._shards):
                lo, hi = bounds[sh], bounds[sh + 1]
                if hi > lo:
                    out[pos[lo:hi]] = arr[r[lo:hi] - self._starts[sh]]
        return out.reshape(*rows.shape, self.dim)


def refine_topk(Q: np.ndarray, cand: np.ndarray, vectors: VectorRows, k: int,
                q_batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Rescore ANN candidates [nq, k'] exactly and keep the top k. Missing (-1) candidates
    sort last and come back as row -1 with score -inf. Queries go in chunks so the
    gathered block stays small (64 x 1000 x 384 fp32 = 98 MB)."""
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.int64)
    kk = min(k, cand.shape[1])
    out_s = np.empty((len(Q), kk), dtype=np.float32)
    out_i = np.empty((len(Q), kk), dtype=np.int64)
    for a in range(0, len(Q), q_batch):
        c = cand[a : a + q_batch]
        S = np.einsum("qd,qkd->qk", Q[a : a + q_batch], vectors.get(c))
        S[c < 0] = -np.inf
        top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        top_s = np.take_along_axis(S, top, 1)
        order = np.argsort(-top_s, axis=1, kind="stable")
        out_s[a : a + q_batch] = np.take_along_axis(top_s, order, 1)
        out_i[a : a + q_batch] = np.take_along_axis(np.take_along_axis(c, top, 1), order, 1)
    out_i[~np.isfinite(out_s)] = -1
    return out_s, out_i


def gather_rows(store: EmbeddingStore, rows: np.ndarray) -> np.ndarray:
    """Fetch specific rows (any shape of indices) as float32 (memory-mapped, no full load)."""
    return VectorRows(store).get(rows)


def recall_vs_exact(store: EmbeddingStore, Q: np.ndarray, exact_scores: np.ndarray, approx_ids: np.ndarray, k: int,
                    tol: float = 1e-5, vectors: VectorRows | None = None) -> float:
    """Tie-aware ANN recall@k: the fraction of approximate top-k hits whose *true* score
    reaches the exact k-th best score. Plain id overlap would punish equally good answers
    when scores tie; and PQ's returned distances are themselves approximate, so true
    scores are recomputed from the stored vectors."""
    ids = approx_ids[:, :k]
    valid = ids >= 0
    vecs = (vectors or VectorRows(store)).get(np.where(valid, ids, -1))
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
