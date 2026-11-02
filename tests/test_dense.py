"""Dense retrieval end to end on the toy collection with a tiny bi-encoder (no downloads)."""

import json
import shutil
import zlib

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

from searchengine.dense.build import encode_collection  # noqa: E402
from searchengine.dense.index import EmbeddingStore, build_faiss_index, exact_topk, set_search_params  # noqa: E402


@pytest.fixture(scope="module")
def encoded(toy_collection, tiny_bi_encoder_dir, tmp_path_factory):
    from searchengine.dense.encoder import BiEncoder

    coll, docs, queries, qrels = toy_collection
    enc = BiEncoder(tiny_bi_encoder_dir, device="cpu", batch_size=64, max_length=64)
    root = tmp_path_factory.mktemp("dense") / "tiny"
    encode_collection(coll, enc, root, shard_size=250, log=lambda m: None, progress=False)
    return root, enc, coll, docs, queries, qrels


def test_encoding_shards_resume_and_normalization(encoded):
    root, enc, coll, docs, *_ = encoded
    store = EmbeddingStore(root)
    assert [sh["n"] for sh in store.meta["shards"]] == [250, 250, 100]
    assert store.n == len(docs) and store.ids == [d.id for d in docs]
    norms = np.concatenate([np.linalg.norm(a, axis=1) for _, a in store.shards()])
    np.testing.assert_allclose(norms, 1.0, atol=2e-3)  # fp16 storage of unit vectors
    # Resume: a second run re-encodes nothing.
    logs = []
    before = (root / "emb_00000.npy").stat().st_mtime_ns
    encode_collection(coll, enc, root, shard_size=250, log=logs.append, progress=False)
    assert sum("skipping" in m for m in logs) == 3
    assert (root / "emb_00000.npy").stat().st_mtime_ns == before
    with pytest.raises(ValueError):
        encode_collection(coll, enc, root, shard_size=100, log=lambda m: None, progress=False)


def test_exact_topk_matches_brute_force(encoded):
    root, enc, *_ = encoded
    store = EmbeddingStore(root)
    E = np.concatenate([a for _, a in store.shards()])
    Q = E[[3, 77, 400]] + 0.01
    _, idx = exact_topk(store, Q, 15)
    ref = np.argsort(-(Q @ E.T), axis=1)[:, :15]
    for a, b in zip(idx, ref):
        assert set(a) == set(b)


class ClusteredEncoder:
    """Deterministic stand-in encoder with realistic geometry: unit vectors around
    topic centroids (real embeddings are clustered, not uniform noise or all-equal)."""

    def __init__(self, dim=48, n_topics=20, seed=0):
        rng = np.random.default_rng(seed)
        self.centroids = rng.normal(size=(n_topics, dim))
        self.dim = dim

    def config(self):
        return {"model": "clustered-fake", "dim": self.dim, "normalized": True, "max_length": None}

    def _vec(self, text):
        h = zlib.crc32(text.encode("utf-8"))  # stable across processes, unlike hash()
        rng = np.random.default_rng(h)
        v = self.centroids[h % len(self.centroids)] + 0.6 * rng.normal(size=self.dim)
        return v / np.linalg.norm(v)

    def encode_passages(self, texts, progress=False):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    encode_queries = encode_passages


@pytest.fixture(scope="module")
def clustered_store(toy_collection, tmp_path_factory):
    coll, *_ = toy_collection
    root = tmp_path_factory.mktemp("dense") / "clustered"
    enc = ClusteredEncoder()
    encode_collection(coll, enc, root, shard_size=250, log=lambda m: None, progress=False)
    return EmbeddingStore(root), enc


@pytest.mark.parametrize("kind,floor", [("flat", 1.0), ("hnsw", 0.97), ("ivfpq", 0.5)])
def test_faiss_indexes_recall_vs_exact(clustered_store, kind, floor):
    from searchengine.dense.index import recall_vs_exact

    store, enc = clustered_store
    index, params = build_faiss_index(store, kind, nlist=8, pq_m=8, hnsw_m=16, train_size=600)
    assert index.ntotal == store.n
    Q = enc.encode_queries([f"query {i}" for i in range(50)])
    exact_s, _ = exact_topk(store, Q, 10)
    set_search_params(index, nprobe=8, ef_search=128, k=10)  # nprobe = nlist: IVF part exhaustive
    _, approx = index.search(Q, 10)
    r = recall_vs_exact(store, Q, exact_s, approx, 10)
    assert r >= floor, (kind, r, params)  # ivfpq: 48-d -> 8 bytes is heavy compression


def test_recall_vs_exact_is_tie_aware(encoded):
    """The tiny random encoder maps every passage to the same vector: all scores tie.
    Id overlap would report <1; tie-aware recall correctly reports 1."""
    from searchengine.dense.index import recall_vs_exact

    root, enc, *_ = encoded
    store = EmbeddingStore(root)
    Q = enc.encode_queries(["the quick fox", "lazy dog"])
    exact_s, exact_i = exact_topk(store, Q, 10)
    other = np.roll(np.arange(store.n), 7)[None, :10].repeat(2, 0)  # different ids, same (tied) scores
    assert recall_vs_exact(store, Q, exact_s, other, 10) == pytest.approx(1.0)


def test_dense_retriever_and_factory(encoded, tmp_path, monkeypatch):
    import faiss

    from searchengine.config import Settings
    from searchengine.factory import build_docs, build_retriever
    from searchengine.retrieval.dense import DenseRetriever

    root, enc, coll, docs, queries, qrels = encoded
    store = EmbeddingStore(root)
    index, _ = build_faiss_index(store, "flat")
    faiss.write_index(index, str(root / "flat.faiss"))
    r = DenseRetriever.load(root, "flat", device="cpu", encoder=enc)
    q = next(iter(queries.values()))
    single = [h.doc_id for h in r.search(q, 20)]
    batch = r.batch_search({"x": q}, k=20)["x"]
    assert single == sorted(batch, key=lambda d: -batch[d])
    assert r.search("   ", 5) == []
    assert r.name == "dense-flat:tiny"

    # Factory: dense + hybrid-local over a settings object pointing at temp dirs.
    idx_dir = tmp_path / "indexes"
    (idx_dir / coll.name / "dense").mkdir(parents=True)
    # Copy rather than symlink: creating symlinks on Windows needs admin rights or Developer Mode.
    shutil.copytree(root, idx_dir / coll.name / "dense" / "tiny")
    from searchengine.retrieval.bm25_local import LocalBM25

    LocalBM25.build(docs).save(idx_dir / coll.name / "bm25-local")
    s = Settings(es_url="http://127.0.0.1:1", data_dir=coll.root.parent, index_dir=idx_dir, runs_dir=tmp_path / "runs")
    meta = json.loads((root / "meta.json").read_text())
    monkeypatch.setattr("searchengine.dense.encoder.model_slug", lambda m: "tiny")
    hybrid = build_retriever("hybrid-local", coll, s, dense_model=meta["encoder"]["model"], device="cpu")
    assert hybrid.name == "rrf(bm25-local+dense-flat:tiny)"
    hits = hybrid.search(q, 10)
    assert len(hits) == 10
    assert type(build_docs("hybrid-local", coll, s, hybrid)).__name__ == "DocStore"
