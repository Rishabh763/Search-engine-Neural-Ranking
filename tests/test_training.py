"""Training pipeline pieces: training-split export, negative mining, grouped data, LCE training.
Runs on CPU with tiny models in seconds; the real runs happen on Kaggle GPUs."""

import json
import random

import numpy as np
import pytest

from searchengine.data.collection import Collection
from searchengine.docstore import DocStore
from searchengine.eval.trec import write_qrels
from searchengine.training.negatives import mine_negatives, read_negatives, select_negatives

from test_sweep_and_hybrid import Fixed  # noqa: E402  (simple fixed-ranking retriever)


def test_select_negatives_skips_positives_and_top():
    ranked = ["p", "a", "b", "c", "d", "e", "f"]
    rng = random.Random(0)
    negs = select_negatives(ranked, {"p"}, skip_top=2, n_neg=10, rng=rng)
    assert negs == ["c", "d", "e", "f"]  # p removed, a and b skipped, fewer than 10 -> all
    negs = select_negatives(ranked, {"p"}, skip_top=1, n_neg=3, rng=rng)
    assert len(negs) == 3 and "p" not in negs and "a" not in negs


class BatchFixed(Fixed):
    def batch_search(self, queries, k=1000):
        return {q: {d: float(len(self.ranking) - i) for i, d in enumerate(self.ranking[:k])} for q in queries}


def test_mine_negatives_writes_jsonl_and_stats(tmp_path):
    r = BatchFixed("fixed", ["d1", "d2", "d3", "d4", "d5", "d6"])
    queries = {"q1": "x", "q2": "y", "q3": "z"}
    qrels = {"q1": {"d2": 1}, "q2": {"d9": 1}, "q3": {"d1": 1}}
    stats = mine_negatives(r, queries, qrels, tmp_path / "n.jsonl", depth=6, skip_top=1, n_neg=3, chunk=2, log=lambda m: None)
    recs = {x["qid"]: x for x in read_negatives(tmp_path / "n.jsonl")}
    assert set(recs) == {"q1", "q2", "q3"}
    assert "d2" not in recs["q1"]["neg"] and "d1" not in recs["q1"]["neg"]  # positive excluded, top skipped
    assert stats["positive_recall_at_depth"] == pytest.approx(2 / 3, abs=1e-3)  # q2's positive never retrieved
    assert json.loads((tmp_path / "n.stats.json").read_text())["queries"] == 3
    # Reproducible: same seed -> same file
    mine_negatives(r, queries, qrels, tmp_path / "m.jsonl", depth=6, skip_top=1, n_neg=3, log=lambda m: None)
    assert (tmp_path / "n.jsonl").read_text() == (tmp_path / "m.jsonl").read_text()


@pytest.fixture
def toy_with_train(toy_collection, tmp_path):
    """Toy collection copy with a train split (its queries reused as training queries)."""
    coll, docs, queries, qrels = toy_collection
    import shutil

    root = tmp_path / "toy"
    shutil.copytree(coll.root, root)
    from searchengine.data.collection import write_queries

    pos = {q: {d: r for d, r in rels.items() if r > 0} for q, rels in qrels.items()}
    write_queries(queries, root / "train" / "queries.tsv")
    write_qrels(pos, root / "train" / "qrels.txt")
    return Collection(root), docs, queries, pos, tmp_path


def test_group_dataset_shapes_and_padding(toy_with_train):
    from searchengine.training.train_ce import GroupDataset

    coll, docs, queries, pos, tmp = toy_with_train
    store = DocStore.open(coll, tmp / "idx")
    recs = [{"qid": q, "pos": sorted(p), "neg": [docs[0].id]} for q, p in pos.items()]  # only 1 neg: needs padding
    ds = GroupDataset(recs, coll.train_queries(), store, group_size=5, seed=1, all_ids=store.ids)
    q, texts = ds.group(0)
    assert q == coll.train_queries()[ds.items[0]["qid"]] and len(texts) == 5
    assert texts == ds.group(0)[1]  # deterministic per (seed, epoch, qid)


def test_lce_training_can_memorize_and_saves_model(toy_with_train, tiny_ce_toy_dir):
    """Plumbing check: a working loop (tokenization, grouping, LCE loss, AMP/scaler,
    optimizer, scheduler) must be able to memorize a few groups. Generalization is
    not testable with a tiny random model; the real runs fine-tune pretrained ones."""
    pytest.importorskip("torch")
    from searchengine.ranking.cross_encoder import CrossEncoderReranker
    from searchengine.training.train_ce import GroupDataset, TrainConfig, train_cross_encoder

    coll, docs, queries, pos, tmp = toy_with_train
    store = DocStore.open(coll, tmp / "idx")
    qs = sorted(pos)[:4]
    recs = [{"qid": q, "pos": [sorted(pos[q])[0]], "neg": [docs[(i * 37 + k) % len(docs)].id for k in range(3)]}
            for i, q in enumerate(qs)]
    ds = GroupDataset(recs, coll.train_queries(), store, group_size=4, seed=0, all_ids=store.ids)
    cfg = TrainConfig(init_model=tiny_ce_toy_dir, out_dir=str(tmp / "model"), epochs=150, lr=1e-3, batch_groups=4,
                      group_size=4, max_length=96, log_every=25, save_every=0, device="cpu", warmup_frac=0.0)
    meta = train_cross_encoder(cfg, ds, log=lambda m: None)
    losses = [h["loss"] for h in meta["history"]]
    assert losses[0] == pytest.approx(np.log(4), abs=0.15)  # untrained head: uniform over G=4
    assert losses[-1] < 0.7, losses  # memorized
    r = CrossEncoderReranker(str(tmp / "model"), device="cpu")  # output is a normal reranker dir
    assert r.score("q", ["a", "b"]).shape == (2,)
    assert (tmp / "model" / "train_meta.json").exists()


class BatchBM25:
    name = "bm25"

    def __init__(self, bm25):
        self.bm25 = bm25

    def batch_search(self, queries, k=1000):
        return self.bm25.batch_search(queries, k=k)


def test_export_train_split_with_fake_ir(tmp_path, monkeypatch):
    from collections import namedtuple

    import searchengine.data.build as build_mod
    from searchengine.data.collection import write_collection
    from searchengine.types import Document

    Q = namedtuple("Q", "query_id text")
    R = namedtuple("R", "query_id doc_id relevance iteration")

    class DS:
        def queries_iter(self):
            return iter([Q(f"t{i}", f"train query {i}") for i in range(50)])

        def qrels_iter(self):
            return iter([R(f"t{i}", f"p{i}", 1 if i % 5 else 0, "0") for i in range(50)])

    class IR:
        @staticmethod
        def load(name):
            assert name == build_mod.TRAIN_IR_ID
            return DS()

    monkeypatch.setattr(build_mod, "_ir", lambda: IR)
    write_collection(tmp_path / "c", [Document("p1", "x")], {"dl19": ({"a": "q"}, {"a": {"p1": 1}})})
    meta = build_mod.export_train_split(tmp_path / "c", n_queries=20, seed=3)
    c = Collection(tmp_path / "c")
    tq, tr = c.train_queries(), c.train_qrels()
    assert len(tq) == 20 == meta["n_queries"] and set(tq) == set(tr)
    assert all(int(q[1:]) % 5 for q in tq)  # queries whose only label is 0 were dropped
    assert all(r > 0 for rels in tr.values() for r in rels.values())


def test_dense_exact_torch_matches_brute_force(toy_collection, tmp_path):
    torch = pytest.importorskip("torch")
    from test_dense import ClusteredEncoder  # realistic geometry, no model download

    from searchengine.dense.build import encode_collection
    from searchengine.retrieval.dense import DenseExactTorch

    coll, docs, queries, _ = toy_collection
    enc = ClusteredEncoder()
    root = tmp_path / "clustered"
    encode_collection(coll, enc, root, shard_size=250, log=lambda m: None, progress=False)
    r = DenseExactTorch.load(root, device="cpu", encoder=enc, block=170)  # several blocks + merge
    E = np.concatenate([np.load(root / f"emb_0000{i}.npy").astype(np.float32) for i in range(3)])
    Q = enc.encode_queries(list(queries.values())[:10])
    S, I = r.search_vectors(Q, 25)
    ref = np.argsort(-(Q @ E.T), axis=1)[:, :25]
    for a, b in zip(I, ref):
        assert set(a) == set(b)
    assert np.all(np.diff(S, axis=1) <= 1e-6)  # sorted descending
    run = r.batch_search({"x": list(queries.values())[0]}, k=5)
    assert len(run["x"]) == 5 and r.name == "dense-exact:clustered"
