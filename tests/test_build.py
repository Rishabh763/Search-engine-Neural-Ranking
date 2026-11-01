"""The ir_datasets build path, with ir_datasets replaced by an in-memory fake
(so the test runs offline and fast)."""

from collections import namedtuple

import pytest

import searchengine.data.build as build_mod
from searchengine.data.collection import Collection

Q = namedtuple("Q", "query_id text")
R = namedtuple("R", "query_id doc_id relevance iteration")
D = namedtuple("D", "doc_id text")


class FakeDataset:
    def __init__(self, docs=None, queries=None, qrels=None):
        self._docs, self._queries, self._qrels = docs or [], queries or [], qrels or []

    def docs_iter(self):
        return iter(self._docs)

    def docs_count(self):
        return len(self._docs)

    def queries_iter(self):
        return iter(self._queries)

    def qrels_iter(self):
        return iter(self._qrels)


@pytest.fixture
def fake_ir(monkeypatch):
    docs = [D(f"p{i}", f"passage {i}") for i in range(5000)]
    dev_q = [Q(f"q{i}", f"query {i}") for i in range(50)] + [Q("unjudged", "no qrels")]
    dev_r = [R(f"q{i}", f"p{i * 7}", 1, "0") for i in range(50)]
    dl_q = [Q("t1", "trec one"), Q("t2", "trec two")]
    dl_r = [R("t1", "p1", 3, "Q0"), R("t1", "p2", 0, "Q0"), R("t2", "p3", 1, "Q0")]
    table = {
        "msmarco-passage": FakeDataset(docs=docs),
        "msmarco-passage/dev/small": FakeDataset(queries=dev_q, qrels=dev_r),
        "msmarco-passage/trec-dl-2019/judged": FakeDataset(queries=dl_q, qrels=dl_r),
    }

    class FakeIR:
        @staticmethod
        def load(name):
            return table[name]

    monkeypatch.setattr(build_mod, "_ir", lambda: FakeIR)


def test_build_subset(tmp_path, fake_ir):
    m = build_mod.build(tmp_path / "sub", "msmarco-passage", ["msmarco-dev", "dl19"], n_docs=300, n_queries=20, seed=1)
    c = Collection(tmp_path / "sub")
    assert m["subset"] is True
    dev_q, dev_r = c.queries("msmarco-dev"), c.qrels("msmarco-dev")
    assert len(dev_q) == 20 and "unjudged" not in dev_q
    assert set(dev_q) == set(dev_r)
    ids = {d.id for d in c.iter_docs()}
    judged = {d for qr in list(dev_r.values()) + list(c.qrels("dl19").values()) for d in qr}
    assert judged <= ids  # every judged passage, including label-0 ones, survives
    assert 290 <= len(ids) <= 300


def test_build_full_and_corpus_mismatch(tmp_path, fake_ir):
    m = build_mod.build(tmp_path / "full", "msmarco-passage", ["dl19"])
    assert m["subset"] is False and m["n_docs"] == 5000
    with pytest.raises(ValueError):
        build_mod.build(tmp_path / "bad", "msmarco-passage", ["scifact"])


def test_ir_datasets_tempfile_patch(tmp_path):
    """The Windows fix: ir_datasets' temp download file must come back already closed."""
    import os

    pytest.importorskip("ir_datasets")
    import ir_datasets.util.download as dl

    build_mod._patch_ir_datasets_tempfiles()
    build_mod._patch_ir_datasets_tempfiles()  # idempotent
    assert isinstance(dl.tempfile, build_mod._TempfileShim)
    f = dl.tempfile.NamedTemporaryFile(delete=False, dir=tmp_path)
    assert isinstance(f, build_mod._ClosedTempFile)
    # What ir_datasets does next: write <name>.tmp then os.replace it onto <name>.
    with open(f.name + ".tmp", "wb") as out:
        out.write(b"ok")
    os.replace(f.name + ".tmp", f.name)
    with open(f.name, "rb") as inp:
        assert inp.read() == b"ok"
    assert dl.tempfile.gettempdir()  # other tempfile attributes still pass through
