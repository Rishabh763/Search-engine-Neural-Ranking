from searchengine.data.build import select_subset
from searchengine.data.collection import Collection, write_collection
from searchengine.data.toy import make_toy_collection
from searchengine.types import Document


def test_collection_roundtrip(tmp_path):
    docs, queries, qrels = make_toy_collection(n_docs=200, n_queries=10)
    docs.append(Document("weird", "tabs\tand\nnewlines", title="Ünïcode title"))
    write_collection(tmp_path / "c", docs, {"toy": (queries, qrels)}, {"seed": 1})
    c = Collection(tmp_path / "c")
    assert len(c) == 201 and c.eval_set_names == ["toy"]
    assert c.queries("toy") == queries
    assert c.qrels("toy") == qrels
    back = list(c.iter_docs())
    assert back[-1] == docs[-1]  # JSONL keeps text exactly, including tabs/newlines


def test_subset_keeps_required_and_is_deterministic():
    docs = [Document(f"d{i}", "x") for i in range(10_000)]
    required = {"d5", "d9999", "d1234"}
    a = [d.id for d in select_subset(iter(docs), len(docs), required, 500, seed=3)]
    b = [d.id for d in select_subset(iter(docs), len(docs), required, 500, seed=3)]
    assert a == b
    assert required <= set(a)
    assert 495 <= len(a) <= 500  # |required| + fill - overlap
