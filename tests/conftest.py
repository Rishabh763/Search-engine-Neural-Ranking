"""Shared fixtures for Phase 2 tests: a toy collection on disk, and stub rerankers."""

import numpy as np
import pytest

from searchengine.data.collection import Collection, write_collection
from searchengine.data.toy import make_toy_collection
from searchengine.ranking.base import Reranker


@pytest.fixture(scope="session")
def toy_collection(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "toy"
    docs, queries, qrels = make_toy_collection(n_docs=600, n_topics=20, n_queries=30)
    write_collection(root, docs, {"toy": (queries, qrels)})
    return Collection(root), docs, queries, qrels


class ReverseLengthReranker(Reranker):
    """Deterministic stand-in for a model: shorter passages score higher."""

    name = "fake:reverse-length"

    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, len(passages)))
        return np.array([-len(p) for p in passages], dtype=np.float32)


class OracleReranker(Reranker):
    """Scores passages by their qrels label: reranking with it must not hurt nDCG."""

    name = "fake:oracle"

    def __init__(self, qrels, queries, docs):
        text_to_id = {d.text: d.id for d in docs}
        self.rels = {queries[q]: qrels[q] for q in qrels}
        self.text_to_id = text_to_id

    def score(self, query, passages):
        rels = self.rels.get(query, {})
        return np.array([rels.get(self.text_to_id.get(p, ""), 0) for p in passages], dtype=np.float32)


@pytest.fixture
def reverse_length_reranker():
    return ReverseLengthReranker()


# ---------- tiny transformer models built on the fly (no downloads) ----------

TINY_WORDS = "the quick brown fox dog jumps over lazy what is a cross encoder search engine fast".split()


def _tiny_bert(d, num_labels=None, extra_words=()):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(dict.fromkeys([*TINY_WORDS, *extra_words]))
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizer(vocab_file=str(d / "vocab.txt"))
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, num_labels=num_labels or 1)
    (BertForSequenceClassification(cfg) if num_labels else BertModel(cfg)).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def tiny_ce_dir(tmp_path_factory):
    """Tiny randomly initialized cross-encoder (BERT + 1-logit head)."""
    return _tiny_bert(tmp_path_factory.mktemp("tiny-ce"), num_labels=1)


@pytest.fixture(scope="session")
def tiny_bi_encoder_dir(tmp_path_factory, toy_collection):
    """Tiny sentence-transformers bi-encoder (BERT + mean pooling + normalize) over the toy vocabulary."""
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    try:  # sentence-transformers >= 6 moved the module classes
        from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer
    except ImportError:
        from sentence_transformers.models import Normalize, Pooling, Transformer

    _, docs, _, _ = toy_collection
    words = sorted({w for d in docs for w in d.text.split()})
    base = _tiny_bert(tmp_path_factory.mktemp("tiny-bert"), extra_words=words)
    word = Transformer(base, max_seq_length=64)
    dim = (getattr(word, "get_embedding_dimension", None) or word.get_word_embedding_dimension)()
    st = SentenceTransformer(modules=[word, Pooling(dim, "mean"), Normalize()])
    out = tmp_path_factory.mktemp("tiny-bi")
    st.save(str(out))
    return str(out)
