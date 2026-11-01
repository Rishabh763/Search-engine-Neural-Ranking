"""CrossEncoderReranker on a tiny randomly initialized BERT built on the fly.

No download: this exercises the real code path (tokenization of pairs, batching,
length sorting, logits -> scores) without the real checkpoint. Scores are
meaningless; what's tested is the plumbing.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

WORDS = "the quick brown fox dog jumps over lazy what is a cross encoder search engine fast".split()


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    d = tmp_path_factory.mktemp("tiny-ce")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizer(vocab_file=str(d / "vocab.txt"))
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, num_labels=1,
    )
    BertForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def reranker(tiny_model_dir):
    from searchengine.ranking.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(tiny_model_dir, device="cpu", batch_size=2)


PASSAGES = ["the quick brown fox", "lazy dog", "a cross encoder search engine jumps over the lazy dog", "fast"]


def test_scores_shape_and_type(reranker):
    s = reranker.score("what is a fox", PASSAGES)
    assert s.shape == (4,) and s.dtype == np.float32 and np.isfinite(s).all()
    assert reranker.score("q", []).shape == (0,)


def test_order_and_batching_invariance(reranker):
    q = "what is a fox"
    base = reranker.score(q, PASSAGES)
    rev = reranker.score(q, PASSAGES[::-1])[::-1]
    np.testing.assert_allclose(base, rev, atol=1e-6)  # length-sorting restores input order
    old = reranker.batch_size
    try:
        reranker.batch_size = 64
        np.testing.assert_allclose(base, reranker.score(q, PASSAGES), atol=1e-5)  # padding doesn't leak
    finally:
        reranker.batch_size = old


def test_truncation_handles_long_passages(reranker):
    long = " ".join(["dog"] * 2000)
    s = reranker.score("what is a dog", [long, "dog"])
    assert np.isfinite(s).all()


def test_info(reranker, tiny_model_dir):
    info = reranker.info()
    assert info["device"] == "cpu" and info["fp16"] is False
    assert reranker.name == "ce:" + os.path.basename(tiny_model_dir)
