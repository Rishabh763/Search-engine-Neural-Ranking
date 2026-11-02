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

@pytest.fixture(scope="module")
def reranker(tiny_ce_dir):
    from searchengine.ranking.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(tiny_ce_dir, device="cpu", batch_size=2)


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


def test_info(reranker, tiny_ce_dir):
    info = reranker.info()
    assert info["device"] == "cpu" and info["fp16"] is False
    assert reranker.name == "ce:" + os.path.basename(tiny_ce_dir)


def test_torch_dynamic_int8_variant(tiny_ce_dir):
    from searchengine.ranking.cross_encoder import CrossEncoderReranker

    q8 = CrossEncoderReranker(tiny_ce_dir, device="cpu", quantize="int8")
    assert q8.name.startswith("ce-int8:") and q8.info()["quantize"] == "int8"
    s = q8.score("what is a fox", PASSAGES)
    assert s.shape == (4,) and np.isfinite(s).all()
    with pytest.raises(ValueError):
        CrossEncoderReranker(tiny_ce_dir, device="cpu", quantize="int4")
