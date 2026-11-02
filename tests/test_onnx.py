"""ONNX export + int8 on a tiny model: correctness of the export path, not speed."""

import json

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def onnx_dir(tiny_ce_dir, tmp_path_factory):
    from searchengine.ranking.onnx_ce import export_cross_encoder

    out = tmp_path_factory.mktemp("onnx")
    export_cross_encoder(tiny_ce_dir, out)
    return out


def test_export_writes_variants_and_validation(onnx_dir):
    meta = json.loads((onnx_dir / "export_meta.json").read_text(encoding="utf-8"))
    assert set(meta["files"]) == {"fp32", "fp16", "int8"}
    assert meta["graph_fused"] is True
    v = meta["validation"]
    # fp32 ONNX must reproduce PyTorch essentially exactly (validated against a fresh model).
    assert v["fp32"]["max_abs_diff"] < 1e-4 and v["fp32"]["mean_spearman"] > 0.99
    assert np.isfinite(v["int8"]["max_abs_diff"])
    assert v["fp16"]["max_abs_diff"] < 1e-2 and v["fp16"]["mean_spearman"] > 0.9
    fp32_size = (onnx_dir / "model_fp32.onnx").stat().st_size
    assert (onnx_dir / "model_int8.onnx").stat().st_size < fp32_size
    assert (onnx_dir / "model_fp16.onnx").stat().st_size < fp32_size


def test_onnx_reranker_matches_torch_and_is_order_invariant(onnx_dir, tiny_ce_dir):
    from searchengine.ranking.cross_encoder import CrossEncoderReranker
    from searchengine.ranking.onnx_ce import OnnxCrossEncoderReranker

    ps = ["the quick brown fox", "lazy dog", "a cross encoder search engine jumps over the lazy dog", "fast"]
    ref = CrossEncoderReranker(tiny_ce_dir, device="cpu").score("what is a fox", ps)
    r = OnnxCrossEncoderReranker(onnx_dir, "fp32", batch_size=3)
    got = r.score("what is a fox", ps)
    np.testing.assert_allclose(got, ref, atol=1e-4)
    np.testing.assert_allclose(r.score("what is a fox", ps[::-1])[::-1], got, atol=1e-6)
    assert r.score("q", []).shape == (0,)
    assert r.name.startswith("onnx-fp32:") and r.info()["runtime"] == "onnxruntime"


def test_factory_detects_onnx_dir(onnx_dir, tiny_ce_dir):
    from searchengine.factory import build_reranker
    from searchengine.ranking.onnx_ce import OnnxCrossEncoderReranker, is_onnx_dir

    assert is_onnx_dir(str(onnx_dir)) and not is_onnx_dir(tiny_ce_dir)
    r = build_reranker(str(onnx_dir), onnx_variant="int8")
    assert isinstance(r, OnnxCrossEncoderReranker) and r.variant == "int8"
    with pytest.raises(ValueError):
        build_reranker(str(onnx_dir), onnx_variant="int4")


def test_unavailable_gpu_provider_gives_actionable_error(onnx_dir):
    from searchengine.ranking.onnx_ce import OnnxCrossEncoderReranker, available_onnx_devices

    assert "cpu" in available_onnx_devices()
    if "dml" not in available_onnx_devices():
        with pytest.raises(RuntimeError, match="onnxruntime-directml"):
            OnnxCrossEncoderReranker(onnx_dir, "fp16", device="dml")
    with pytest.raises(ValueError):
        OnnxCrossEncoderReranker(onnx_dir, "fp16", device="tpu")
    # "auto" never silently picks a GPU for ONNX
    assert OnnxCrossEncoderReranker(onnx_dir, "fp32", device="auto").device == "cpu"


def test_shape_bucketing_does_not_change_scores(onnx_dir):
    from searchengine.ranking.onnx_ce import OnnxCrossEncoderReranker

    ps = ["the quick brown fox", "lazy dog", "a cross encoder search engine jumps over the lazy dog", "fast"]
    r = OnnxCrossEncoderReranker(onnx_dir, "fp32")
    plain = r.score("what is a fox", ps)
    r.pad_to_multiple_of = 32  # what GPU providers use
    np.testing.assert_allclose(r.score("what is a fox", ps), plain, atol=1e-5)
