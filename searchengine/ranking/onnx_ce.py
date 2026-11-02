r"""Cross-encoder inference with ONNX Runtime, optionally int8-quantized.

Why it's faster on CPU than PyTorch eager mode:
1. Graph optimization. ONNX Runtime's transformer optimizer fuses each BERT layer's
   ~20 small ops (MatMul, Add, Softmax, LayerNorm, GELU ...) into a few fused
   kernels (Attention, SkipLayerNormalization, BiasGelu): fewer memory round trips,
   less per-op overhead.
2. Dynamic int8 quantization (CPU). Weights are stored as int8 with a per-tensor scale,
   w ~= s_w * q_w. Activations are quantized on the fly per batch,
   x ~= s_x * (q_x - z_x). The MatMul then runs in int8 with int32 accumulation and
   one float rescale, y ~= s_w * s_x * (q_w . (q_x - z_x)). That's 4x less weight
   memory traffic and integer SIMD (AVX2/AVX-512 VNNI) instead of fp32 FMAs.
   The approximation error is small and *measured* (score diff and ranking
   agreement against fp32), not assumed.

3. GPU execution providers. The same exported graph (best: the fp16 variant) runs on
   DirectML (any DirectX 12 GPU on Windows, including AMD/Intel integrated graphics)
   or CUDA, by choosing the ONNX Runtime package: onnxruntime-directml / -gpu.

Export notes (each one learned the hard way, see tests):
- Export with attn_implementation="eager". The fused SDPA path in recent
  transformers traces mask-construction logic that is shape-specialized.
- Validate against a *freshly loaded* reference model: the object that went
  through the exporter is not guaranteed to behave like the original afterwards.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path, PurePath

import numpy as np

from searchengine.ranking.base import Reranker

INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


def _load_reference(model_name: str):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(model_name, attn_implementation="eager").eval()


def export_cross_encoder(
    model_name: str,
    out_dir: Path,
    optimize: bool = True,
    quantize: bool = True,
    opset: int = 17,
    validation_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """Export a HF cross-encoder to ONNX (fp32, optionally fused + int8). Returns metadata."""
    import torch
    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.save_pretrained(out_dir)
    model = _load_reference(model_name)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    example = tok(["example query"] * 2, ["a short passage", "another passage"], padding=True, return_tensors="pt")
    if "token_type_ids" not in example:  # e.g. RoBERTa-style models
        example["token_type_ids"] = torch.zeros_like(example["input_ids"])
    args = tuple(example[k] for k in INPUT_NAMES)
    raw = out_dir / "model_raw.onnx"
    t0 = time.perf_counter()
    try:  # TorchScript exporter: single self-contained file, well supported by ORT tools
        torch.onnx.export(
            Wrapper(model), args, str(raw), dynamo=False, opset_version=opset,
            input_names=list(INPUT_NAMES), output_names=["logits"],
            dynamic_axes={**{k: {0: "batch", 1: "seq"} for k in INPUT_NAMES}, "logits": {0: "batch"}},
        )
        exporter = "torchscript"
    except Exception:  # newer torch may only ship the dynamo exporter
        B, S = torch.export.Dim("batch"), torch.export.Dim("seq")
        torch.onnx.export(
            Wrapper(model), args, str(raw), dynamo=True, opset_version=max(opset, 18), external_data=False,
            input_names=list(INPUT_NAMES), output_names=["logits"],
            dynamic_shapes={k: {0: B, 1: S} for k in INPUT_NAMES},
        )
        exporter = "dynamo"
    export_s = time.perf_counter() - t0

    fp32 = out_dir / "model_fp32.onnx"
    fused = False
    if optimize:
        try:
            from onnxruntime.transformers import optimizer

            cfg = model.config
            opt = optimizer.optimize_model(
                str(raw), model_type="bert", num_heads=cfg.num_attention_heads, hidden_size=cfg.hidden_size
            )
            opt.save_model_to_file(str(fp32))
            fused = True
            # fp16 copy for GPU execution providers (DirectML, CUDA): half the memory traffic,
            # and GPUs run fp16 at ~2x their fp32 rate. Inputs/outputs stay int64/fp32.
            opt.convert_float_to_float16(keep_io_types=True)
            opt.save_model_to_file(str(out_dir / "model_fp16.onnx"))
        except Exception as e:  # optimization is a speedup, never a requirement
            print(f"  graph fusion skipped ({type(e).__name__}: {e}); using the unfused graph")
    if not fused:
        raw.replace(fp32)
    elif raw.exists():
        raw.unlink()

    variants = {"fp32": fp32}
    if (out_dir / "model_fp16.onnx").exists():
        variants["fp16"] = out_dir / "model_fp16.onnx"
    if quantize:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8 = out_dir / "model_int8.onnx"
        # Fused ops (Attention, SkipLayerNormalization) live in the com.microsoft
        # domain, which ONNX shape inference can't type; telling the quantizer the
        # default tensor type is float lets it quantize the MatMuls around them.
        quantize_dynamic(
            str(fp32), str(int8), weight_type=QuantType.QInt8,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        variants["int8"] = int8

    meta = {
        "source_model": model_name,
        "exporter": exporter,
        "opset": opset,
        "graph_fused": fused,
        "export_seconds": round(export_s, 1),
        "num_labels": int(model.config.num_labels),
        "files": {k: {"name": v.name, "mb": round(v.stat().st_size / 2**20, 1)} for k, v in variants.items()},
    }
    # Validate every variant against a freshly loaded PyTorch model.
    pairs = validation_pairs or [
        ("what is a cross encoder", "A cross-encoder reads the query and passage together."),
        ("how long do you bake muffins", "Bake muffins for 20 to 25 minutes at 375 degrees."),
        ("define podiatry", "Podiatry is the branch of medicine dealing with the foot."),
        ("who started tesla", "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning."),
    ]
    def write_meta():
        with open(out_dir / "export_meta.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, indent=2)

    write_meta()  # the ONNX reranker reads it, including during validation
    meta["validation"] = validate_against_torch(model_name, out_dir, pairs, list(variants))
    write_meta()
    return meta


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 1.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def validate_against_torch(model_name: str, onnx_dir: Path, pairs: list[tuple[str, str]], variants: list[str]) -> dict:
    """Compare each variant with PyTorch fp32, per query (rankings are per query):
    max |score diff|, mean Spearman rank correlation, and top-1 / top-5 agreement."""
    import torch

    from searchengine.ranking.cross_encoder import CrossEncoderReranker

    ref = CrossEncoderReranker(model_name, device="cpu", fp16=False)
    by_query: dict[str, list[str]] = {}
    for q, p in pairs:
        by_query.setdefault(q, []).append(p)
    with torch.inference_mode():
        ref_scores = {q: ref.score(q, ps) for q, ps in by_query.items()}
    out = {}
    for v in variants:
        r = OnnxCrossEncoderReranker(onnx_dir, variant=v)
        diffs, rhos, top1, top5 = [], [], [], []
        for q, ps in by_query.items():
            a, b = ref_scores[q], r.score(q, ps)
            diffs.append(float(np.abs(a - b).max()))
            rhos.append(_spearman(a, b))
            top1.append(int(np.argmax(a) == np.argmax(b)))
            k = min(5, len(a))
            top5.append(len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k)
        out[v] = {
            "max_abs_diff": max(diffs),
            "mean_spearman": float(np.mean(rhos)),
            "top1_agreement": float(np.mean(top1)),
            "top5_overlap": float(np.mean(top5)),
            "n_queries": len(by_query),
            "n_pairs": len(pairs),
        }
    return out


PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    # DirectML: any DirectX 12 GPU on Windows, including AMD/Intel integrated GPUs.
    # Needs the onnxruntime-directml package instead of onnxruntime.
    "dml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",  # needs onnxruntime-gpu
}


def available_onnx_devices() -> list[str]:
    import onnxruntime as ort

    have = set(ort.get_available_providers())
    return [d for d, p in PROVIDERS.items() if p in have]


class OnnxCrossEncoderReranker(Reranker):
    def __init__(
        self,
        onnx_dir: str | Path,
        variant: str = "int8",
        max_length: int = 512,
        batch_size: int = 64,
        threads: int | None = None,
        device: str = "cpu",
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if device == "auto":
            device = "cpu"  # explicit opt-in to GPUs: a silent switch would change what CPU numbers mean
        if device not in PROVIDERS:
            raise ValueError(f"ONNX device must be one of {sorted(PROVIDERS)}, got {device!r}")
        if PROVIDERS[device] not in ort.get_available_providers():
            hint = {"dml": "pip uninstall -y onnxruntime && pip install onnxruntime-directml",
                    "cuda": "pip uninstall -y onnxruntime && pip install onnxruntime-gpu"}.get(device, "")
            raise RuntimeError(f"{PROVIDERS[device]} not available in this onnxruntime build. {hint}")

        self.dir = Path(onnx_dir)
        with open(self.dir / "export_meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        if variant not in self.meta["files"]:
            raise ValueError(f"variant {variant!r} not exported; have {sorted(self.meta['files'])}")
        self.variant = variant
        self.device = device
        suffix = "" if device == "cpu" else f"-{device}"
        self.name = f"onnx-{variant}{suffix}:{PurePath(self.meta['source_model']).name}"
        # GPUs prefer few distinct input shapes; pad sequence length to a multiple of 32
        # (masked padding does not change scores; verified in tests).
        self.pad_to_multiple_of = None if device == "cpu" else 32
        self.max_length = max_length
        self.batch_size = batch_size
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        if device == "dml":  # required by the DirectML provider (see ORT DirectML docs)
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [PROVIDERS[device]] + (["CPUExecutionProvider"] if device != "cpu" else [])
        self.session = ort.InferenceSession(str(self.dir / self.meta["files"][variant]["name"]), so, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        # ONNX Runtime docs: a DirectML session supports only one thread calling Run at a
        # time. The API serves requests from a thread pool, so serialize GPU runs; CPU
        # sessions are thread-safe and stay lock-free.
        self._run_lock = threading.Lock() if device != "cpu" else None
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.num_labels = int(self.meta.get("num_labels", 1))
        self.load_seconds = time.perf_counter() - t0

    def info(self) -> dict:
        return {
            "name": self.name,
            "model": self.meta["source_model"],
            "runtime": "onnxruntime",
            "variant": self.variant,
            "graph_fused": self.meta.get("graph_fused"),
            "device": self.device,
            "provider": self.active_provider,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "file_mb": self.meta["files"][self.variant]["mb"],
        }

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        order = np.argsort([len(p) for p in passages], kind="stable")
        out = np.empty(len(passages), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer(
                [query] * len(idx), [passages[i] for i in idx], padding=True,
                truncation="longest_first", max_length=self.max_length, return_tensors="np",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            if "token_type_ids" not in enc:
                enc["token_type_ids"] = np.zeros_like(enc["input_ids"])
            feeds = {k: enc[k].astype(np.int64) for k in INPUT_NAMES if k in self._inputs}
            if self._run_lock is not None:
                with self._run_lock:
                    logits = self.session.run(None, feeds)[0].astype(np.float32)
            else:
                logits = self.session.run(None, feeds)[0].astype(np.float32)
            out[idx] = logits[:, 0] if self.num_labels == 1 else logits[:, -1] - logits[:, 0]
        return out


def is_onnx_dir(path: str) -> bool:
    try:
        return (Path(path) / "export_meta.json").exists()
    except OSError:
        return False
