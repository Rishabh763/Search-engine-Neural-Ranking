r"""Cross-encoder reranker.

A cross-encoder reads the query and passage as ONE sequence,

    [CLS] query tokens [SEP] passage tokens [SEP]

so self-attention runs across both texts at every layer (a query token can attend
directly to passage tokens). A linear head on the final [CLS] vector gives the
relevance logit s = w^T h_[CLS] + b. That joint encoding is why it is far more
accurate than a bi-encoder, and why it is expensive: nothing can be precomputed,
each candidate costs a full forward pass, and attention is O((|q| + |d|)^2) per
layer. Hence the cascade: BM25 narrows millions of passages to N candidates, and
only those N are scored here.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (6 layers, 384 hidden, ~22M
params), trained on MS MARCO with BM25-mined negatives. Its model card reports
MRR@10 ~= 0.39 on MS MARCO dev and nDCG@10 ~= 0.74 on TREC DL 2019 when reranking
BM25 candidates on the full corpus: our sanity anchors.

Engineering details that affect latency:
- All N candidates of a query go through in as few batches as possible, with
  dynamic padding (each batch is padded to its own longest pair, not to max_length).
- Pairs are sorted by length before batching, so similar lengths share a batch
  and less compute is wasted on padding. Scores are restored to input order.
- fp16 on GPU; inference_mode() on any device (no autograd bookkeeping).
- Optional dynamic int8 on CPU (quantize="int8"). For ONNX Runtime variants see onnx_ce.py.
  Which CPU variant is fastest depends on the machine: measured on one AVX-512 Xeon,
  torch fp32 1.0x, ONNX fp32 0.47x, ONNX int8 1.0x, torch int8 1.8x.
"""

from __future__ import annotations

import time
from pathlib import PurePath

import numpy as np

from searchengine.ranking.base import Reranker

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dynamic_int8(model):
    """PyTorch dynamic quantization: nn.Linear weights -> int8, activations quantized per batch.

    Uses fbgemm/x86 int8 kernels. Measured 1.8x faster than fp32 on an AVX-512 Xeon
    for a MiniLM-L6-shaped model; the speedup on other CPUs must be measured
    (scripts/benchmark.py), not assumed.
    """
    import warnings

    import torch

    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:  # very old torch
        from torch.quantization import quantize_dynamic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # deprecation notice pointing to torchao
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 64,
        fp16: bool | None = None,
        quantize: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name_or_path
        self.name = f"ce:{PurePath(model_name_or_path).name}"  # works for HF ids and Windows paths
        self.device = resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.quantize = quantize
        if quantize not in (None, "int8"):
            raise ValueError("quantize must be None or 'int8'")
        if quantize == "int8":
            if self.device != "cpu":
                raise ValueError("dynamic int8 quantization is a CPU optimization; use fp16 on GPU")
            model = _dynamic_int8(model.eval())
            self.name = f"ce-int8:{PurePath(model_name_or_path).name}"
        self.fp16 = (self.device == "cuda") if fp16 is None else fp16
        if self.fp16:
            model = model.half()
        self.model = model.to(self.device).eval()
        self.num_labels = int(self.model.config.num_labels)
        self.load_seconds = time.perf_counter() - t0
        self._torch = torch

    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "name": self.name,
            "model": self.model_name,
            "device": self.device,
            "fp16": self.fp16,
            "quantize": self.quantize,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "params_millions": round(n_params / 1e6, 1),
            "torch_threads": self._torch.get_num_threads(),
        }

    def _logits_to_scores(self, logits) -> np.ndarray:
        logits = logits.float()
        if self.num_labels == 1:
            s = logits[:, 0]
        else:  # 2-class heads: margin of "relevant" over "not relevant"
            s = logits[:, -1] - logits[:, 0]
        return s.cpu().numpy().astype(np.float32)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        torch = self._torch
        order = np.argsort([len(p) for p in passages], kind="stable")
        out = np.empty(len(passages), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                enc = self.tokenizer(
                    [query] * len(idx),
                    [passages[i] for i in idx],
                    padding=True,
                    truncation="longest_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out[idx] = self._logits_to_scores(self.model(**enc).logits)
        return out
