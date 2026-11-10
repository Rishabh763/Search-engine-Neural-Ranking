"""Assemble retrievers, document sources, rerankers and pipelines from plain settings.

Shared by the CLI scripts and the API so both run exactly the same components.
"""

from __future__ import annotations

import os

from searchengine.config import Settings, get_settings
from searchengine.data.collection import Collection
from searchengine.pipeline import SearchPipeline


def build_retriever(
    backend: str,
    coll: Collection,
    s: Settings,
    index_name: str | None = None,
    dense_model: str | None = None,
    dense_index: str = "flat",
    device: str = "auto",
    nprobe: int = 32,
    ef_search: int = 128,
    rrf_k: int = 60,
    fusion_depth: int = 1000,
    refine: int = 0,
):
    """backend: local | es | dense | dense-exact | hybrid-es | hybrid-local.
    refine: dense ANN only; rescore this many candidates exactly (0 = off)."""
    if backend == "local":
        from searchengine.retrieval.bm25_local import LocalBM25

        return LocalBM25.load(s.index_dir / coll.name / "bm25-local")
    if backend == "es":
        from searchengine.index.es import make_client
        from searchengine.retrieval.bm25_es import ElasticsearchBM25

        return ElasticsearchBM25(make_client(s.es_url), index_name or coll.name)
    if backend == "dense":
        from searchengine.dense.encoder import default_dense_model, model_slug
        from searchengine.dense.index import dense_dir
        from searchengine.retrieval.dense import DenseRetriever

        model = dense_model or default_dense_model()
        root = dense_dir(s.index_dir, coll.name, model_slug(model))
        return DenseRetriever.load(root, dense_index, device=device, nprobe=nprobe, ef_search=ef_search, refine=refine)
    if backend == "dense-exact":
        from searchengine.dense.encoder import default_dense_model, model_slug
        from searchengine.dense.index import dense_dir
        from searchengine.retrieval.dense import DenseExactTorch

        model = dense_model or default_dense_model()
        return DenseExactTorch.load(dense_dir(s.index_dir, coll.name, model_slug(model)), device=device)
    if backend in ("hybrid-es", "hybrid-local"):
        from searchengine.retrieval.hybrid import RRFHybrid

        sparse = build_retriever(backend.split("-", 1)[1], coll, s, index_name=index_name)
        dense = build_retriever("dense", coll, s, dense_model=dense_model, dense_index=dense_index,
                                device=device, nprobe=nprobe, ef_search=ef_search, refine=refine)
        return RRFHybrid([sparse, dense], k_rrf=rrf_k, depth=fusion_depth)
    raise ValueError(f"backend must be local, es, dense, hybrid-es or hybrid-local; got {backend!r}")


def build_docs(backend: str, coll: Collection, s: Settings, retriever=None):
    """Elasticsearch serves text from _source; otherwise use the offset DocStore."""
    if backend in ("es", "hybrid-es") and retriever is not None:
        return retriever
    from searchengine.docstore import DocStore

    return DocStore.open(coll, s.index_dir)


def build_reranker(
    model: str | None,
    device: str = "auto",
    batch_size: int = 64,
    max_length: int = 512,
    quantize: str | None = None,
    onnx_variant: str = "int8",
):
    """`model` is a HF id / local checkpoint (PyTorch), or a directory made by
    scripts/export_onnx.py (ONNX Runtime). quantize="int8" = PyTorch dynamic int8 (CPU)."""
    if not model or model.lower() == "none":
        return None
    from searchengine.ranking.onnx_ce import OnnxCrossEncoderReranker, is_onnx_dir

    if is_onnx_dir(model):
        return OnnxCrossEncoderReranker(model, variant=onnx_variant, batch_size=batch_size, max_length=max_length,
                                        device=device)
    from searchengine.ranking.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(model, device=device, batch_size=batch_size, max_length=max_length, quantize=quantize)


def build_pipeline(
    collection: str,
    backend: str = "es",
    model: str | None = None,
    device: str = "auto",
    settings: Settings | None = None,
    quantize: str | None = None,
    onnx_variant: str = "int8",
    dense_index: str = "flat",
    nprobe: int = 32,
    refine: int = 0,
) -> SearchPipeline:
    s = settings or get_settings()
    coll = Collection.from_name(collection, s.data_dir)
    # The dense query encoder runs on PyTorch, which has no DirectML device: "dml" only
    # applies to ONNX rerankers, so the encoder falls back to auto (CPU / CUDA / MPS).
    torch_device = device if device in ("auto", "cpu", "cuda", "mps") else "auto"
    retriever = build_retriever(backend, coll, s, dense_index=dense_index, device=torch_device, nprobe=nprobe, refine=refine)
    reranker = build_reranker(model, device, quantize=quantize, onnx_variant=onnx_variant)
    return SearchPipeline(retriever, build_docs(backend, coll, s, retriever), reranker)
