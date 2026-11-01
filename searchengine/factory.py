"""Assemble retrievers, document sources, rerankers and pipelines from plain settings.

Shared by the CLI scripts and the API so both run exactly the same components.
"""

from __future__ import annotations

from searchengine.config import Settings, get_settings
from searchengine.data.collection import Collection
from searchengine.pipeline import SearchPipeline


def build_retriever(backend: str, coll: Collection, s: Settings, index_name: str | None = None):
    if backend == "local":
        from searchengine.retrieval.bm25_local import LocalBM25

        return LocalBM25.load(s.index_dir / coll.name / "bm25-local")
    if backend == "es":
        from searchengine.index.es import make_client
        from searchengine.retrieval.bm25_es import ElasticsearchBM25

        return ElasticsearchBM25(make_client(s.es_url), index_name or coll.name)
    raise ValueError(f"backend must be 'local' or 'es', got {backend!r}")


def build_docs(backend: str, coll: Collection, s: Settings, retriever=None):
    """Elasticsearch serves text from _source; otherwise use the offset DocStore."""
    if backend == "es" and retriever is not None and hasattr(retriever, "get_documents"):
        return retriever
    from searchengine.docstore import DocStore

    return DocStore.open(coll, s.index_dir)


def build_reranker(model: str | None, device: str = "auto", batch_size: int = 64, max_length: int = 512):
    if not model or model.lower() == "none":
        return None
    from searchengine.ranking.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(model, device=device, batch_size=batch_size, max_length=max_length)


def build_pipeline(
    collection: str,
    backend: str = "es",
    model: str | None = None,
    device: str = "auto",
    settings: Settings | None = None,
) -> SearchPipeline:
    s = settings or get_settings()
    coll = Collection.from_name(collection, s.data_dir)
    retriever = build_retriever(backend, coll, s)
    return SearchPipeline(retriever, build_docs(backend, coll, s, retriever), build_reranker(model, device))
