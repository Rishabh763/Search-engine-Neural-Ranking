"""HTTP search API.

Run (from the project root, with Elasticsearch up):
    uvicorn searchengine.api:app --port 8000
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI, or:
    curl "http://127.0.0.1:8000/search?q=what+is+a+cross+encoder&k=5"

Configuration (environment variables):
    SEARCH_COLLECTION   collection name                (default msmarco-100k)
    SEARCH_BACKEND      es | local | dense | hybrid-es | hybrid-local   (default es)
    DENSE_MODEL         bi-encoder for dense/hybrid    (default msmarco-MiniLM-L6-cos-v5)
    DENSE_INDEX         flat | hnsw | ivfpq            (default flat)
    RERANK_MODEL        HF model id/path, ONNX export dir, or "none"
                                                       (default cross-encoder/ms-marco-MiniLM-L-6-v2)
    RERANK_QUANTIZE     int8 = PyTorch dynamic int8 on CPU (default off)
    RERANK_ONNX_VARIANT int8 | fp32, when RERANK_MODEL is an ONNX export dir (default int8)
    RERANK_DEVICE       auto | cpu | cuda | mps        (default auto)

Endpoints are plain `def` (not `async def`): retrieval and model inference are
blocking, CPU/GPU-bound calls, and FastAPI runs `def` endpoints in a worker
thread pool so they don't block the event loop.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from searchengine.pipeline import SearchPipeline

SNIPPET_CHARS = 300


class ResultOut(BaseModel):
    rank: int
    id: str
    title: str
    snippet: str
    score: float
    bm25_score: float  # first-stage score (BM25, dense cosine, or RRF, by backend)
    bm25_rank: int
    rerank_score: float | None


class TimingsOut(BaseModel):
    retrieve_ms: float
    fetch_ms: float
    rerank_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    k: int
    candidates: int
    reranked: bool
    results: list[ResultOut]
    timings: TimingsOut


def create_app(pipeline: SearchPipeline | None = None, meta: dict | None = None) -> FastAPI:
    """Pass a pipeline for tests; otherwise one is built from env vars at startup."""
    state: dict = {"pipeline": pipeline, "meta": meta or {}}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["pipeline"] is None:
            from searchengine.factory import build_pipeline
            from searchengine.ranking.cross_encoder import DEFAULT_MODEL

            collection = os.environ.get("SEARCH_COLLECTION", "msmarco-100k")
            backend = os.environ.get("SEARCH_BACKEND", "es")
            model = os.environ.get("RERANK_MODEL", DEFAULT_MODEL)
            state["pipeline"] = build_pipeline(
                collection, backend, model, os.environ.get("RERANK_DEVICE", "auto"),
                quantize=os.environ.get("RERANK_QUANTIZE") or None,
                onnx_variant=os.environ.get("RERANK_ONNX_VARIANT", "int8"),
                dense_index=os.environ.get("DENSE_INDEX", "flat"),
            )
            state["meta"] = {"collection": collection, "backend": backend}
            # Warm up so the first real request doesn't pay model/JIT/cache start-up.
            state["pipeline"].search("warm up query", k=10, candidates=20)
        yield

    app = FastAPI(title="Large-Scale Search Engine", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        p: SearchPipeline = state["pipeline"]
        return {
            "status": "ok",
            **state["meta"],
            "retriever": p.retriever.name,
            "reranker": p.reranker.info() if p.reranker else None,
        }

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str = Query(..., min_length=1, max_length=1000, description="query text"),
        k: int = Query(10, ge=1, le=100, description="results to return"),
        candidates: int = Query(100, ge=1, le=1000, description="first-stage candidates to rerank"),
        rerank: bool = Query(True, description="apply the cross-encoder"),
    ) -> SearchResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="query is blank")
        if candidates < k:
            raise HTTPException(status_code=422, detail="candidates must be >= k")
        p: SearchPipeline = state["pipeline"]
        results, t = p.search(q, k=k, candidates=candidates, rerank=rerank)
        reranked = rerank and p.reranker is not None
        return SearchResponse(
            query=q,
            k=k,
            candidates=candidates if reranked else k,
            reranked=reranked,
            results=[
                ResultOut(
                    rank=r.rank,
                    id=r.doc_id,
                    title=r.title,
                    snippet=r.text[:SNIPPET_CHARS] + ("..." if len(r.text) > SNIPPET_CHARS else ""),
                    score=r.score,
                    bm25_score=r.first_stage_score,
                    bm25_rank=r.first_stage_rank,
                    rerank_score=r.rerank_score,
                )
                for r in results
            ],
            timings=TimingsOut(**t.__dict__),
        )

    return app


def __getattr__(name: str):
    # `uvicorn searchengine.api:app` builds the real app lazily, so importing this
    # module in tests doesn't try to load models or connect to Elasticsearch.
    if name == "app":
        return create_app()
    raise AttributeError(name)
