"""HTTP search API.

Run (from the project root, with Elasticsearch up):
    uvicorn searchengine.api:app --port 8000
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI, or:
    curl "http://127.0.0.1:8000/search?q=what+is+a+cross+encoder&k=5"

Configuration (environment variables):
    SEARCH_COLLECTION   collection name                (default msmarco-100k)
    SEARCH_BACKEND      es | local | dense | hybrid-es | hybrid-local   (default es)
    DENSE_MODEL         bi-encoder for dense/hybrid    (default msmarco-MiniLM-L6-cos-v5)
    DENSE_INDEX         flat | hnsw | ivfpq | a named index such as ivfpq96 (default flat)
    DENSE_NPROBE        IVF cells searched per query   (default 32)
    DENSE_REFINE        rescore this many ANN candidates exactly from the stored
                        vectors (default 0 = off; 1000 recommended with IVF-PQ)
    RERANK_MODEL        HF model id/path, ONNX export dir, or "none"
                                                       (default cross-encoder/ms-marco-MiniLM-L-6-v2)
    RERANK_QUANTIZE     int8 = PyTorch dynamic int8 on CPU (default off)
    RERANK_ONNX_VARIANT int8 | fp32, when RERANK_MODEL is an ONNX export dir (default int8)
    RERANK_DEVICE       auto | cpu | cuda | mps | dml  (default auto; dml = DirectML, ONNX only)
    SEARCH_CANDIDATES   default `candidates` per request (default 100; 20 is the measured
                        sweet spot with dense retrieval, see README "Operating points")

Endpoints are plain `def` (not `async def`): retrieval and model inference are
blocking, CPU/GPU-bound calls, and FastAPI runs `def` endpoints in a worker
thread pool so they don't block the event loop.
"""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from searchengine.pipeline import SearchPipeline

SNIPPET_CHARS = 300


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))  # overflow-safe


class ResultOut(BaseModel):
    rank: int
    id: str
    title: str
    snippet: str
    score: float
    first_stage_score: float  # BM25 score, dense cosine, or RRF score, depending on the backend
    first_stage_rank: int
    rerank_score: float | None
    # sigmoid(rerank logit). The MS MARCO cross-encoders are trained with binary
    # cross-entropy, so this approximates P(relevant); other rerankers may not be calibrated.
    relevance: float | None


class TimingsOut(BaseModel):
    retrieve_ms: float
    fetch_ms: float
    rerank_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    retriever: str
    k: int
    candidates: int
    reranked: bool
    filtered_out: int  # results dropped by min_relevance
    results: list[ResultOut]
    timings: TimingsOut


def create_app(pipeline: SearchPipeline | None = None, meta: dict | None = None) -> FastAPI:
    """Pass a pipeline for tests; otherwise one is built from env vars at startup."""
    state: dict = {"pipeline": pipeline, "meta": meta or {}}
    default_candidates = int(os.environ.get("SEARCH_CANDIDATES", "100"))

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
                nprobe=int(os.environ.get("DENSE_NPROBE", "32")),
                refine=int(os.environ.get("DENSE_REFINE", "0")),
            )
            state["meta"] = {"collection": collection, "backend": backend}
            # Warm up so the first real request doesn't pay model/JIT/cache start-up.
            state["pipeline"].search("warm up query", k=10, candidates=20)
        yield

    app = FastAPI(title="Large-Scale Search Engine", version="0.3.0", lifespan=lifespan)

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
        candidates: int = Query(default_candidates, ge=1, le=1000, description="first-stage candidates to rerank"),
        rerank: bool = Query(True, description="apply the cross-encoder"),
        min_relevance: float = Query(0.0, ge=0.0, le=1.0, description="drop reranked results below this relevance"),
    ) -> SearchResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="query is blank")
        if candidates < k:
            raise HTTPException(status_code=422, detail="candidates must be >= k")
        p: SearchPipeline = state["pipeline"]
        results, t = p.search(q, k=k, candidates=candidates, rerank=rerank)
        reranked = rerank and p.reranker is not None
        out = []
        for r in results:
            rel = sigmoid(r.rerank_score) if r.rerank_score is not None else None
            if rel is not None and rel < min_relevance:
                continue
            out.append(
                ResultOut(
                    rank=r.rank,
                    id=r.doc_id,
                    title=r.title,
                    snippet=r.text[:SNIPPET_CHARS] + ("..." if len(r.text) > SNIPPET_CHARS else ""),
                    score=r.score,
                    first_stage_score=r.first_stage_score,
                    first_stage_rank=r.first_stage_rank,
                    rerank_score=r.rerank_score,
                    relevance=None if rel is None else round(rel, 4),
                )
            )
        return SearchResponse(
            query=q,
            retriever=p.retriever.name,
            k=k,
            candidates=candidates if reranked else k,
            reranked=reranked,
            filtered_out=len(results) - len(out),
            results=out,
            timings=TimingsOut(**t.__dict__),
        )

    return app


def __getattr__(name: str):
    # `uvicorn searchengine.api:app` builds the real app lazily, so importing this
    # module in tests doesn't try to load models or connect to Elasticsearch.
    if name == "app":
        return create_app()
    raise AttributeError(name)
