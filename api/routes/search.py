"""
routes/search.py

Dummy stub FastAPI route for the /search endpoint.
"""

from fastapi import APIRouter

from api.schemas import SearchRequest, SearchResponse, SearchResultItem
from ranking.reranker import Reranker
from retrieval.candidate_retriever import CandidateRetriever

router = APIRouter()

# TODO: replace with a real corpus loaded from the built index (ingestion/build_index.py)
_DUMMY_CORPUS = [
    ["search", "engine", "index", "documents"],
    ["bm25", "ranks", "documents", "by", "relevance"],
    ["neural", "rerankers", "use", "dense", "embeddings"],
]
_DUMMY_TEXTS = {
    "doc_1": "search engines index documents",
    "doc_2": "bm25 ranks documents by relevance",
    "doc_3": "neural rerankers use dense embeddings",
}
_DUMMY_IDS = ["doc_1", "doc_2", "doc_3"]

_retriever = CandidateRetriever(_DUMMY_CORPUS, _DUMMY_IDS)
_reranker = Reranker()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """TODO: wire this up to the real index + reranking pipeline."""
    candidates = _retriever.retrieve(request.query, top_k=request.top_k)

    for c in candidates:
        c["text"] = _DUMMY_TEXTS.get(c["doc_id"], "")

    if request.use_reranker:
        candidates = _reranker.rerank(request.query, candidates, top_k=request.top_k)
        results = [
            SearchResultItem(doc_id=c["doc_id"], text=c["text"], score=c["final_score"])
            for c in candidates
        ]
    else:
        results = [
            SearchResultItem(doc_id=c["doc_id"], text=c["text"], score=c["score"])
            for c in candidates
        ]

    return SearchResponse(query=request.query, results=results, total_results=len(results))
