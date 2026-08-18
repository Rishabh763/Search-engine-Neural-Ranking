"""
schemas.py

Dummy stub Pydantic schemas for the search API request/response bodies.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Raw user search query")
    top_k: int = Field(10, description="Number of results to return")
    use_reranker: bool = Field(True, description="Whether to apply neural reranking")


class SearchResultItem(BaseModel):
    doc_id: str
    title: Optional[str] = None
    text: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total_results: int


class HealthResponse(BaseModel):
    status: str = "ok"
