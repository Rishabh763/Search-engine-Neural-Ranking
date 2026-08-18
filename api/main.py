"""
main.py

Dummy stub FastAPI application entrypoint.

Run with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI

from api.routes.search import router as search_router
from api.schemas import HealthResponse

app = FastAPI(title="Large-Scale Search Engine API", version="0.1.0")

app.include_router(search_router, prefix="/api/v1", tags=["search"])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/")
def root():
    return {"message": "Large-scale search engine API [stub]. See /docs for endpoints."}
