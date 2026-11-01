"""Retriever interface. Every first-stage retriever (BM25, dense, hybrid) implements it,
so evaluation, fusion and the API never depend on a concrete backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from searchengine.types import Hit, Queries, Run, hits_to_run_entry


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def search(self, query: str, k: int = 1000) -> list[Hit]:
        """Top-k hits for one query, sorted by descending score."""

    def batch_search(self, queries: Queries, k: int = 1000) -> Run:
        """Override when the backend can do better than a loop (vectorized, msearch)."""
        return {qid: hits_to_run_entry(self.search(text, k)) for qid, text in queries.items()}
