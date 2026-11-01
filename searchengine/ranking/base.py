"""Reranker interface: score (query, passage) pairs; higher = more relevant."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """One float per passage, same order as the input."""

    def info(self) -> dict:
        return {"name": self.name}
