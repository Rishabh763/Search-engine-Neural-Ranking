"""Core data types shared across retrieval, ranking and evaluation.

The two central structures follow the TREC conventions used by trec_eval,
pytrec_eval and ranx, so runs can be exchanged with those tools directly:

    Run   = {query_id: {doc_id: score}}      higher score = more relevant
    Qrels = {query_id: {doc_id: relevance}}  graded int labels, 0 = judged non-relevant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

Run = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]
Queries = Dict[str, str]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True, order=True)
class Hit:
    # order=True sorts by score first; callers sort with reverse=True.
    score: float
    doc_id: str = field(compare=False)


def hits_to_run_entry(hits: list[Hit]) -> Dict[str, float]:
    return {h.doc_id: float(h.score) for h in hits}
