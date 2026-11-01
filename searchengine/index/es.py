"""Elasticsearch index management for BM25 retrieval.

Design decisions (each one is a likely interview question):

- One primary shard. BM25 statistics (df, avgdl) are computed per shard. With
  several shards each shard scores with its own local IDF, so the same document
  can score differently depending on where it lands. One shard makes scores
  exact and matches the local implementation. 8.8M passages (~3-4 GB index) fit
  in one shard comfortably; at larger scale you'd shard and use
  `search_type=dfs_query_then_fetch` or accept the approximation.
- BM25 parameters live in the index's *default* similarity, so they can be
  changed later by close -> update settings -> open, without reindexing.
  MS MARCO-tuned values from Anserini are k1=0.82, b=0.68. Lucene's defaults
  are k1=1.2, b=0.75. We start from Anserini's untuned k1=0.9, b=0.4.
- One searchable field, `contents` = title + text, analyzed with the built-in
  `english` analyzer. `title` and `text` are kept in _source (for snippets and
  for feeding the reranker) but not indexed. `contents` is excluded from
  _source so the text isn't stored twice.
- Bulk loading with refresh disabled and 0 replicas, then refresh and an
  optional force-merge to 1 segment (fewer segments = faster search).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from searchengine.types import Document

SEARCH_FIELD = "contents"


def make_client(url: str, timeout: int = 120):
    from elasticsearch import Elasticsearch

    return Elasticsearch(url, request_timeout=timeout)


def index_definition(k1: float, b: float) -> dict:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
            "similarity": {"default": {"type": "BM25", "k1": k1, "b": b}},
        },
        "mappings": {
            "_source": {"excludes": [SEARCH_FIELD]},
            "properties": {
                SEARCH_FIELD: {"type": "text", "analyzer": "english"},
                "title": {"type": "text", "index": False},
                "text": {"type": "text", "index": False},
            },
        },
    }


def create_index(es, name: str, k1: float = 0.9, b: float = 0.4, overwrite: bool = False) -> None:
    if es.indices.exists(index=name):
        if not overwrite:
            raise FileExistsError(f"Index {name!r} exists. Pass overwrite=True (--overwrite) to rebuild it.")
        es.indices.delete(index=name)
    body = index_definition(k1, b)
    es.indices.create(index=name, settings=body["settings"], mappings=body["mappings"])


@dataclass
class IndexingStats:
    n_docs: int
    n_errors: int
    seconds: float

    @property
    def docs_per_second(self) -> float:
        return self.n_docs / self.seconds if self.seconds else 0.0


def _actions(name: str, docs: Iterable[Document]):
    for d in docs:
        yield {
            "_index": name,
            "_id": d.id,
            "_source": {
                SEARCH_FIELD: f"{d.title} {d.text}" if d.title else d.text,
                "title": d.title,
                "text": d.text,
            },
        }


def bulk_index(es, name: str, docs: Iterable[Document], chunk_size: int = 2000, force_merge: bool = True) -> IndexingStats:
    from elasticsearch import helpers

    t0 = time.perf_counter()
    n_ok = n_err = 0
    for ok, info in helpers.streaming_bulk(
        es, _actions(name, docs), chunk_size=chunk_size, raise_on_error=False, max_retries=3
    ):
        if ok:
            n_ok += 1
        else:
            n_err += 1
            if n_err <= 5:
                print(f"  indexing error: {info}")
    es.indices.put_settings(index=name, settings={"index": {"refresh_interval": "1s"}})
    es.indices.refresh(index=name)
    if force_merge:
        es.options(request_timeout=3600).indices.forcemerge(index=name, max_num_segments=1)
    return IndexingStats(n_ok, n_err, time.perf_counter() - t0)


def set_bm25_params(es, name: str, k1: float, b: float) -> None:
    """Change BM25 k1/b on an existing index without reindexing."""
    es.indices.close(index=name)
    try:
        es.indices.put_settings(index=name, settings={"index": {"similarity": {"default": {"type": "BM25", "k1": k1, "b": b}}}})
    finally:
        es.indices.open(index=name)
        es.cluster.health(index=name, wait_for_status="yellow", timeout="60s")


def get_bm25_params(es, name: str) -> dict:
    s = es.indices.get_settings(index=name)[name]["settings"]["index"]
    sim = s.get("similarity", {}).get("default", {})
    return {"k1": float(sim.get("k1", 1.2)), "b": float(sim.get("b", 0.75))}
