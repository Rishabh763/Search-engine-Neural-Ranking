"""BM25 retrieval through Elasticsearch.

A `match` query on an analyzed field is exactly a BM25 sum: the query text goes
through the same `english` analyzer, each resulting token becomes a SHOULD
clause, and the document score is the sum of the per-term BM25 scores.

Batch evaluation uses `_msearch` so 1000 queries cost ~16 HTTP round trips at
batch_size=64 instead of 1000.
"""

from __future__ import annotations

from searchengine.index.es import SEARCH_FIELD
from searchengine.retrieval.base import Retriever
from searchengine.types import Hit, Queries, Run


class ElasticsearchBM25(Retriever):
    name = "bm25-es"

    def __init__(self, es, index: str, field: str = SEARCH_FIELD):
        self.es = es
        self.index = index
        self.field = field

    def _body(self, query: str, k: int) -> dict:
        return {
            "query": {"match": {self.field: {"query": query}}},
            "size": k,
            "_source": False,
            # Exact hit counting forces visiting every match; we only need top-k.
            "track_total_hits": False,
        }

    @staticmethod
    def _parse(resp: dict) -> list[Hit]:
        return [Hit(float(h["_score"]), h["_id"]) for h in resp["hits"]["hits"]]

    def search(self, query: str, k: int = 1000) -> list[Hit]:
        if not query.strip():
            return []
        resp = self.es.search(index=self.index, **self._body(query, k))
        return self._parse(resp)

    def batch_search(self, queries: Queries, k: int = 1000, batch_size: int = 64) -> Run:
        run: Run = {}
        items = list(queries.items())
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            searches: list[dict] = []
            for _, text in chunk:
                searches.append({"index": self.index})
                searches.append(self._body(text, k))
            resp = self.es.msearch(searches=searches)
            for (qid, _), r in zip(chunk, resp["responses"]):
                if "error" in r:
                    raise RuntimeError(f"query {qid} failed: {r['error']}")
                run[qid] = {h.doc_id: h.score for h in self._parse(r)}
        return run

    def get_documents(self, doc_ids: list[str]) -> dict[str, dict]:
        """Fetch stored title/text for ids (used for snippets and by the reranker)."""
        if not doc_ids:
            return {}
        resp = self.es.mget(index=self.index, ids=doc_ids)
        return {d["_id"]: d["_source"] for d in resp["docs"] if d.get("found")}
