"""Random access to passage text by id.

The reranker needs the text of each candidate. Two sources implement the same
`get_documents(ids)` interface:

- DocStore: byte offsets into the collection's corpus.jsonl, so a lookup is one
  seek + one line read. No server needed; used for offline evaluation and with
  the local BM25 backend. Offsets are built once and cached next to the indexes.
- ElasticsearchBM25.get_documents: an `_mget` against the index's _source, for
  the Elasticsearch backend (one round trip for all candidates).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from searchengine.data.collection import Collection
from searchengine.types import Document


class DocumentSource(Protocol):
    def get_documents(self, doc_ids: list[str]) -> dict[str, Document]: ...


class DocStore:
    def __init__(self, corpus_path: Path, ids: list[str], offsets: np.ndarray):
        self.corpus_path = Path(corpus_path)
        self.ids = ids
        self.offsets = offsets
        self._row = {d: i for i, d in enumerate(ids)}
        self._fh = None
        self._lock = threading.Lock()  # one shared file handle; the API serves from a thread pool

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, collection: Collection, store_dir: Path) -> "DocStore":
        corpus = collection.root / "corpus.jsonl"
        ids: list[str] = []
        offsets: list[int] = []
        pos = 0
        with open(corpus, "rb") as f:  # binary: offsets must be byte positions
            for line in f:
                ids.append(json.loads(line)["id"])
                offsets.append(pos)
                pos += len(line)
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(offsets, dtype=np.int64)
        np.save(store_dir / "offsets.npy", arr)
        with open(store_dir / "ids.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(ids, f)
        with open(store_dir / "meta.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"corpus_bytes": pos, "n_docs": len(ids)}, f)
        return cls(corpus, ids, arr)

    @classmethod
    def open(cls, collection: Collection, index_dir: Path) -> "DocStore":
        """Load the cached offsets, (re)building them if missing or stale."""
        store_dir = Path(index_dir) / collection.name / "docstore"
        corpus = collection.root / "corpus.jsonl"
        meta_path = store_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("corpus_bytes") == corpus.stat().st_size:
                with open(store_dir / "ids.json", encoding="utf-8") as f:
                    ids = json.load(f)
                return cls(corpus, ids, np.load(store_dir / "offsets.npy"))
        return cls.build(collection, store_dir)

    def get_documents(self, doc_ids: list[str]) -> dict[str, Document]:
        out: dict[str, Document] = {}
        # Read in file order: sequential-ish seeks are kinder to the disk.
        rows = sorted((self._row[d], d) for d in doc_ids if d in self._row)
        with self._lock:
            if self._fh is None:
                self._fh = open(self.corpus_path, "rb")
            lines = []
            for row, d in rows:
                self._fh.seek(int(self.offsets[row]))
                lines.append((d, self._fh.readline()))
        for d, line in lines:
            r = json.loads(line.decode("utf-8"))
            out[d] = Document(r["id"], r["text"], r.get("title", ""))
        return out

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
