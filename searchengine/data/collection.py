"""On-disk format for a prepared collection (a corpus plus its evaluation sets).

    data/processed/<collection>/
        manifest.json                  provenance: sources, seed, counts
        corpus.jsonl                   {"id", "title", "text"} one per line
        evalsets/<name>/queries.tsv    qid <TAB> text
        evalsets/<name>/qrels.txt      TREC format: qid 0 docid rel

Plain text formats on purpose: they stream, diff, and load into any IR tool.
All files are UTF-8 with '\\n' newlines regardless of OS.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from searchengine.eval.trec import read_qrels, write_qrels
from searchengine.types import Document, Qrels, Queries


def _clean(s: str) -> str:
    return " ".join(s.split())  # collapses tabs/newlines that would break TSV


def write_queries(queries: Queries, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, text in queries.items():
            f.write(f"{qid}\t{_clean(text)}\n")


def read_queries(path: Path) -> Queries:
    queries: Queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[qid] = text
    return queries


def write_collection(
    out_dir: Path,
    docs: Iterable[Document],
    eval_sets: dict[str, tuple[Queries, Qrels]],
    manifest: dict | None = None,
) -> dict:
    """Write a collection to disk. Returns the manifest that was written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "title": d.title, "text": d.text}, ensure_ascii=False) + "\n")
            n_docs += 1
    es_meta = {}
    for name, (queries, qrels) in eval_sets.items():
        d = out_dir / "evalsets" / name
        write_queries(queries, d / "queries.tsv")
        write_qrels(qrels, d / "qrels.txt")
        es_meta[name] = {"n_queries": len(queries), "n_judgments": sum(len(v) for v in qrels.values())}
    m = dict(manifest or {})
    m.update(
        {
            "name": out_dir.name,
            "n_docs": n_docs,
            "eval_sets": es_meta,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    with open(out_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=2)
    return m


class Collection:
    """Read access to a prepared collection directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        mpath = self.root / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(
                f"No manifest at {mpath}. Run scripts/prepare_data.py first to build this collection."
            )
        with open(mpath, encoding="utf-8") as f:
            self.manifest: dict = json.load(f)

    @classmethod
    def from_name(cls, name: str, data_dir: Path) -> "Collection":
        return cls(Path(data_dir) / name)

    @property
    def name(self) -> str:
        return self.manifest["name"]

    def __len__(self) -> int:
        return int(self.manifest["n_docs"])

    @property
    def eval_set_names(self) -> list[str]:
        return sorted(self.manifest.get("eval_sets", {}))

    def iter_docs(self) -> Iterator[Document]:
        with open(self.root / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                yield Document(id=r["id"], text=r["text"], title=r.get("title", ""))

    def _evalset_dir(self, name: str) -> Path:
        d = self.root / "evalsets" / name
        if not d.exists():
            raise KeyError(f"Collection {self.name!r} has no eval set {name!r}; has {self.eval_set_names}")
        return d

    def queries(self, eval_set: str) -> Queries:
        return read_queries(self._evalset_dir(eval_set) / "queries.tsv")

    def qrels(self, eval_set: str) -> Qrels:
        return read_qrels(self._evalset_dir(eval_set) / "qrels.txt")

    # ---- training split (written by scripts/prepare_train.py) ----

    def _train_dir(self) -> Path:
        d = self.root / "train"
        if not (d / "queries.tsv").exists():
            raise FileNotFoundError(f"No training split in {self.root}: run scripts/prepare_train.py first")
        return d

    def train_queries(self) -> Queries:
        return read_queries(self._train_dir() / "queries.tsv")

    def train_qrels(self) -> Qrels:
        return read_qrels(self._train_dir() / "qrels.txt")
