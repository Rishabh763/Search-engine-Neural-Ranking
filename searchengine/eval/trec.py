"""Read/write TREC-format qrels and run files (the lingua franca of IR tooling).

qrels line:  <qid> <iteration> <docid> <relevance>
run line:    <qid> Q0 <docid> <rank> <score> <tag>
"""

from __future__ import annotations

import re
from pathlib import Path

from searchengine.types import Qrels, Run

_UNSAFE = re.compile(r'[<>:"/\\|?*]')


def safe_name(tag: str) -> str:
    """File-system-safe run name. On Windows NTFS a ':' in a file name doesn't fail,
    it silently writes an *alternate data stream* (a hidden stream attached to the file
    named up to the colon): invisible in Explorer, lost on copy/zip, untracked by git.
    'dense-flat:msmarco-MiniLM' -> 'dense-flat-msmarco-MiniLM'."""
    return _UNSAFE.sub("-", tag)


def read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            qid, _, docid, rel = parts[:4]
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def write_qrels(qrels: Qrels, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                f.write(f"{qid} 0 {docid} {rel}\n")


def read_run(path: Path) -> Run:
    run: Run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            qid, _, docid, _rank, score = parts[:5]
            run.setdefault(qid, {})[docid] = float(score)
    return run


def write_run(run: Run, path: Path, tag: str = "run", depth: int | None = None) -> None:
    """Write a run sorted by descending score (ties broken like trec_eval)."""
    from searchengine.eval.metrics import rank_documents

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, scores in run.items():
            ranked = rank_documents(scores)
            if depth is not None:
                ranked = ranked[:depth]
            for rank, docid in enumerate(ranked, start=1):
                # repr() keeps full float precision: rounding to a few decimals turns
                # close scores into ties and silently changes the ranking on re-read.
                f.write(f"{qid} Q0 {docid} {rank} {float(scores[docid])!r} {tag}\n")
