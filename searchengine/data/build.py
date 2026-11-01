"""Build collections from ir_datasets, either the full corpus or a reproducible subset.

Why subsets, and how they are built
-----------------------------------
The full MS MARCO passage corpus has 8,841,823 passages. Developing against all
of it makes every iteration slow, so we first build subsets of 100K-1M passages.
A random subset alone is useless for evaluation: almost none of the judged
passages would survive. So a subset is

    (every passage judged for the sampled queries)  U  (uniform random fill)

Including judged non-relevant passages (label 0) matters on TREC DL: those were
pooled from strong systems, so they are hard distractors and keep the subset
from being trivially easy.

Caveat (also printed in the manifest): a subset is still easier than the full
corpus because most distractors are gone. Numbers on a subset are for fast
iteration and relative comparison only; headline numbers come from the full
corpus.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from searchengine.data.collection import write_collection
from searchengine.data.registry import CORPORA, get_eval_set
from searchengine.types import Document, Qrels, Queries


class _ClosedTempFile:
    """Stands in for a NamedTemporaryFile whose handle was already closed."""

    def __init__(self, name: str):
        self.name = name


class _TempfileShim:
    """`tempfile` module proxy whose NamedTemporaryFile returns a closed file.

    ir_datasets (<= 0.6.3) downloads small files into
    `tempfile.NamedTemporaryFile(delete=False)` but never closes that handle, then
    `os.replace`s a finished download onto its path. Linux allows replacing an open
    file; Windows does not (WinError 5 "Access is denied"). Closing the handle right
    after creation is all the fix needs: ir_datasets only ever uses `.name`.
    """

    def __getattr__(self, attr):
        return getattr(tempfile, attr)

    @staticmethod
    def NamedTemporaryFile(*args, **kwargs):
        f = tempfile.NamedTemporaryFile(*args, **kwargs)
        f.close()
        return _ClosedTempFile(f.name)


def _patch_ir_datasets_tempfiles() -> None:
    import ir_datasets.util.download as dl

    if not isinstance(dl.tempfile, _TempfileShim):
        dl.tempfile = _TempfileShim()


def _ir():
    try:
        import ir_datasets
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install ir_datasets") from e
    # Harmless on Linux/macOS, required on Windows; applying it everywhere keeps
    # one code path (and lets the test suite exercise it on any OS).
    _patch_ir_datasets_tempfiles()
    return ir_datasets


def load_eval_set(name: str, n_queries: int | None, seed: int) -> tuple[Queries, Qrels]:
    """Load queries + qrels for an eval set, optionally sampling n_queries of them.

    Only queries that have at least one judgment are kept (a query with no
    judgments cannot be scored). Sampling is seeded and order-independent.
    """
    spec = get_eval_set(name)
    ds = _ir().load(spec.ir_datasets_id)
    qrels: Qrels = {}
    for q in ds.qrels_iter():
        qrels.setdefault(q.query_id, {})[q.doc_id] = int(q.relevance)
    queries: Queries = {q.query_id: q.text for q in ds.queries_iter() if q.query_id in qrels}

    if n_queries is not None and n_queries < len(queries):
        rng = random.Random(f"{seed}:{name}")
        keep = set(rng.sample(sorted(queries), n_queries))
        queries = {k: v for k, v in queries.items() if k in keep}
    qrels = {qid: qrels[qid] for qid in queries}
    return queries, qrels


def _iter_corpus(corpus_key: str) -> tuple[Iterator[Document], int | None]:
    ds = _ir().load(CORPORA[corpus_key])
    total = ds.docs_count()

    def gen():
        for d in ds.docs_iter():
            yield Document(id=d.doc_id, text=d.text, title=getattr(d, "title", "") or "")

    return gen(), total


def select_subset(
    docs: Iterator[Document],
    total: int,
    required: set[str],
    n_docs: int,
    seed: int,
) -> Iterator[Document]:
    """Stream docs, yielding all `required` ones plus a uniform random fill.

    With the corpus size known, fill positions are sampled up front, so the pass
    is a single stream with O(n_docs) memory for the index set and none for text.
    Output size is |required| + fill - overlap (overlap is tiny in practice).
    """
    fill = max(0, n_docs - len(required))
    rng = random.Random(seed)
    positions = set(rng.sample(range(total), min(fill, total)))
    for i, d in enumerate(docs):
        if d.id in required or i in positions:
            yield d


def build(
    out_dir: Path,
    corpus: str,
    eval_sets: list[str],
    n_docs: int | None = None,
    n_queries: int | None = None,
    seed: int = 13,
) -> dict:
    """Build a collection directory. n_docs=None means the full corpus."""
    for es in eval_sets:
        if get_eval_set(es).corpus != corpus:
            raise ValueError(f"Eval set {es!r} belongs to corpus {get_eval_set(es).corpus!r}, not {corpus!r}")

    loaded = {es: load_eval_set(es, n_queries, seed) for es in eval_sets}
    required = {d for _, qrels in loaded.values() for per_q in qrels.values() for d in per_q}

    docs, total = _iter_corpus(corpus)
    if total is None:  # rare: corpus size unknown, count with an extra pass
        total = sum(1 for _ in _iter_corpus(corpus)[0])
        docs, _ = _iter_corpus(corpus)

    is_subset = n_docs is not None and n_docs < total
    if is_subset:
        docs = select_subset(docs, total, required, n_docs, seed)

    docs = tqdm(docs, total=total if not is_subset else None, desc=f"writing {Path(out_dir).name}", unit="doc")
    manifest = {
        "corpus": corpus,
        "corpus_ir_datasets_id": CORPORA[corpus],
        "corpus_total_docs": total,
        "subset": is_subset,
        "n_docs_requested": n_docs,
        "n_queries_requested": n_queries,
        "seed": seed,
        "eval_set_sources": {es: get_eval_set(es).ir_datasets_id for es in eval_sets},
        "note": (
            "Subset = all judged passages for sampled queries + uniform random fill. "
            "Easier than the full corpus; use for iteration and relative comparisons only."
            if is_subset
            else "Full corpus."
        ),
    }
    return write_collection(Path(out_dir), docs, loaded, manifest)
