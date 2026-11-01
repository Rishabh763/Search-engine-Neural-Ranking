r"""In-process BM25 over a sparse inverted index (SciPy CSR).

Purpose: a transparent reference implementation to cross-check Elasticsearch, to
run unit tests without a server, and to make the scoring math explicit. For the
full 8.8M corpus use Elasticsearch; this one is meant for up to ~1M passages.

Scoring (textbook Robertson BM25, which is also exactly what Elasticsearch computes):

    score(q, d) = sum_{t in q} qtf(t) * IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

About (k1 + 1): Lucene 8 removed it from its BM25Similarity (it scales every
score by the same constant, so rankings are unchanged), but Elasticsearch keeps it
through its own LegacyBM25Similarity for score compatibility. scripts/check_bm25_parity.py
caught this: without the factor, every local score was exactly 1/(k1+1) = 0.526 of
ES's. qtf(t) is the term's count in the query (Lucene builds one SHOULD clause per
query token, so repeated terms add up the same way).
Document length: Lucene stores |d| in a single byte (SmallFloat.intToByte4):
exact below 40 tokens, then rounded down to 4 significant bits, so a 55-token
passage is scored as 54 and a 100-token one as 96 (avgdl stays exact). With
lucene_norms=True (default) we apply the same quantization, so scores match ES
to float32 precision; lucene_norms=False gives textbook BM25 with exact lengths.

Index layout ("impact" index)
-----------------------------
TF is a CSR matrix of shape (V, N): row t is the postings list of term t
(doc indices + term frequencies). Applying BM25 to each nonzero gives the impact
matrix W of the same shape, precomputed once. A query is then a sparse 1 x V
vector q of term counts, and all document scores are one sparse product

    s = q @ W        (1 x N, nonzero only for docs containing a query term)

which is exactly term-at-a-time evaluation: only postings of query terms are
touched. Changing k1 or b only recomputes W from TF (no re-tokenizing), which is
what makes the k1/b grid search cheap.
"""

from __future__ import annotations

import json
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from searchengine.retrieval.base import Retriever
from searchengine.text import Analyzer
from searchengine.types import Document, Hit, Queries, Run


# --- Lucene SmallFloat.intToByte4 / byte4ToInt (org.apache.lucene.util.SmallFloat) ---
# 4-bit float codec: 3 mantissa bits + implicit leading 1, shifted. The first 24
# byte values are reserved for exact small integers.
_NUM_FREE_VALUES = 24  # 255 - longToInt4(Integer.MAX_VALUE)


def _long_to_int4(i: int) -> int:
    num_bits = i.bit_length()
    if num_bits < 4:
        return i  # "subnormal": stored exactly
    shift = num_bits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i: int) -> int:
    bits, shift = i & 0x07, (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


def lucene_quantize_length(n: int) -> int:
    """The document length Lucene's BM25 actually uses for a field of n tokens."""
    if n < _NUM_FREE_VALUES:
        return n
    return _NUM_FREE_VALUES + _int4_to_long(_long_to_int4(n - _NUM_FREE_VALUES))


def _quantize_lengths(lengths: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(lengths.astype(np.int64), return_inverse=True)
    table = np.array([lucene_quantize_length(int(u)) for u in uniq], dtype=np.float32)
    return table[inv].reshape(lengths.shape)


class LocalBM25(Retriever):
    name = "bm25-local"

    def __init__(
        self,
        doc_ids: list[str],
        vocab: dict[str, int],
        tf: sp.csr_matrix,
        doc_len: np.ndarray,
        k1: float = 0.9,
        b: float = 0.4,
        analyzer: Analyzer | None = None,
        lucene_norms: bool = True,
    ):
        self.doc_ids = doc_ids
        self.lucene_norms = lucene_norms
        self.vocab = vocab
        self.tf = tf.tocsr()
        self.doc_len = doc_len.astype(np.float32)
        self.analyzer = analyzer or Analyzer()
        self.n_docs = len(doc_ids)
        # Lucene's docCount: docs that actually have tokens in the field.
        self.doc_count = int(np.count_nonzero(self.doc_len))
        self.avgdl = float(self.doc_len.sum() / max(self.doc_count, 1))
        self.df = np.diff(self.tf.indptr).astype(np.float64)  # postings length per term
        self.idf = np.log1p((self.doc_count - self.df + 0.5) / (self.df + 0.5))
        self.norm_len = _quantize_lengths(self.doc_len) if lucene_norms else self.doc_len
        self._id_to_idx: dict[str, int] | None = None
        self.set_params(k1, b)

    # ---------------- construction ----------------

    @classmethod
    def build(
        cls,
        docs: Iterable[Document],
        k1: float = 0.9,
        b: float = 0.4,
        analyzer: Analyzer | None = None,
        total: int | None = None,
        lucene_norms: bool = True,
    ) -> "LocalBM25":
        analyzer = analyzer or Analyzer()
        vocab: dict[str, int] = {}
        doc_ids: list[str] = []
        lens = array("i")
        rows, cols, vals = array("i"), array("i"), array("i")  # compact buffers, not Python lists
        for j, d in enumerate(tqdm(docs, total=total, desc="indexing", unit="doc")):
            doc_ids.append(d.id)
            toks = analyzer(f"{d.title} {d.text}" if d.title else d.text)
            lens.append(len(toks))
            for term, c in Counter(toks).items():
                tid = vocab.setdefault(term, len(vocab))
                rows.append(tid)
                cols.append(j)
                vals.append(c)
        tf = sp.csr_matrix(
            (np.frombuffer(vals, dtype=np.int32), (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
            shape=(len(vocab), len(doc_ids)),
            dtype=np.float32,
        )
        tf.sum_duplicates()
        tf.sort_indices()
        return cls(doc_ids, vocab, tf, np.frombuffer(lens, dtype=np.int32).copy(), k1, b, analyzer, lucene_norms)

    def set_params(self, k1: float, b: float) -> None:
        """Recompute the impact matrix W for new k1, b. Cost: O(nnz), no re-tokenizing."""
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = float(k1), float(b)
        tf = self.tf
        # Column (doc) index of each nonzero, to look up its length normalizer.
        doc_of_nnz = tf.indices
        norm = self.k1 * (1.0 - self.b + self.b * self.norm_len / self.avgdl)  # per doc
        term_of_nnz = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        f = tf.data
        impact = self.idf[term_of_nnz] * f * (self.k1 + 1.0) / (f + norm[doc_of_nnz])
        self.W = sp.csr_matrix((impact.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)

    # ---------------- search ----------------

    def _query_vector(self, text: str) -> sp.csr_matrix | None:
        counts = Counter(t for t in self.analyzer(text) if t in self.vocab)
        if not counts:
            return None
        cols = np.fromiter((self.vocab[t] for t in counts), dtype=np.int32, count=len(counts))
        data = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
        return sp.csr_matrix((data, (np.zeros(len(cols), dtype=np.int32), cols)), shape=(1, len(self.vocab)))

    def score_docs(self, query: str, doc_ids: list[str]) -> np.ndarray:
        """BM25 scores of specific documents (0 for unknown ids or no term match)."""
        if self._id_to_idx is None:
            self._id_to_idx = {d: i for i, d in enumerate(self.doc_ids)}
        out = np.zeros(len(doc_ids), dtype=np.float32)
        q = self._query_vector(query)
        if q is None:
            return out
        dense = (q @ self.W).toarray().ravel()
        for j, d in enumerate(doc_ids):
            i = self._id_to_idx.get(d)
            if i is not None:
                out[j] = dense[i]
        return out

    def _topk_from_row(self, scores: sp.csr_matrix, k: int) -> list[Hit]:
        idx, vals = scores.indices, scores.data
        if idx.size == 0:
            return []
        if idx.size > k:
            part = np.argpartition(-vals, k - 1)[:k]
            idx, vals = idx[part], vals[part]
        # Sort by score desc, then doc id desc: same tie order as trec_eval/rank_documents.
        hits = [Hit(float(v), self.doc_ids[i]) for i, v in zip(idx, vals)]
        hits.sort(key=lambda h: (h.score, h.doc_id), reverse=True)
        return hits

    def search(self, query: str, k: int = 1000) -> list[Hit]:
        q = self._query_vector(query)
        if q is None:
            return []
        return self._topk_from_row(q @ self.W, k)

    def batch_search(self, queries: Queries, k: int = 1000, batch_size: int = 64) -> Run:
        """Vectorized: one sparse (B x V) @ (V x N) product per batch of queries."""
        run: Run = {}
        items = list(queries.items())
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            vecs = [self._query_vector(t) for _, t in chunk]
            nonempty = [i for i, v in enumerate(vecs) if v is not None]
            for i, (qid, _) in enumerate(chunk):
                run[qid] = {}
            if not nonempty:
                continue
            S = (sp.vstack([vecs[i] for i in nonempty]) @ self.W).tocsr()
            for row, i in enumerate(nonempty):
                hits = self._topk_from_row(S.getrow(row), k)
                run[chunk[i][0]] = {h.doc_id: h.score for h in hits}
        return run

    # ---------------- persistence ----------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "tf.npz", self.tf)
        np.save(path / "doc_len.npy", self.doc_len)
        with open(path / "meta.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "lucene_norms": self.lucene_norms,
                    "analyzer": self.analyzer.config(),
                    "n_docs": self.n_docs,
                },
                f,
                indent=2,
            )
        with open(path / "doc_ids.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.doc_ids, f)
        with open(path / "vocab.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path, k1: float | None = None, b: float | None = None) -> "LocalBM25":
        path = Path(path)
        with open(path / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(path / "doc_ids.json", encoding="utf-8") as f:
            doc_ids = json.load(f)
        with open(path / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(
            doc_ids,
            vocab,
            sp.load_npz(path / "tf.npz"),
            np.load(path / "doc_len.npy"),
            meta["k1"] if k1 is None else k1,
            meta["b"] if b is None else b,
            Analyzer.from_config(meta["analyzer"]),
            meta.get("lucene_norms", False),  # indexes saved before this option used exact lengths
        )
