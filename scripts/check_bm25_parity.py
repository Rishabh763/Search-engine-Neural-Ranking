"""Score-level parity between the local BM25 and Elasticsearch.

Comparing aggregate metrics (nDCG etc.) can hide or exaggerate differences. This
compares the actual numbers: for every query, take ES's top-k (doc, score) pairs,
score the same docs with the local index, and report
  - relative score error over all (query, doc) pairs: median / p99 / max
  - docs ES returned that the local index scores 0 (a sure sign of analyzer drift)
  - top-k agreement: identical sets, and mean overlap
If analyzer, IDF, avgdl and length quantization all match, errors sit at float32
rounding level (~1e-6) and top-k sets agree except where scores tie.

python scripts/check_bm25_parity.py --collection msmarco-100k
"""

import argparse

import numpy as np

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.index.es import get_bm25_params, make_client
from searchengine.retrieval.bm25_es import ElasticsearchBM25
from searchengine.retrieval.bm25_local import LocalBM25


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--index-name", default=None)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    es = make_client(s.es_url)
    index = args.index_name or coll.name
    params = get_bm25_params(es, index)
    local = LocalBM25.load(s.index_dir / coll.name / "bm25-local", k1=params["k1"], b=params["b"])
    print(f"ES {params}  local k1={local.k1} b={local.b} lucene_norms={local.lucene_norms} analyzer={local.analyzer.config()}")

    queries = {f"{e}/{q}": t for e in coll.eval_set_names for q, t in coll.queries(e).items()}
    es_run = ElasticsearchBM25(es, index).batch_search(queries, k=args.k)

    rel_err, zero_local, same_set, overlap = [], 0, 0, []
    worst: list[tuple[float, str, str]] = []
    for qid, text in queries.items():
        hits = es_run[qid]
        if not hits:
            continue
        ids = list(hits)
        es_scores = np.array([hits[d] for d in ids], dtype=np.float64)
        ours = local.score_docs(text, ids).astype(np.float64)
        zero_local += int(np.sum(ours == 0))
        err = np.abs(ours - es_scores) / np.maximum(es_scores, 1e-9)
        rel_err.extend(err.tolist())
        worst.append((float(err.max()), qid, text))
        local_top = {h.doc_id for h in local.search(text, args.k)}
        same_set += int(local_top == set(ids))
        overlap.append(len(local_top & set(ids)) / len(ids))

    e = np.asarray(rel_err)
    n = len(overlap)
    print(f"\n{n} queries, {e.size} (query, doc) pairs from ES top-{args.k}")
    print(f"  relative score error: median={np.median(e):.2e}  p99={np.percentile(e, 99):.2e}  max={e.max():.2e}")
    print(f"  ES hits the local index scores 0: {zero_local}")
    print(f"  identical top-{args.k} sets: {same_set}/{n} ({100 * same_set / n:.1f}%)   mean overlap: {np.mean(overlap):.4f}")
    print("\nLargest per-query errors:")
    for err, qid, text in sorted(worst, reverse=True)[:8]:
        print(f"  {err:.2e}  {qid:28s} {text[:70]}")


if __name__ == "__main__":
    main()
