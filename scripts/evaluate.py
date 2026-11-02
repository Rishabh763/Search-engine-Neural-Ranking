"""Run a retriever over a collection's eval sets, write TREC runs and metrics.

python scripts/evaluate.py --collection msmarco-100k --retriever bm25-es
python scripts/evaluate.py --collection msmarco-100k --retriever bm25-local --eval-sets dl19 dl20
python scripts/evaluate.py --collection msmarco-100k --retriever dense --dense-index flat
python scripts/evaluate.py --collection msmarco-100k --retriever hybrid-es

Outputs, per eval set, under RUNS_DIR/<collection>/<eval_set>/:
    <tag>.run    TREC run file (top --k per query)
    <tag>.json   config, mean metrics, per-query metrics, latency percentiles
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.envinfo import environment_info, short_label
from searchengine.data.collection import Collection
from searchengine.data.registry import spec_or_default
from searchengine.eval.latency import measure_latency
from searchengine.eval.metrics import evaluate
from searchengine.eval.trec import write_run


def load_retriever(args, coll, s):
    if args.retriever == "bm25-local":
        from searchengine.retrieval.bm25_local import LocalBM25

        r = LocalBM25.load(s.index_dir / coll.name / "bm25-local", k1=args.k1, b=args.b)
        return r, {"k1": r.k1, "b": r.b}
    if args.retriever == "bm25-es":
        from searchengine.index.es import get_bm25_params, make_client, set_bm25_params
        from searchengine.retrieval.bm25_es import ElasticsearchBM25

        es = make_client(s.es_url)
        name = args.index_name or coll.name
        if args.k1 is not None or args.b is not None:
            cur = get_bm25_params(es, name)
            set_bm25_params(es, name, args.k1 if args.k1 is not None else cur["k1"], args.b if args.b is not None else cur["b"])
        return ElasticsearchBM25(es, name), get_bm25_params(es, name)
    from searchengine.factory import build_retriever

    backend = {"dense": "dense", "hybrid-es": "hybrid-es", "hybrid-local": "hybrid-local"}[args.retriever]
    r = build_retriever(
        backend, coll, s, index_name=args.index_name, dense_model=args.dense_model, dense_index=args.dense_index,
        device=args.device, nprobe=args.nprobe, ef_search=args.ef_search, rrf_k=args.rrf_k, fusion_depth=args.k,
    )
    params = {"dense_model": args.dense_model, "dense_index": args.dense_index}
    if args.dense_index == "ivfpq":
        params["nprobe"] = args.nprobe
    if args.dense_index == "hnsw":
        params["ef_search"] = args.ef_search
    if backend.startswith("hybrid"):
        params["rrf_k"] = args.rrf_k
    return r, params


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--retriever", choices=["bm25-local", "bm25-es", "dense", "hybrid-es", "hybrid-local"], required=True)
    p.add_argument("--eval-sets", nargs="*", default=None, help="default: all in the collection")
    p.add_argument("--k", type=int, default=1000, help="retrieval depth")
    p.add_argument("--k1", type=float, default=None)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--index-name", default=None)
    p.add_argument("--dense-model", default=None, help="default: DENSE_MODEL env or msmarco-MiniLM-L6-cos-v5")
    p.add_argument("--dense-index", choices=["flat", "hnsw", "ivfpq"], default="flat")
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--ef-search", type=int, default=128)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--device", default="auto")
    p.add_argument("--latency-queries", type=int, default=200, help="queries timed one-by-one (0 = skip)")
    p.add_argument("--tag", default=None, help="output name (default: retriever name)")
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    retriever, params = load_retriever(args, coll, s)
    env = environment_info()
    print(f"machine: {short_label(env)}")
    tag = args.tag or retriever.name

    for es_name in args.eval_sets or coll.eval_set_names:
        spec = spec_or_default(es_name)
        queries, qrels = coll.queries(es_name), coll.qrels(es_name)

        t0 = time.perf_counter()
        run = retriever.batch_search(queries, k=args.k)
        batch_secs = time.perf_counter() - t0

        res = evaluate(run, qrels, spec.metrics, rel_threshold=spec.rel_threshold)
        lat = measure_latency(lambda q: retriever.search(q, args.k), list(queries.values())[: args.latency_queries]) if args.latency_queries else {}

        out_dir = s.runs_dir / coll.name / es_name
        write_run(run, out_dir / f"{tag}.run", tag=tag)
        report = {
            "collection": coll.name,
            "eval_set": es_name,
            "tag": tag,
            "retriever": retriever.name,
            "params": params,
            "k": args.k,
            "rel_threshold": spec.rel_threshold,
            "primary_metric": spec.primary_metric,
            "n_queries": res.n_queries,
            "metrics": res.mean,
            "batch_seconds": round(batch_secs, 3),
            "batch_qps": round(len(queries) / batch_secs, 1) if batch_secs else None,
            "latency": lat,
            "environment": env,
            "per_query": res.per_query,
        }
        with open(out_dir / f"{tag}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)

        lat_s = f"  p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms p99={lat['p99_ms']:.1f}ms" if lat else ""
        print(f"[{coll.name}/{es_name}] {tag} {params}\n  {res.format()}{lat_s}")


if __name__ == "__main__":
    main()
