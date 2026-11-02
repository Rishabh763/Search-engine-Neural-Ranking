"""Rerank a saved first-stage run with a cross-encoder and evaluate it.

Reads   runs/<collection>/<eval_set>/<first-stage>.run   (written by evaluate.py)
Writes  runs/<collection>/<eval_set>/<tag>.run and <tag>.json (same report format,
        so compare_runs.py works across first-stage and reranked systems)

python scripts/rerank.py --collection msmarco-100k --first-stage bm25-es --depth 100
python scripts/rerank.py --collection msmarco-100k --first-stage bm25-es --depth 100 --max-queries 200   # quick look

Latency in the report is the rerank stage per query: fetching the candidates'
text + scoring all `depth` pairs (first-stage latency is in the first-stage report).
"""

import argparse
import json
import time

import numpy as np

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.envinfo import environment_info, short_label
from searchengine.data.collection import Collection
from searchengine.data.registry import spec_or_default
from searchengine.docstore import DocStore
from searchengine.eval.metrics import evaluate
from searchengine.eval.trec import read_run, safe_name, write_run
from searchengine.factory import build_reranker
from searchengine.pipeline import rerank_run
from searchengine.ranking.cross_encoder import DEFAULT_MODEL


def reranker_label(name: str) -> str:
    """'ce:X' -> 'X' (plain PyTorch fp32); 'onnx-int8:X' -> 'onnx-int8-X'; 'ce-int8:X' -> 'ce-int8-X'."""
    return name.split(":", 1)[1] if name.startswith("ce:") else name.replace(":", "-")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--first-stage", default="bm25-es", help="tag of the first-stage run to rerank")
    p.add_argument("--eval-sets", nargs="*", default=None)
    p.add_argument("--depth", type=int, default=100, help="candidates reranked per query")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps; ONNX models also: dml (DirectML GPU)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--quantize", choices=["int8"], default=None, help="PyTorch dynamic int8 (CPU)")
    p.add_argument("--onnx-variant", choices=["int8", "fp32", "fp16"], default="int8", help="when --model is an ONNX export dir")
    p.add_argument("--max-queries", type=int, default=None, help="evaluate only the first N queries per set")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    docs = DocStore.open(coll, s.index_dir)
    reranker = build_reranker(args.model, args.device, args.batch_size, args.max_length, args.quantize, args.onnx_variant)
    info = reranker.info()
    env = environment_info()
    print(f"machine: {short_label(env)}")
    print(f"reranker: {info}  (loaded in {reranker.load_seconds:.1f}s)")
    first_stage = safe_name(args.first_stage)
    tag = safe_name(args.tag or f"{first_stage}+{reranker_label(reranker.name)}@{args.depth}")

    for es_name in args.eval_sets or coll.eval_set_names:
        spec = spec_or_default(es_name)
        run_dir = s.runs_dir / coll.name / es_name
        first_path = run_dir / f"{first_stage}.run"
        if not first_path.exists():
            raise SystemExit(f"{first_path} not found: run scripts/evaluate.py for that retriever first")
        queries, qrels = coll.queries(es_name), coll.qrels(es_name)
        if args.max_queries:
            keep = sorted(queries)[: args.max_queries]
            queries = {q: queries[q] for q in keep}
            qrels = {q: qrels[q] for q in keep if q in qrels}
        first = read_run(first_path)

        # Warm up (first batch pays CUDA init / memory allocation / thread pool startup).
        q0 = next(iter(queries))
        rerank_run({q0: first.get(q0, {})}, {q0: queries[q0]}, docs, reranker, depth=args.depth, progress=False)

        t0 = time.perf_counter()
        run, lat = rerank_run(first, queries, docs, reranker, depth=args.depth)
        secs = time.perf_counter() - t0

        res = evaluate(run, qrels, spec.metrics, rel_threshold=spec.rel_threshold)
        base = evaluate({q: first.get(q, {}) for q in queries}, qrels, spec.metrics, rel_threshold=spec.rel_threshold)
        a = np.asarray(lat)
        latency = {
            "n": int(a.size),
            "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
        }
        write_run(run, run_dir / f"{tag}.run", tag=tag)
        report = {
            "collection": coll.name,
            "eval_set": es_name,
            "tag": tag,
            "retriever": tag,
            "first_stage": first_stage,
            "depth": args.depth,
            "reranker": info,
            "rel_threshold": spec.rel_threshold,
            "primary_metric": spec.primary_metric,
            "n_queries": res.n_queries,
            "metrics": res.mean,
            "first_stage_metrics": base.mean,
            "pairs_per_second": round(len(queries) * args.depth / secs, 1),
            "latency": latency,
            "environment": env,
            "per_query": res.per_query,
        }
        with open(run_dir / f"{tag}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)

        pm = spec.primary_metric
        print(
            f"[{coll.name}/{es_name}] {tag}\n  {res.format()}\n"
            f"  {pm}: {base.mean[pm]:.4f} -> {res.mean[pm]:.4f}  ({res.mean[pm] - base.mean[pm]:+.4f})"
            f"   rerank p50={latency['p50_ms']:.0f}ms p95={latency['p95_ms']:.0f}ms"
            f"   {report['pairs_per_second']:.0f} pairs/s"
        )


if __name__ == "__main__":
    main()
