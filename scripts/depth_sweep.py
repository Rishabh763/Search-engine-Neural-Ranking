"""Quality and latency vs reranking depth (how many BM25 candidates the reranker sees).

Quality for every depth comes from one existing depth-D reranked run (no extra
model calls, see searchengine/eval/sweep.py). Latency is measured: for a sample
of queries, the reranker scores the top-d passages, timed per query.

python scripts/depth_sweep.py --collection msmarco-100k --reranked "bm25-es+ms-marco-MiniLM-L-6-v2@100"

The reranker used for timing is rebuilt from the reranked run's report, so
quality and latency always describe the same model variant. Output:
runs/<collection>/sweeps/<reranked-tag>__<hostname>.json and .csv; plot several
with scripts/plot_tradeoff.py.
"""

import argparse
import csv
import json
import random
import time

import numpy as np

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.data.registry import spec_or_default
from searchengine.docstore import DocStore
from searchengine.envinfo import environment_info, short_label
from searchengine.eval.metrics import evaluate, rank_documents
from searchengine.eval.sweep import run_at_depth
from searchengine.eval.trec import read_run, safe_name
from searchengine.factory import build_reranker
from searchengine.pipeline import passage_text


def reranker_from_report(info: dict, device: str):
    variant = info.get("variant")
    if info.get("runtime") == "onnxruntime":
        raise SystemExit("For ONNX runs pass --model <export dir> --onnx-variant explicitly")
    return build_reranker(
        info["model"], device=device, batch_size=info.get("batch_size", 64),
        max_length=info.get("max_length", 512), quantize=info.get("quantize"), onnx_variant=variant or "int8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--reranked", required=True, help="tag of a reranked run (from rerank.py)")
    p.add_argument("--depths", type=int, nargs="+", default=[5, 10, 20, 30, 50, 100])
    p.add_argument("--eval-sets", nargs="*", default=None, help="default: those the reranked run exists for")
    p.add_argument("--latency-queries", type=int, default=30, help="0 = skip timing")
    p.add_argument("--latency-eval-set", default="dl19")
    p.add_argument("--model", default=None, help="override the reranker used for timing (e.g. an ONNX dir)")
    p.add_argument("--onnx-variant", default="int8")
    p.add_argument("--quantize", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    s = get_settings()
    args.reranked = safe_name(args.reranked)
    coll = Collection.from_name(args.collection, s.data_dir)
    sets = args.eval_sets or [
        e for e in coll.eval_set_names if (s.runs_dir / coll.name / e / f"{args.reranked}.json").exists()
    ]
    if not sets:
        raise SystemExit(f"no reranked run {args.reranked!r} found under {s.runs_dir / coll.name}")

    quality: dict[int, dict[str, dict]] = {d: {} for d in args.depths}
    report0 = None
    for es_name in sets:
        d = s.runs_dir / coll.name / es_name
        with open(d / f"{args.reranked}.json", encoding="utf-8") as f:
            rep = json.load(f)
        report0 = report0 or rep
        spec = spec_or_default(es_name)
        first = read_run(d / f"{safe_name(rep['first_stage'])}.run")
        reranked = read_run(d / f"{args.reranked}.run")
        qrels = coll.qrels(es_name)
        queries = set(reranked)
        first = {q: first.get(q, {}) for q in queries}
        quality_bm25 = evaluate(first, qrels, spec.metrics, rel_threshold=spec.rel_threshold).mean
        for depth in [0] + args.depths:
            if depth > rep["depth"]:
                continue
            run = first if depth == 0 else run_at_depth(first, reranked, depth, rep["depth"])
            m = quality_bm25 if depth == 0 else evaluate(run, qrels, spec.metrics, rel_threshold=spec.rel_threshold).mean
            quality.setdefault(depth, {})[es_name] = m

    # ---- latency ----
    env = environment_info()
    latency: dict[int, dict] = {}
    info = report0["reranker"]
    if args.latency_queries:
        reranker = (
            build_reranker(args.model, args.device, quantize=args.quantize, onnx_variant=args.onnx_variant)
            if args.model else reranker_from_report(info, args.device)
        )
        info = reranker.info()
        les = args.latency_eval_set if args.latency_eval_set in coll.eval_set_names else sets[0]
        first = read_run(s.runs_dir / coll.name / les / f"{safe_name(report0['first_stage'])}.run")
        queries = coll.queries(les)
        qids = random.Random(0).sample(sorted(first), min(args.latency_queries, len(first)))
        docs = DocStore.open(coll, s.index_dir)
        maxd = max(args.depths)
        slates = {}
        for q in qids:
            top = rank_documents(first[q])[:maxd]
            texts = docs.get_documents(top)
            slates[q] = [passage_text(texts[x]) for x in top if x in texts]
        reranker.score(queries[qids[0]], slates[qids[0]][:20])  # warmup
        print(f"timing {reranker.name} on {len(qids)} queries per depth ({short_label(env)})")
        for depth in args.depths:
            t = []
            for q in qids:
                t0 = time.perf_counter()
                reranker.score(queries[q], slates[q][:depth])
                t.append((time.perf_counter() - t0) * 1000)
            a = np.asarray(t)
            latency[depth] = {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}

    # ---- report ----
    pm = {e: spec_or_default(e).primary_metric for e in sets}
    rows = []
    print(f"\n{'depth':>5}  " + "  ".join(f"{e}:{pm[e]:>9}" for e in sets) + "   rerank p50 / p95 (ms)")
    for depth in sorted(quality):
        if not quality[depth]:
            continue
        lat = latency.get(depth, {})
        row = {"depth": depth, **{f"{e}:{pm[e]}": quality[depth][e][pm[e]] for e in sets if e in quality[depth]}}
        row.update({"p50_ms": lat.get("p50_ms"), "p95_ms": lat.get("p95_ms")})
        rows.append(row)
        cells = "  ".join(f"{quality[depth][e][pm[e]]:>{len(e) + len(pm[e]) + 1}.4f}" for e in sets)
        lat_s = f"{lat['p50_ms']:8.0f} / {lat['p95_ms']:.0f}" if lat else ("  (first stage only, no reranking)" if depth == 0 else "")
        print(f"{depth:>5}  {cells}   {lat_s}")

    out_dir = s.runs_dir / coll.name / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(f"{args.reranked}__{env.get('hostname', 'host')}")
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {"collection": coll.name, "reranked": args.reranked, "first_stage": report0["first_stage"],
             "reranker": info, "environment": env, "primary_metrics": pm,
             "quality": {str(k): v for k, v in quality.items() if v}, "latency": {str(k): v for k, v in latency.items()}},
            f, indent=2,
        )
    with open(out_dir / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_dir / stem}.json/.csv")


if __name__ == "__main__":
    main()
