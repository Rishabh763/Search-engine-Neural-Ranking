"""Standard latency benchmark: run the same suite on any machine and compare.

python scripts/benchmark.py --collection msmarco-100k

Measures, on real queries and their real BM25 candidates:
  - first-stage latency per query: BM25 (Elasticsearch, local), dense (each FAISS
    index that exists), hybrid RRF
  - reranker latency per query at several depths, for every variant this machine
    can run: PyTorch fp32 on CPU (always, for like-for-like CPU comparison), fp16 on
    an NVIDIA GPU, fp32 on an Apple GPU (MPS), PyTorch dynamic int8 (CPU), ONNX fp32 /
    int8 (if exported with scripts/export_onnx.py)

Writes results/benchmarks/<hostname>_<timestamp>.json with the hardware
fingerprint. Commit those files: results from different machines then sit side
by side (scripts/compare_benchmarks.py prints them as one table).
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from searchengine.config import PROJECT_ROOT, get_settings
from searchengine.data.collection import Collection
from searchengine.docstore import DocStore
from searchengine.envinfo import environment_info, short_label
from searchengine.pipeline import passage_text
from searchengine.ranking.cross_encoder import DEFAULT_MODEL


def timeit(fn, items, warmup: int = 3) -> dict:
    for x in items[:warmup]:
        fn(x)
    t = []
    for x in items:
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(t)
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "n": len(t)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", default="msmarco-100k")
    p.add_argument("--eval-set", default="msmarco-dev")
    p.add_argument("--queries", type=int, default=50, help="queries for first-stage timing")
    p.add_argument("--rerank-queries", type=int, default=10, help="queries per reranker variant and depth")
    p.add_argument("--depths", type=int, nargs="+", default=[20, 100])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--onnx-dir", default=None, help="default: models/<model-name>-onnx if it exists")
    p.add_argument("--skip", nargs="*", default=[], choices=["bm25-es", "bm25-local", "dense", "hybrid", "rerank"])
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    env = environment_info()
    print(f"machine: {short_label(env)}")
    qs = coll.queries(args.eval_set if args.eval_set in coll.eval_set_names else coll.eval_set_names[0])
    qtexts = [qs[q] for q in random.Random(0).sample(sorted(qs), min(args.queries, len(qs)))]
    results: dict = {"collection": coll.name, "environment": env, "first_stage": {}, "rerank": {}}

    from searchengine.factory import build_retriever

    first_stage = {}

    def try_stage(name, make):
        if name.split(":")[0] in args.skip:
            return
        try:
            r = make()
        except Exception as e:
            print(f"  {name}: unavailable ({type(e).__name__}: {str(e)[:80]})")
            return
        first_stage[name] = r
        for k in (100, 1000):
            res = timeit(lambda q: r.search(q, k), qtexts)
            results["first_stage"][f"{name}@k{k}"] = res
            print(f"  {name:22s} k={k:<5d} p50={res['p50_ms']:8.2f} ms   p95={res['p95_ms']:8.2f} ms")

    print("first stage (per query):")
    try_stage("bm25-es", lambda: build_retriever("es", coll, s))
    try_stage("bm25-local", lambda: build_retriever("local", coll, s))
    for kind in ("flat", "hnsw", "ivfpq"):
        try_stage(f"dense:{kind}", lambda kind=kind: build_retriever("dense", coll, s, dense_index=kind))
    if "bm25-es" in first_stage and "dense:flat" in first_stage and "hybrid" not in args.skip:
        from searchengine.retrieval.hybrid import RRFHybrid

        try_stage("hybrid:es+flat", lambda: RRFHybrid([first_stage["bm25-es"], first_stage["dense:flat"]]))

    if "rerank" not in args.skip:
        # Real candidate slates: BM25 top-max(depth) passages for a few queries.
        bm25 = first_stage.get("bm25-es") or first_stage.get("bm25-local")
        if bm25 is None:
            print("rerank: needs a BM25 index for realistic candidates; skipped")
        else:
            docs = DocStore.open(coll, s.index_dir)
            maxd = max(args.depths)
            slates = []
            for q in qtexts[: args.rerank_queries]:
                ids = [h.doc_id for h in bm25.search(q, maxd)]
                texts = docs.get_documents(ids)
                slates.append((q, [passage_text(texts[d]) for d in ids if d in texts]))
            # CPU variants always run, so CPUs are compared like for like; GPU variants are extra rows.
            variants = [("torch-fp32", dict(model=args.model, quantize=None, device="cpu"))]
            if env.get("cuda_available"):
                variants = [("torch-fp16-cuda", dict(model=args.model, device="cuda")), *variants]
            if env.get("mps_available"):
                variants = [("torch-fp32-mps", dict(model=args.model, device="mps")), *variants]
            variants.append(("torch-int8", dict(model=args.model, quantize="int8", device="cpu")))
            onnx_dir = Path(args.onnx_dir) if args.onnx_dir else PROJECT_ROOT / "models" / f"{Path(args.model).name}-onnx"
            if (onnx_dir / "export_meta.json").exists():
                variants += [("onnx-fp32", dict(model=str(onnx_dir), onnx_variant="fp32")),
                             ("onnx-int8", dict(model=str(onnx_dir), onnx_variant="int8"))]
                from searchengine.ranking.onnx_ce import available_onnx_devices

                for dev in available_onnx_devices():
                    if dev != "cpu":  # GPU providers: fp16 (and fp32 for comparison)
                        variants += [(f"onnx-fp16-{dev}", dict(model=str(onnx_dir), onnx_variant="fp16", device=dev)),
                                     (f"onnx-fp32-{dev}", dict(model=str(onnx_dir), onnx_variant="fp32", device=dev))]
            else:
                print(f"  (no ONNX export at {onnx_dir}; run scripts/export_onnx.py to include it)")
            from searchengine.factory import build_reranker

            print(f"rerank (per query, {len(slates)} queries):")
            for vname, kw in variants:
                device = kw.pop("device", "cpu")
                try:
                    rr = build_reranker(kw.pop("model"), device=device, **kw)
                except Exception as e:
                    print(f"  {vname}: unavailable ({type(e).__name__}: {str(e)[:80]})")
                    continue
                for depth in args.depths:
                    res = timeit(lambda x: rr.score(x[0], x[1][:depth]), slates, warmup=1)
                    res["pairs_per_s"] = round(depth / (res["p50_ms"] / 1000), 1)
                    results["rerank"][f"{vname}@{depth}"] = res
                    print(f"  {vname:16s} depth={depth:<4d} p50={res['p50_ms']:8.1f} ms   p95={res['p95_ms']:8.1f} ms   {res['pairs_per_s']:7.0f} pairs/s")

    out_dir = PROJECT_ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    host = (env.get("hostname") or "host").replace(" ", "_")
    path = out_dir / f"{host}_{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
