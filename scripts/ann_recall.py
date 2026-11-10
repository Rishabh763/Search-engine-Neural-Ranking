"""Measure an existing ANN index against exact search, with and without refinement.

python scripts/ann_recall.py --collection msmarco-full --index ivfpq96 --nprobe 64 --refine 0 100 200 500 1000
python scripts/ann_recall.py --collection msmarco-full --index ivfpq48 --nprobe 32 64 128 --refine 0 1000

For each (nprobe, refine) pair it reports, on real eval queries:
    recall@10 / recall@100 vs exact   tie-aware (searchengine.dense.index.recall_vs_exact)
    batch ms/query                    all queries in one call (offline throughput)
    single p50 / p95 ms               one query at a time (serving), on queries not used
                                      anywhere else in the run (see below)

Ground truth is computed here by exact search over the stored vectors (CPU, streamed
shard by shard: on 8.8M x 384 and 1,000 queries this takes a few minutes). Query
encoding time is excluded everywhere: it is the same for every configuration.

Latency includes the reads of refinement candidates from the memory-mapped shards,
which is the cost that matters: a new query touches rows nobody has read yet. So each
configuration is timed on its own slice of fresh queries, disjoint from the recall
queries and from every other configuration's slice (otherwise the recall pass would
have pulled exactly those rows into the page cache and the timing would flatter
refinement). A few other queries warm up threads and the index first. Rows can still
be cached from earlier runs of any script; re-run after a reboot for a cold number.
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.dense.encoder import BiEncoder, default_dense_model, model_slug
from searchengine.dense.index import (
    EmbeddingStore, VectorRows, dense_dir, exact_topk, recall_vs_exact, refine_topk, set_search_params,
)
from searchengine.envinfo import environment_info, short_label


def search(index, vectors, Q, k, refine):
    depth = max(k, refine) if refine else k
    _, I = index.search(Q, depth)
    if refine:
        _, I = refine_topk(Q, I, vectors, k)
    return I


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--model", default=None, help="default: DENSE_MODEL env or msmarco-MiniLM-L6-cos-v5")
    p.add_argument("--index", required=True, help="index file name, e.g. ivfpq96 (for ivfpq96.faiss)")
    p.add_argument("--eval-set", default=None, help="queries from this eval set (default: the first one)")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--nprobe", type=int, nargs="+", default=[64])
    p.add_argument("--ef-search", type=int, default=128)
    p.add_argument("--refine", type=int, nargs="+", default=[0, 100, 200, 500, 1000])
    p.add_argument("--latency-queries", type=int, default=200, help="fresh queries timed per configuration")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import faiss

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    root = dense_dir(s.index_dir, coll.name, model_slug(args.model or default_dense_model()))
    store = EmbeddingStore(root)
    vectors = VectorRows(store)
    index = faiss.read_index(str(root / f"{args.index}.faiss"))
    env = environment_info()
    print(f"machine: {short_label(env)}\n{store.n} vectors x {store.dim}, index {args.index} ({type(index).__name__})")

    es_name = args.eval_set or coll.eval_set_names[0]
    all_texts = list(coll.queries(es_name).values())
    n_cfg = len(args.nprobe) * len(args.refine)
    warm_n = 20
    n_lat = max(0, min(args.latency_queries, (len(all_texts) - args.queries - warm_n) // max(n_cfg, 1)))
    if n_lat < args.latency_queries:
        print(f"note: only {n_lat} fresh latency queries per configuration fit in {es_name}")
    texts = all_texts[: args.queries]
    extra = all_texts[args.queries : args.queries + warm_n + n_cfg * n_lat]
    enc_cfg = store.meta["encoder"]
    enc = BiEncoder(enc_cfg["model"], device=args.device, max_length=enc_cfg.get("max_length"),
                    query_prefix=enc_cfg.get("query_prefix"), passage_prefix=enc_cfg.get("passage_prefix"))
    Q = np.ascontiguousarray(enc.encode_queries(texts), dtype=np.float32)
    Q_extra = np.ascontiguousarray(enc.encode_queries(extra), dtype=np.float32) if extra else np.zeros((0, store.dim), np.float32)
    Q_warm, Q_lat = Q_extra[:warm_n], Q_extra[warm_n:]

    t0 = time.perf_counter()
    exact_s, _ = exact_topk(store, Q, 100)
    print(f"exact ground truth for {len(Q)} {es_name} queries in {time.perf_counter() - t0:.0f}s")

    rows = []
    cfg_i = 0
    for nprobe in args.nprobe:
        for refine in args.refine:
            lat_q = Q_lat[cfg_i * n_lat : (cfg_i + 1) * n_lat]
            cfg_i += 1
            set_search_params(index, nprobe, args.ef_search, max(100, refine))
            t0 = time.perf_counter()
            I = search(index, vectors, Q, 100, refine)
            batch_ms = (time.perf_counter() - t0) * 1000 / len(Q)
            r10 = recall_vs_exact(store, Q, exact_s, I, 10, vectors=vectors)
            r100 = recall_vs_exact(store, Q, exact_s, I, 100, vectors=vectors) if refine == 0 or refine >= 100 else None
            for q in Q_warm:  # warm-up: FAISS thread pool, index pages (not lat_q's rows)
                search(index, vectors, q[None], 10, refine)
            times = []
            for q in lat_q:
                t = time.perf_counter()
                search(index, vectors, q[None], 10, refine)
                times.append((time.perf_counter() - t) * 1000)
            row = {"nprobe": nprobe, "refine": refine, "recall@10": round(r10, 4),
                   "recall@100": None if r100 is None else round(r100, 4), "batch_ms_per_query": round(batch_ms, 3),
                   "single_p50_ms": round(float(np.percentile(times, 50)), 2) if times else None,
                   "single_p95_ms": round(float(np.percentile(times, 95)), 2) if times else None}
            rows.append(row)
            r100_s = "  -   " if r100 is None else f"{r100:.4f}"
            lat_s = (f"single p50 {row['single_p50_ms']:7.2f} ms  p95 {row['single_p95_ms']:7.2f} ms" if times else "")
            print(f"nprobe={nprobe:<4} refine={refine:<5} recall@10={r10:.4f}  recall@100={r100_s}  "
                  f"batch {batch_ms:6.2f} ms/q   {lat_s}")

    out = root / f"{args.index}.refine.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"index": args.index, "eval_set": es_name, "n_queries": len(Q), "latency_queries_per_config": n_lat,
                   "latency_method": "fresh queries per configuration, disjoint from recall queries",
                   "results": rows, "environment": env}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
