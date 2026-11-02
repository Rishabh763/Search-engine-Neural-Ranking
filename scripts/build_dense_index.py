"""Build a FAISS index over encoded passages and measure its ANN recall vs exact search.

python scripts/build_dense_index.py --collection msmarco-100k --type flat
python scripts/build_dense_index.py --collection msmarco-100k --type hnsw
python scripts/build_dense_index.py --collection msmarco-100k --type ivfpq --nprobe 32

For approximate indexes (hnsw, ivfpq) the eval queries are searched both exactly
(streaming brute force over the shards) and with the index, and the overlap of the
two top-k lists is reported: "recall vs exact". That isolates the approximation
error from the model's own retrieval quality.
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.dense.encoder import DEFAULT_DENSE_MODEL, BiEncoder, model_slug
from searchengine.dense.index import EmbeddingStore, build_faiss_index, dense_dir, exact_topk, recall_vs_exact, set_search_params
from searchengine.envinfo import environment_info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    p.add_argument("--type", choices=["flat", "hnsw", "ivfpq"], default="flat")
    p.add_argument("--nlist", type=int, default=None)
    p.add_argument("--pq-m", type=int, default=48)
    p.add_argument("--nbits", type=int, default=8)
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--nprobe", type=int, default=32, help="ivfpq: cells searched per query (recall check)")
    p.add_argument("--ef-search", type=int, default=128, help="hnsw: candidate list size (recall check)")
    p.add_argument("--recall-queries", type=int, default=300, help="0 = skip the ANN recall check")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import faiss

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    root = dense_dir(s.index_dir, coll.name, model_slug(args.model))
    store = EmbeddingStore(root)
    print(f"{store.n} vectors x {store.dim} from {root}")
    index, params = build_faiss_index(
        store, args.type, nlist=args.nlist, pq_m=args.pq_m, nbits=args.nbits,
        hnsw_m=args.hnsw_m, ef_construction=args.ef_construction,
    )
    path = root / f"{args.type}.faiss"
    faiss.write_index(index, str(path))
    params["file_mb"] = round(path.stat().st_size / 2**20, 1)
    print(f"built {args.type} in {params['build_seconds']}s, {params['file_mb']} MB")

    if args.recall_queries and args.type != "flat":
        enc = BiEncoder(store.meta["encoder"]["model"], device=args.device, max_length=store.meta["encoder"].get("max_length"))
        qs = [q for e in coll.eval_set_names for q in coll.queries(e).values()][: args.recall_queries]
        Q = enc.encode_queries(qs)
        k = 100
        exact_scores, _ = exact_topk(store, Q, k)
        set_search_params(index, args.nprobe, args.ef_search, k)
        t0 = time.perf_counter()
        _, approx = index.search(Q, k)
        params["search_ms_per_query"] = round((time.perf_counter() - t0) * 1000 / len(Q), 3)
        for kk in (10, 100):
            params[f"recall@{kk}_vs_exact"] = round(recall_vs_exact(store, Q, exact_scores, approx, kk), 4)
        params.update(recall_check_queries=len(qs), nprobe=args.nprobe if args.type == "ivfpq" else None,
                      ef_search=args.ef_search if args.type == "hnsw" else None)
        print(f"ANN recall vs exact (tie-aware): @10={params['recall@10_vs_exact']}  @100={params['recall@100_vs_exact']}"
              f"   search {params['search_ms_per_query']} ms/query (batched)")
    params["environment"] = environment_info()
    with open(root / f"{args.type}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
