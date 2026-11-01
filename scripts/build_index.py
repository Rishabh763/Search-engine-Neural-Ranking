"""Build a BM25 index for a prepared collection.

python scripts/build_index.py --collection msmarco-100k --backend es
python scripts/build_index.py --collection msmarco-100k --backend local
"""

import argparse
import json

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--backend", choices=["es", "local"], required=True)
    p.add_argument("--k1", type=float, default=0.9)
    p.add_argument("--b", type=float, default=0.4)
    p.add_argument("--index-name", default=None, help="ES index name (default: collection name)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-force-merge", action="store_true")
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)

    if args.backend == "local":
        from searchengine.retrieval.bm25_local import LocalBM25

        out = s.index_dir / coll.name / "bm25-local"
        if out.exists() and not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild")
        import time

        t0 = time.perf_counter()
        bm25 = LocalBM25.build(coll.iter_docs(), args.k1, args.b, total=len(coll))
        secs = time.perf_counter() - t0
        bm25.save(out)
        stats = {"n_docs": bm25.n_docs, "vocab": len(bm25.vocab), "postings": int(bm25.tf.nnz), "seconds": round(secs, 1)}
        print(json.dumps(stats, indent=2), f"\nSaved to {out}")
    else:
        from searchengine.index.es import bulk_index, create_index, make_client

        name = args.index_name or coll.name
        es = make_client(s.es_url)
        create_index(es, name, args.k1, args.b, overwrite=args.overwrite)
        stats = bulk_index(es, name, coll.iter_docs(), force_merge=not args.no_force_merge)
        print(
            f"Indexed {stats.n_docs} docs into {name!r} in {stats.seconds:.1f}s "
            f"({stats.docs_per_second:,.0f} docs/s), errors={stats.n_errors}"
        )


if __name__ == "__main__":
    main()
