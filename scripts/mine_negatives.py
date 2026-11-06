"""Mine hard negatives for the training queries with a first-stage retriever.

python scripts/mine_negatives.py --collection msmarco-full --source dense-exact      # GPU (Kaggle)
python scripts/mine_negatives.py --collection msmarco-100k --source bm25-local

Writes <RUNS_DIR>/<collection>/training/negatives-<source>.jsonl (+ .stats.json with
how often the judged positive was retrieved at all, a useful sanity number).
"""

import argparse
import json

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.factory import build_retriever
from searchengine.training.negatives import mine_negatives


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--source", choices=["dense-exact", "dense", "bm25-local", "bm25-es"], default="dense-exact")
    p.add_argument("--dense-model", default=None)
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--skip-top", type=int, default=3, help="skip the top non-positives (likely unjudged relevant)")
    p.add_argument("--n-neg", type=int, default=15, help="negatives kept per query (training samples from these)")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    backend = {"dense-exact": "dense-exact", "dense": "dense", "bm25-local": "local", "bm25-es": "es"}[args.source]
    retriever = build_retriever(backend, coll, s, dense_model=args.dense_model, device=args.device)
    queries, qrels = coll.train_queries(), coll.train_qrels()
    if args.max_queries:
        keep = sorted(queries)[: args.max_queries]
        queries = {q: queries[q] for q in keep}
    out = args.out or s.runs_dir / coll.name / "training" / f"negatives-{args.source}.jsonl"
    print(f"mining with {retriever.name} for {len(queries)} training queries -> {out}")
    stats = mine_negatives(retriever, queries, qrels, out, depth=args.depth, skip_top=args.skip_top, n_neg=args.n_neg)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
