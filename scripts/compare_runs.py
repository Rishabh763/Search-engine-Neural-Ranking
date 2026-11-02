"""Paired significance test between two evaluated systems on one metric.

Two report files (one eval set):
    python scripts/compare_runs.py runs/msmarco-100k/dl19/bm25-es.json runs/msmarco-100k/dl19/other.json

Two run tags, pooled over several eval sets (more queries = more statistical power):
    python scripts/compare_runs.py --collection msmarco-100k --sets dl19 dl20 \
        --a "bm25-es+onnx-fp16-dml-ms-marco-MiniLM-L-6-v2@100" \
        --b "dense-flat-msmarco-MiniLM-L6-cos-v5+onnx-fp16-dml-ms-marco-MiniLM-L-6-v2@100"

Pooling is valid when the sets use the same metric (TREC DL 2019 and 2020 both use
nDCG@10 with the same judging scheme) and queries don't overlap (they don't).
"""

import argparse
import json

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.eval.significance import paired_randomization_test
from searchengine.eval.trec import safe_name


def load(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a_file", nargs="?", help="baseline report (.json from evaluate.py / rerank.py)")
    p.add_argument("b_file", nargs="?", help="candidate report")
    p.add_argument("--collection", default=None)
    p.add_argument("--sets", nargs="+", default=None, help="eval sets to pool, e.g. dl19 dl20")
    p.add_argument("--a", default=None, help="baseline run tag (with --collection/--sets)")
    p.add_argument("--b", default=None, help="candidate run tag")
    p.add_argument("--metric", default=None, help="default: the eval set's primary metric")
    p.add_argument("--resamples", type=int, default=10_000)
    args = p.parse_args()

    if args.a_file and args.b_file:
        pairs = [(load(args.a_file), load(args.b_file))]
    elif args.collection and args.sets and args.a and args.b:
        root = get_settings().runs_dir / args.collection
        pairs = [(load(root / e / f"{safe_name(args.a)}.json"), load(root / e / f"{safe_name(args.b)}.json")) for e in args.sets]
    else:
        p.error("give two report files, or --collection --sets --a --b")

    metric = args.metric or pairs[0][0]["primary_metric"]
    a_scores, b_scores = {}, {}
    for ra, rb in pairs:
        if args.metric is None and ra["primary_metric"] != metric:
            p.error(f"{ra['eval_set']} uses {ra['primary_metric']}, not {metric}: pass --metric explicitly")
        es = ra["eval_set"]
        a_scores.update({f"{es}/{q}": v for q, v in ra["per_query"][metric].items()})
        b_scores.update({f"{es}/{q}": v for q, v in rb["per_query"][metric].items()})
        if len(pairs) > 1:
            single = paired_randomization_test(ra["per_query"][metric], rb["per_query"][metric], n_resamples=args.resamples)
            print(f"  {es:12s} {single.format()}")
    res = paired_randomization_test(a_scores, b_scores, n_resamples=args.resamples)
    ta, tb = pairs[0][0].get("tag", pairs[0][0]["retriever"]), pairs[0][1].get("tag", pairs[0][1]["retriever"])
    label = "pooled " + "+".join(r[0]["eval_set"] for r in pairs) if len(pairs) > 1 else pairs[0][0]["eval_set"]
    print(f"{metric} ({label}): A={ta}  B={tb}\n  {res.format()}")


if __name__ == "__main__":
    main()
