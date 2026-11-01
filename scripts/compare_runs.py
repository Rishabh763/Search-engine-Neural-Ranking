"""Paired significance test between two evaluated systems on one metric.

python scripts/compare_runs.py runs/msmarco-100k/msmarco-dev/bm25-es.json \
                               runs/msmarco-100k/msmarco-dev/bm25-local.json --metric mrr@10
"""

import argparse
import json

import _bootstrap  # noqa: F401

from searchengine.eval.significance import paired_randomization_test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="baseline report (.json from evaluate.py)")
    p.add_argument("b", help="candidate report")
    p.add_argument("--metric", default=None, help="default: the eval set's primary metric")
    p.add_argument("--resamples", type=int, default=10_000)
    args = p.parse_args()

    with open(args.a, encoding="utf-8") as f:
        ra = json.load(f)
    with open(args.b, encoding="utf-8") as f:
        rb = json.load(f)
    metric = args.metric or ra["primary_metric"]
    res = paired_randomization_test(ra["per_query"][metric], rb["per_query"][metric], n_resamples=args.resamples)
    print(f"{metric}: A={ra.get('tag', ra['retriever'])}  B={rb.get('tag', rb['retriever'])}\n  {res.format()}")


if __name__ == "__main__":
    main()
