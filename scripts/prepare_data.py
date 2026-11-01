"""Download a corpus + eval sets via ir_datasets and write a collection directory.

Examples
--------
# 100K-passage MS MARCO subset with 1000 dev queries + all TREC DL 2019/2020 queries
python scripts/prepare_data.py --name msmarco-100k --corpus msmarco-passage \
    --eval-sets msmarco-dev dl19 dl20 --n-docs 100000 --n-queries 1000

# Full MS MARCO (8.8M passages, all 6980 dev/small queries)
python scripts/prepare_data.py --name msmarco-full --corpus msmarco-passage --eval-sets msmarco-dev dl19 dl20

# A BEIR dataset (small, always full)
python scripts/prepare_data.py --name scifact --corpus scifact --eval-sets scifact

# Offline synthetic collection (no download), for checking the install
python scripts/prepare_data.py --name toy --toy
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401

from searchengine.config import PROJECT_ROOT, get_settings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="collection name (directory under DATA_DIR)")
    p.add_argument("--corpus", default="msmarco-passage")
    p.add_argument("--eval-sets", nargs="+", default=["msmarco-dev"])
    p.add_argument("--n-docs", type=int, default=None, help="subset size; omit for the full corpus")
    p.add_argument("--n-queries", type=int, default=None, help="cap on queries per eval set")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--toy", action="store_true", help="write a synthetic collection instead (no download)")
    args = p.parse_args()

    out = get_settings().data_dir / args.name
    if args.toy:
        from searchengine.data.collection import write_collection
        from searchengine.data.toy import make_toy_collection

        docs, queries, qrels = make_toy_collection()
        m = write_collection(out, docs, {"toy": (queries, qrels)}, {"corpus": "toy", "subset": False})
    else:
        # Keep ir_datasets downloads inside the project unless the user set it.
        os.environ.setdefault("IR_DATASETS_HOME", str(PROJECT_ROOT / "data" / "raw" / "ir_datasets"))
        from searchengine.data.build import build

        m = build(out, args.corpus, args.eval_sets, args.n_docs, args.n_queries, args.seed)
    print(json.dumps(m, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
