"""Export MS MARCO training queries + positives into a prepared collection.

python scripts/prepare_train.py --collection msmarco-full --n-queries 200000

Writes data/processed/<collection>/train/{queries.tsv, qrels.txt, meta.json}. The
training queries come from MS MARCO's train split, disjoint from every eval set.
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401

from searchengine.config import PROJECT_ROOT, get_settings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--n-queries", type=int, default=200_000, help="0 = all ~500K judged training queries")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    os.environ.setdefault("IR_DATASETS_HOME", str(PROJECT_ROOT / "data" / "raw" / "ir_datasets"))
    from searchengine.data.build import export_train_split

    root = get_settings().data_dir / args.collection
    if not (root / "manifest.json").exists():
        raise SystemExit(f"{root} is not a prepared collection: run scripts/prepare_data.py first")
    meta = export_train_split(root, args.n_queries or None, args.seed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
