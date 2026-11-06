"""Encode a collection's passages with a bi-encoder into resumable float16 shards.

python scripts/encode_corpus.py --collection msmarco-100k
python scripts/encode_corpus.py --collection msmarco-full --batch-size 256      # on a GPU machine

Interrupt any time and re-run the same command: finished shards are skipped.
That matters for the full corpus (8.8M passages), which on a free cloud GPU takes
hours and may be cut off. Throughput (passages/s) and the machine fingerprint are
recorded in meta.json, so encoding speed is comparable across machines.
"""

import argparse

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.dense.build import encode_collection
from searchengine.dense.encoder import BiEncoder, default_dense_model, model_slug
from searchengine.dense.index import dense_dir
from searchengine.envinfo import environment_info, short_label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--model", default=None, help="default: DENSE_MODEL env or msmarco-MiniLM-L6-cos-v5")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--shard-size", type=int, default=100_000)
    args = p.parse_args()
    args.model = args.model or default_dense_model()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    out = dense_dir(s.index_dir, coll.name, model_slug(args.model))
    enc = BiEncoder(args.model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    env = environment_info()
    print(f"encoder {enc.config()}\nmachine: {short_label(env)}  device={enc.device} fp16={enc.fp16}")
    meta = encode_collection(coll, enc, out, args.shard_size, environment=env)
    n = sum(sh["n"] for sh in meta["shards"])
    print(f"done: {n} passages, {meta['docs_per_second']} passages/s -> {out}")


if __name__ == "__main__":
    main()
