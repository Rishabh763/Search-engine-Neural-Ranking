"""Split Elasticsearch query latency into server time and transport overhead.

Every ES search response carries `took`: milliseconds spent inside Elasticsearch.
Client wall time minus `took` is everything else: HTTP, JSON (de)serialization,
the Windows -> Docker/WSL2 network hop, and the Python client. Measuring both
shows where the time actually goes before optimizing anything.

python scripts/es_latency_probe.py --collection msmarco-100k
python scripts/es_latency_probe.py --collection msmarco-100k --urls http://localhost:9200 http://127.0.0.1:9200
"""

import argparse
import time

import numpy as np

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.index.es import SEARCH_FIELD, make_client


def pct(xs):
    a = np.asarray(xs)
    return f"p50={np.percentile(a, 50):6.1f}  p95={np.percentile(a, 95):6.1f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--index-name", default=None)
    p.add_argument("--eval-set", default="msmarco-dev")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--urls", nargs="+", default=None)
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    index = args.index_name or coll.name
    queries = list(coll.queries(args.eval_set).values())[: args.n]

    for url in args.urls or [s.es_url]:
        es = make_client(url)
        print(f"\n{url}   ({len(queries)} queries, all times in ms)")

        rt = []
        for _ in range(50):
            t0 = time.perf_counter()
            es.info()
            rt.append((time.perf_counter() - t0) * 1000)
        print(f"  GET / (no search work)   client {pct(rt[5:])}")

        for k in (10, 100, 1000):
            wall, took = [], []
            for i, q in enumerate(queries):
                t0 = time.perf_counter()
                r = es.search(index=index, query={"match": {SEARCH_FIELD: q}}, size=k, source=False, track_total_hits=False)
                w = (time.perf_counter() - t0) * 1000
                if i >= 10:  # warmup
                    wall.append(w)
                    took.append(r["took"])
            overhead = np.asarray(wall) - np.asarray(took)
            print(f"  search k={k:<5d}  client {pct(wall)} | server(took) {pct(took)} | overhead {pct(overhead)}")


if __name__ == "__main__":
    main()
