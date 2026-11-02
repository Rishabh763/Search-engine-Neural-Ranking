"""Load test the running search API: latency percentiles and throughput vs concurrency.

Start the API first (in another terminal), then:
    python scripts/load_test.py --collection msmarco-100k
    python scripts/load_test.py --collection msmarco-100k --concurrency 1 2 4 8 --requests 100 --candidates 20

Sends real benchmark queries over HTTP, `--concurrency` requests in flight at a
time, and reports per level:
  - client latency p50 / p95 / p99: what a user waits, including queueing
  - throughput (requests/s)
  - server-side stage times (retrieve / fetch / rerank), averaged from the
    response's `timings`, so you can see *where* time goes as load rises

What to expect on a single machine: with one GPU or a few cores, concurrent
requests mostly queue for the same compute. Latency then grows roughly linearly
with concurrency while throughput plateaus; the plateau is the system's capacity.
That knee, not the single-request latency, is what sizes a deployment.

Writes results/loadtests/<hostname>_<timestamp>.json with the server config
(from /health) and the machine fingerprint.
"""

import argparse
import asyncio
import json
import random
import time
from datetime import datetime

import numpy as np

import _bootstrap  # noqa: F401

from searchengine.config import PROJECT_ROOT, get_settings
from searchengine.data.collection import Collection
from searchengine.envinfo import environment_info, short_label


async def run_level(client, url: str, queries: list[str], concurrency: int, params: dict) -> dict:
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    stages: dict[str, list[float]] = {"retrieve_ms": [], "fetch_ms": [], "rerank_ms": [], "total_ms": []}
    errors = 0

    async def one(q: str) -> None:
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.get(f"{url}/search", params={**params, "q": q})
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    errors += 1
                    return
                lat.append(ms)
                for key, v in r.json()["timings"].items():
                    stages.setdefault(key, []).append(v)
            except Exception:
                errors += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one(q) for q in queries))
    wall = time.perf_counter() - t0
    a = np.asarray(lat) if lat else np.asarray([np.nan])
    return {
        "concurrency": concurrency,
        "requests": len(queries),
        "errors": errors,
        "throughput_rps": round(len(lat) / wall, 2),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "p99_ms": round(float(np.percentile(a, 99)), 1),
        "server_mean_ms": {k: round(float(np.mean(v)), 1) for k, v in stages.items() if v},
    }


async def main_async(args) -> None:
    import httpx

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    sets = args.eval_sets or coll.eval_set_names
    pool = [q for e in sets for q in coll.queries(e).values()]
    rng = random.Random(0)
    params = {"k": args.k}
    if args.candidates:
        params["candidates"] = args.candidates

    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=max(args.concurrency) + 4)) as client:
        try:
            health = (await client.get(f"{args.url}/health")).json()
        except Exception as e:
            raise SystemExit(f"API not reachable at {args.url} ({e}). Start it with: uvicorn searchengine.api:app --port 8000")
        reranker = (health.get("reranker") or {}).get("name", "none")
        env = environment_info()
        print(f"server: retriever={health.get('retriever')}  reranker={reranker}  params={params}")
        print(f"machine: {short_label(env)}\n")

        # Warm up: model kernels, caches, GPU clocks (laptop GPUs idle down between requests).
        for q in rng.sample(pool, min(args.warmup, len(pool))):
            await client.get(f"{args.url}/search", params={**params, "q": q})

        header = f"{'conc':>4} {'req/s':>7} {'p50':>8} {'p95':>8} {'p99':>8}   server mean: retrieve / fetch / rerank / total (ms)  errors"
        print(header)
        levels = []
        for c in args.concurrency:
            qs = [rng.choice(pool) for _ in range(args.requests)]
            res = await run_level(client, args.url, qs, c, params)
            levels.append(res)
            sm = res["server_mean_ms"]
            print(
                f"{c:>4} {res['throughput_rps']:>7.2f} {res['p50_ms']:>8.1f} {res['p95_ms']:>8.1f} {res['p99_ms']:>8.1f}"
                f"   {sm.get('retrieve_ms', 0):>8.1f} / {sm.get('fetch_ms', 0):.1f} / {sm.get('rerank_ms', 0):.1f} / {sm.get('total_ms', 0):.1f}"
                f"   {res['errors']}"
            )

    out_dir = PROJECT_ROOT / "results" / "loadtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{(env.get('hostname') or 'host').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"url": args.url, "server": health, "params": params, "levels": levels, "environment": env}, f, indent=2)
    print(f"\nwrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--collection", default="msmarco-100k", help="source of realistic queries")
    p.add_argument("--eval-sets", nargs="*", default=None)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--requests", type=int, default=60, help="requests per concurrency level")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=None, help="default: the server's SEARCH_CANDIDATES")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
