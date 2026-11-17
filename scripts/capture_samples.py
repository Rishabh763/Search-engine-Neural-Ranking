"""Record live API responses for the frontend's offline fallback.

The deployed frontend calls the search API on the laptop. When the laptop is off, the
page has nothing to talk to, so it shows these recorded responses instead, labelled as
recorded, with the date they were captured. Run with the server up:

    python scripts/capture_samples.py                      # -> Frontend/public/samples.json
    python scripts/capture_samples.py --candidates 100 --api http://127.0.0.1:8000

Each query is sent once after one warm-up request, so the stored timings are what a
visitor with a warm server would have seen. Standard library only.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    "how do vaccines train the immune system",
    "what causes the northern lights",
    "difference between a virus and bacteria",
    "how long to boil an egg",
    "why is the sky blue",
    "symptoms of vitamin d deficiency",
]


def get(url: str, timeout: float = 60.0) -> tuple[dict, float]:
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local API
        body = json.load(r)
    return body, (time.perf_counter() - t0) * 1000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--out", type=Path, default=ROOT / "Frontend" / "public" / "samples.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--queries", nargs="*", default=EXAMPLES)
    args = p.parse_args()

    api = args.api.rstrip("/")
    health, _ = get(f"{api}/health")
    get(f"{api}/search?" + urllib.parse.urlencode({"q": "warm up", "k": args.k, "candidates": args.candidates}))
    samples = []
    for q in args.queries:
        params = {"q": q, "k": args.k, "candidates": args.candidates, "rerank": "true", "dedup": "true"}
        body, ms = get(f"{api}/search?" + urllib.parse.urlencode(params))
        samples.append({"query": q, "params": {k: v for k, v in params.items() if k != "q"},
                        "client_ms": round(ms, 1), "response": body})
        t = body["timings"]
        print(f"{q!r}: {len(body['results'])} results, server {t['total_ms']:.0f} ms, round trip {ms:.0f} ms,"
              f" {body['collapsed']} copies collapsed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "health": health,
                   "samples": samples}, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
