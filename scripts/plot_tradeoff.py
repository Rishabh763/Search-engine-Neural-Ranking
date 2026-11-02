"""Plot ranking quality vs rerank latency for every depth sweep of a collection.

python scripts/plot_tradeoff.py --collection msmarco-100k --eval-set dl19

One curve per (reranker variant, machine), one point per depth, BM25 as a dashed
baseline. Written to runs/<collection>/sweeps/tradeoff_<eval_set>.png.
"""

import argparse
import json

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.envinfo import short_label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--eval-set", default="dl19")
    p.add_argument("--metric", default=None, help="default: the eval set's primary metric")
    p.add_argument("--files", nargs="*", default=None, help="sweep JSONs (default: all for the collection)")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = get_settings()
    sweep_dir = s.runs_dir / args.collection / "sweeps"
    files = args.files or sorted(sweep_dir.glob("*.json"))
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline = None
    for path in files:
        with open(path, encoding="utf-8") as f:
            sw = json.load(f)
        metric = args.metric or sw["primary_metrics"].get(args.eval_set)
        if not metric or not sw["latency"]:
            continue
        pts = []
        for depth, lat in sorted(sw["latency"].items(), key=lambda x: int(x[0])):
            q = sw["quality"].get(depth, {}).get(args.eval_set)
            if q is not None:
                pts.append((lat["p50_ms"], q[metric], int(depth)))
        if not pts:
            continue
        if baseline is None and "0" in sw["quality"] and args.eval_set in sw["quality"]["0"]:
            baseline = sw["quality"]["0"][args.eval_set][metric]
        xs, ys, ds = zip(*pts)
        label = f"{sw['reranker']['name']}  [{short_label(sw['environment'])}]"
        ax.plot(xs, ys, marker="o", label=label)
        for x, y, d in pts:
            ax.annotate(str(d), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=8)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="grey", label=f"BM25 only ({baseline:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("rerank latency per query, p50 (ms, log scale)")
    ax.set_ylabel(f"{args.metric or 'primary metric'} on {args.eval_set}")
    ax.set_title(f"Quality vs latency by reranking depth ({args.collection})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    out = sweep_dir / f"tradeoff_{args.eval_set}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
