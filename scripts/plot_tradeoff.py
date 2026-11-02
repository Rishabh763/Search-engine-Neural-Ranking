"""Plot ranking quality vs rerank latency for every depth sweep of a collection.

python scripts/plot_tradeoff.py --collection msmarco-100k --eval-set dl19
python scripts/plot_tradeoff.py --collection msmarco-100k --eval-set dl19 --out docs/tradeoff_dl19.png

One curve per (first stage, reranker variant, machine); one point per depth,
labelled with the depth; a dotted line per first stage for its quality without
reranking. Default output: runs/<collection>/sweeps/tradeoff_<eval_set>.png.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.envinfo import short_label

PRETTY_METRIC = {"ndcg@10": "nDCG@10", "mrr@10": "MRR@10", "recall@100": "Recall@100", "recall@1000": "Recall@1000"}
PRETTY_SET = {"dl19": "TREC DL 2019", "dl20": "TREC DL 2020", "msmarco-dev": "MS MARCO dev"}


def first_stage_label(tag: str) -> str:
    if tag.startswith("rrf("):
        return "Hybrid RRF"
    if tag.startswith("dense"):
        return "Dense"
    if tag.startswith("bm25"):
        return "BM25"
    return tag


def variant_label(info: dict) -> str:
    """'onnx-fp16-dml:ms-marco-MiniLM-L-6-v2' -> 'MiniLM-L6 ONNX fp16 (GPU, DirectML)'."""
    prefix, _, model = info.get("name", "").partition(":")
    model = model.replace("ms-marco-", "").replace("-v2", "").replace("MiniLM-L-", "MiniLM-L")
    runtimes = {
        "ce": "PyTorch fp32", "ce-int8": "PyTorch int8",
        "onnx-fp32": "ONNX fp32", "onnx-int8": "ONNX int8", "onnx-fp16": "ONNX fp16",
        "onnx-fp16-dml": "ONNX fp16 (GPU, DirectML)", "onnx-fp32-dml": "ONNX fp32 (GPU, DirectML)",
        "onnx-fp16-cuda": "ONNX fp16 (CUDA)",
    }
    runtime = runtimes.get(prefix, prefix)
    if prefix == "ce" and info.get("device") not in (None, "cpu"):
        runtime += f" ({info['device']})"
    return f"{model} {runtime}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--eval-set", default="dl19")
    p.add_argument("--metric", default=None, help="default: the eval set's primary metric")
    p.add_argument("--files", nargs="*", default=None, help="sweep JSONs (default: all for the collection)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = get_settings()
    sweep_dir = s.runs_dir / args.collection / "sweeps"
    files = [Path(f) for f in args.files] if args.files else sorted(sweep_dir.glob("*.json"))
    sweeps = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            sweeps.append(json.load(f))
    sweeps = [sw for sw in sweeps if sw.get("latency") and args.eval_set in sw.get("primary_metrics", {})]
    if not sweeps:
        raise SystemExit(f"no sweeps with latency for {args.eval_set} in {sweep_dir}")
    metric = args.metric or sweeps[0]["primary_metrics"][args.eval_set]
    same_machine = len({sw["environment"].get("cpu") for sw in sweeps}) == 1

    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    baselines: dict[str, float] = {}
    sweeps.sort(key=lambda sw: (first_stage_label(sw["first_stage"]), variant_label(sw["reranker"])))
    for i, sw in enumerate(sweeps):
        fs = first_stage_label(sw["first_stage"])
        q0 = sw["quality"].get("0", {}).get(args.eval_set)
        if q0:
            baselines[fs] = q0[metric]
        pts = []
        for depth, lat in sorted(sw["latency"].items(), key=lambda x: int(x[0])):
            q = sw["quality"].get(depth, {}).get(args.eval_set)
            if q is not None:
                pts.append((lat["p50_ms"], q[metric], int(depth)))
        if not pts:
            continue
        xs, ys, _ = zip(*pts)
        color = colors[i % len(colors)]
        label = f"{fs} → {variant_label(sw['reranker'])}"
        if not same_machine:
            label += f"  [{short_label(sw['environment'])}]"
        ax.plot(xs, ys, "--" if fs == "Hybrid RRF" else "-", marker="o" if fs == "BM25" else "s",
                color=color, label=label, lw=2, ms=6)
        for x, y, d in pts:
            ax.annotate(str(d), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color=color)
    for fs, y in sorted(baselines.items()):
        ax.axhline(y, linestyle=":", color="grey", lw=1)
        ax.annotate(f"{fs} only, no reranking: {y:.3f}", (0.99, y), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color="dimgrey")

    ax.set_xscale("log")
    ax.set_xlabel("rerank latency per query, p50 (ms, log scale); point labels = candidates reranked")
    ax.set_ylabel(f"{PRETTY_METRIC.get(metric, metric)} on {PRETTY_SET.get(args.eval_set, args.eval_set)}")
    title = f"Quality vs latency by reranking depth ({args.collection})"
    if same_machine:
        cpu = sweeps[0]["environment"].get("cpu", "")
        title += f"\n{cpu}" + (" + integrated GPU via DirectML" if any("dml" in sw["reranker"].get("name", "") for sw in sweeps) else "")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    out = Path(args.out) if args.out else sweep_dir / f"tradeoff_{args.eval_set}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
