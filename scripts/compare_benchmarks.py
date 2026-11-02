"""Print all saved benchmark results (results/benchmarks/*.json) as one table per section.

python scripts/compare_benchmarks.py
python scripts/compare_benchmarks.py --markdown > results/benchmarks/README.md
"""

import argparse
import json

import _bootstrap  # noqa: F401

from searchengine.config import PROJECT_ROOT
from searchengine.envinfo import short_label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    files = sorted((PROJECT_ROOT / "results" / "benchmarks").glob("*.json"))
    if not files:
        raise SystemExit("no benchmark results yet: run scripts/benchmark.py")
    runs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            runs.append((f.stem, json.load(fh)))

    for section in ("first_stage", "rerank"):
        keys = sorted({k for _, r in runs for k in r.get(section, {})})
        if not keys:
            continue
        header = ["machine"] + keys
        rows = [[f"{short_label(r['environment'])} ({stem})"] + [
            f"{r[section][k]['p50_ms']:.1f}" if k in r.get(section, {}) else "-" for k in keys] for stem, r in runs]
        title = "First stage, p50 ms per query" if section == "first_stage" else "Reranking, p50 ms per query"
        if args.markdown:
            print(f"\n### {title}\n")
            print("| " + " | ".join(header) + " |")
            print("|" + "---|" * len(header))
            for row in rows:
                print("| " + " | ".join(row) + " |")
        else:
            print(f"\n{title}")
            widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
            for row in [header, *rows]:
                print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))


if __name__ == "__main__":
    main()
