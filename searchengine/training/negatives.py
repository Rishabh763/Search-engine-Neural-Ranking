r"""Hard-negative mining from a first-stage retriever.

A reranker learns the boundary between relevant passages and the passages the
first stage *actually* puts in front of it. Random negatives are trivially easy
(the model already scores them ~0, so the gradient sigma(s) - y ~ 0 teaches nothing);
negatives retrieved by our own retriever sit near the decision boundary.

The catch on MS MARCO is false negatives: labels are sparse (~1 judged positive per
query), so many top-ranked "negatives" are actually relevant but unjudged. Training
the model to push those down teaches it the wrong thing. Two standard mitigations:
  - skip the very top of the ranking (`skip_top`): the highest-ranked non-positives
    are the most likely false negatives;
  - sample from a window (here ranks skip_top..depth) instead of always the hardest.
Teacher-score denoising (drop negatives a strong reranker scores as relevant) is a
further option, not used by default.

Output: one JSON line per query: {"qid", "pos": [doc ids], "neg": [doc ids]}.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from searchengine.eval.metrics import rank_documents
from searchengine.types import Qrels, Queries


def select_negatives(ranked: list[str], positives: set[str], skip_top: int, n_neg: int, rng: random.Random) -> list[str]:
    """From a ranking, drop positives, skip the top `skip_top`, sample `n_neg` of the rest."""
    pool = [d for d in ranked if d not in positives][skip_top:]
    if len(pool) <= n_neg:
        return pool
    return rng.sample(pool, n_neg)


def mine_negatives(
    retriever,
    queries: Queries,
    qrels: Qrels,
    out_path: Path,
    depth: int = 100,
    skip_top: int = 3,
    n_neg: int = 15,
    seed: int = 13,
    chunk: int = 2048,
    log: Callable[[str], None] = print,
) -> dict:
    """Retrieve top-`depth` for every training query (in chunks, streaming to JSONL)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qids = sorted(q for q in queries if q in qrels)
    stats = {"queries": 0, "positive_in_topk": 0, "negatives": 0, "depth": depth, "skip_top": skip_top, "n_neg": n_neg}
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(qids), chunk):
            part = {q: queries[q] for q in qids[start : start + chunk]}
            run = retriever.batch_search(part, k=depth)
            for q in part:
                pos = {d for d, r in qrels[q].items() if r > 0}
                ranked = rank_documents(run.get(q, {}))
                rng = random.Random(f"{seed}:{q}")  # per-query seed: order-independent, reproducible
                neg = select_negatives(ranked, pos, skip_top, n_neg, rng)
                f.write(json.dumps({"qid": q, "pos": sorted(pos), "neg": neg}) + "\n")
                stats["queries"] += 1
                stats["negatives"] += len(neg)
                stats["positive_in_topk"] += int(any(d in pos for d in ranked))
            log(f"  mined {min(start + chunk, len(qids))}/{len(qids)} queries")
    stats["positive_recall_at_depth"] = round(stats["positive_in_topk"] / max(stats["queries"], 1), 4)
    stats["mean_negatives"] = round(stats["negatives"] / max(stats["queries"], 1), 2)
    with open(out_path.with_suffix(".stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=2)
    return stats


def read_negatives(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
