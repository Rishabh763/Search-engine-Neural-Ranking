"""How many top-k slots do near-duplicates waste, and what does collapsing do to the metrics?

python scripts/dedup_stats.py --collection msmarco-full --run "np1024-r500+ce@100" --eval-sets dl19 dl20
python scripts/dedup_stats.py --collection msmarco-full --run "dense-exact-msmarco-MiniLM-L6-cos-v5+ms-marco-MiniLM-L-6-v2@100" --eval-sets msmarco-dev

For each query the ranked run is collapsed exactly as the API does it
(searchengine.dedup.collapse, greedy in rank order) until k distinct passages are
kept. Reported per eval set:

    queries with >= 1 duplicate in the top k, and duplicate slots per query
    judged-relevant passages that collapsing removed from the top k, and how many
        of those were represented by a passage that is itself judged relevant
    distinct relevant passages in the top k, before and after: each group of
        copies counts once, and is relevant if any copy is judged relevant (the
        same grouping for both rankings), i.e. how much relevant *content* a user sees
    group-aware reciprocal rank@k: the first slot showing a judged passage or a copy
        of one; with sparse labels (MS MARCO dev: ~1 judged passage per query) the
        plain metric depends on which copy the annotator happened to judge
    the metrics before and after, from a collapsed run written next to the input
        (<run>+dedup.run / .json), so compare_runs.py can test the difference

Read the metric change with care: nDCG and MRR score document *ids*, not content.
If a passage and its copy are both judged relevant, the original ranking is paid
twice for showing the same text (the metrics do not penalise redundancy; alpha-nDCG
was designed for that), so collapsing can lower the score while improving what a
user sees. If the copy is unjudged, as on MS MARCO dev where one passage per query
is judged, collapsing frees a slot that a different, possibly relevant, passage
fills.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.data.registry import spec_or_default
from searchengine.dedup import DEFAULT_THRESHOLD, collapse
from searchengine.docstore import DocStore
from searchengine.envinfo import environment_info
from searchengine.eval.metrics import evaluate, rank_documents
from searchengine.eval.trec import read_run, safe_name, write_run
from searchengine.pipeline import passage_text


def collapse_ranked(ranked: list[str], docs: DocStore, k: int, depth: int, threshold: float):
    """Collapse the top of one ranked list until k are kept (or `depth` is exhausted).
    Fetches text in growing chunks, so a query without duplicates costs k reads."""
    n = min(len(ranked), max(2 * k, k + 5))
    while True:
        head = ranked[:n]
        texts = docs.get_documents(head)
        head = [d for d in head if d in texts]
        c = collapse([passage_text(texts[d]) for d in head], k=k, threshold=threshold)
        if len(c.kept) >= k or n >= min(depth, len(ranked)):
            return head, c
        n = min(depth, len(ranked), 2 * n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--run", required=True, help="tag of a ranked run (e.g. a reranked run from rerank.py)")
    p.add_argument("--eval-sets", nargs="*", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--depth", type=int, default=100, help="how far down to look for replacements")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--max-queries", type=int, default=None)
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    docs = DocStore.open(coll, s.index_dir)
    src = safe_name(args.run)
    # A non-default threshold gets its own tag, so a sweep never overwrites the default run.
    suffix = "" if args.threshold == DEFAULT_THRESHOLD else f"{args.threshold:g}"
    tag = safe_name(f"{src}+dedup{suffix}")

    for es_name in args.eval_sets or coll.eval_set_names:
        spec = spec_or_default(es_name)
        run_dir = s.runs_dir / coll.name / es_name
        path = run_dir / f"{src}.run"
        if not path.exists():
            print(f"[{es_name}] {path} not found, skipped")
            continue
        queries, qrels = coll.queries(es_name), coll.qrels(es_name)
        qids = sorted(q for q in queries if q in qrels)
        if args.max_queries:
            qids = qids[: args.max_queries]
        run = read_run(path)
        t = spec.rel_threshold

        out, dup_slots, rel_dropped, rel_dropped_rep_rel, examples = {}, [], 0, 0, []
        distinct_before, distinct_after, grr_before, grr_after = [], [], [], []
        for q in qids:
            ranked = rank_documents(run.get(q, {}))
            head, c = collapse_ranked(ranked, docs, args.k, args.depth, args.threshold)
            kept = [head[i] for i in c.kept]
            dropped_top = [(rep, i) for rep, idx in c.duplicates.items() for i in idx if i < args.k]
            dup_slots.append(len(dropped_top))
            judged = qrels.get(q, {})
            for rep, i in dropped_top:
                if judged.get(head[i], 0) >= t:
                    rel_dropped += 1
                    rel_dropped_rep_rel += judged.get(head[rep], 0) >= t
            # Distinct relevant content: group = the kept passage a copy collapsed into.
            group = {i: i for i in c.kept}
            for rep, idx in c.duplicates.items():
                group.update({i: rep for i in idx})
            grade: dict[int, int] = {}
            for i, g in group.items():
                grade[g] = max(grade.get(g, 0), judged.get(head[i], 0))
            distinct_before.append(len({group[i] for i in range(min(args.k, c.examined)) if grade[group[i]] >= t}))
            distinct_after.append(sum(1 for g in c.kept if grade[g] >= t))
            # Group-aware reciprocal rank: the first slot whose passage OR a copy of it is judged relevant.
            top = [group[i] for i in range(min(args.k, c.examined))]
            grr_before.append(next((1 / (r + 1) for r, g in enumerate(top) if grade[g] >= t), 0.0))
            grr_after.append(next((1 / (r + 1) for r, g in enumerate(c.kept[: args.k]) if grade[g] >= t), 0.0))
            if dropped_top and len(examples) < 3:
                examples.append((q, [(head[rep], head[i]) for rep, i in dropped_top[:3]]))
            # Collapsed ranking: the kept passages, then everything else in its old order.
            kept_set = set(kept)
            rest = [d for d in ranked if d not in kept_set]
            order = kept + rest
            out[q] = {d: float(len(order) - r) for r, d in enumerate(order)}

        base = evaluate({q: run.get(q, {}) for q in qids}, {q: qrels[q] for q in qids}, spec.metrics, rel_threshold=t)
        res = evaluate(out, {q: qrels[q] for q in qids}, spec.metrics, rel_threshold=t)
        a = np.asarray(dup_slots)
        stats = {
            "queries": len(qids),
            "k": args.k,
            "threshold": args.threshold,
            "queries_with_duplicates": int((a > 0).sum()),
            "share_queries_with_duplicates": round(float((a > 0).mean()), 4),
            "duplicate_slots_per_query": round(float(a.mean()), 4),
            "share_of_topk_slots": round(float(a.sum() / (len(a) * args.k)), 4),
            "max_duplicate_slots": int(a.max()) if len(a) else 0,
            "judged_relevant_dropped": rel_dropped,
            "judged_relevant_dropped_with_relevant_representative": rel_dropped_rep_rel,
            "distinct_relevant_in_topk_before": round(float(np.mean(distinct_before)), 4),
            "distinct_relevant_in_topk_after": round(float(np.mean(distinct_after)), 4),
            "group_aware_rr_before": round(float(np.mean(grr_before)), 4),
            "group_aware_rr_after": round(float(np.mean(grr_after)), 4),
        }
        write_run(out, run_dir / f"{tag}.run", tag=tag)
        with open(run_dir / f"{tag}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"collection": coll.name, "eval_set": es_name, "tag": tag, "retriever": tag, "source_run": src,
                       "rel_threshold": t, "primary_metric": spec.primary_metric, "n_queries": res.n_queries,
                       "metrics": res.mean, "source_metrics": base.mean, "dedup": stats,
                       "environment": environment_info(), "per_query": res.per_query}, f, indent=2)

        pm = spec.primary_metric
        print(f"[{coll.name}/{es_name}] {src}  (k={args.k}, containment >= {args.threshold})")
        print(f"  queries with a duplicate in the top {args.k}: {stats['queries_with_duplicates']}/{len(qids)}"
              f" ({100 * stats['share_queries_with_duplicates']:.1f}%);  duplicate slots per query"
              f" {stats['duplicate_slots_per_query']:.2f} ({100 * stats['share_of_topk_slots']:.1f}% of all slots);"
              f"  worst query {stats['max_duplicate_slots']}")
        print(f"  judged-relevant copies removed from the top {args.k}: {rel_dropped}"
              f" (represented by a judged-relevant passage: {rel_dropped_rep_rel})")
        db, da = np.asarray(distinct_before), np.asarray(distinct_after)
        print(f"  distinct relevant passages in the top {args.k}: {db.mean():.2f} -> {da.mean():.2f}"
              f" ({da.mean() - db.mean():+.2f} per query; more in {int((da > db).sum())} queries,"
              f" fewer in {int((da < db).sum())})")
        print(f"  group-aware RR@{args.k} (a copy of a judged passage counts): {np.mean(grr_before):.4f} ->"
              f" {np.mean(grr_after):.4f}")
        print(f"  {pm} (scores ids, not content): {base.mean[pm]:.4f} -> {res.mean[pm]:.4f} ({res.mean[pm] - base.mean[pm]:+.4f})"
              f"   -> compare_runs.py --a \"{src}\" --b \"{tag}\"")
        for q, pairs in examples:
            print(f"  e.g. query {q}: " + ", ".join(f"{d} duplicates {r}" for r, d in pairs))


if __name__ == "__main__":
    main()
