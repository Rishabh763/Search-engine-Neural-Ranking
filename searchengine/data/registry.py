"""Which datasets the project knows about, and how each one is officially scored.

Scoring conventions matter: the same run can look better or worse depending on
the relevance threshold and metric depth, so they are fixed here once rather
than chosen ad hoc per experiment.

- MS MARCO dev/small: ~1 judged relevant passage per query and sparse labels,
  so MRR@10 is the official metric. Precision@k is not meaningful there
  (unjudged passages are often relevant but count as misses).
- TREC DL 2019/2020: deep, graded judgments (0-3). nDCG@10 is official. For
  binary metrics (MRR, recall) on passages, relevance >= 2 counts as relevant.
- BEIR: nDCG@10 is the standard headline, recall@100 for first-stage quality.
"""

from __future__ import annotations

from dataclasses import dataclass

# corpus key -> ir_datasets id of the document collection
CORPORA: dict[str, str] = {
    "msmarco-passage": "msmarco-passage",
    "scifact": "beir/scifact",
    "fiqa": "beir/fiqa",
    "nfcorpus": "beir/nfcorpus",
}


@dataclass(frozen=True)
class EvalSetSpec:
    name: str
    ir_datasets_id: str
    corpus: str
    metrics: tuple[str, ...]
    primary_metric: str
    # minimum label counted as relevant by binary metrics (MRR, recall, precision)
    rel_threshold: int = 1


EVAL_SETS: dict[str, EvalSetSpec] = {
    s.name: s
    for s in [
        EvalSetSpec(
            "msmarco-dev",
            "msmarco-passage/dev/small",
            "msmarco-passage",
            ("mrr@10", "recall@100", "recall@1000", "ndcg@10"),
            "mrr@10",
        ),
        EvalSetSpec(
            "dl19",
            "msmarco-passage/trec-dl-2019/judged",
            "msmarco-passage",
            ("ndcg@10", "mrr@10", "recall@100", "recall@1000"),
            "ndcg@10",
            rel_threshold=2,
        ),
        EvalSetSpec(
            "dl20",
            "msmarco-passage/trec-dl-2020/judged",
            "msmarco-passage",
            ("ndcg@10", "mrr@10", "recall@100", "recall@1000"),
            "ndcg@10",
            rel_threshold=2,
        ),
        EvalSetSpec("scifact", "beir/scifact/test", "scifact", ("ndcg@10", "recall@100", "mrr@10"), "ndcg@10"),
        EvalSetSpec("fiqa", "beir/fiqa/test", "fiqa", ("ndcg@10", "recall@100", "mrr@10"), "ndcg@10"),
        EvalSetSpec("nfcorpus", "beir/nfcorpus/test", "nfcorpus", ("ndcg@10", "recall@100", "mrr@10"), "ndcg@10"),
    ]
}


def get_eval_set(name: str) -> EvalSetSpec:
    try:
        return EVAL_SETS[name]
    except KeyError:
        raise KeyError(f"Unknown eval set {name!r}. Known: {sorted(EVAL_SETS)}") from None
