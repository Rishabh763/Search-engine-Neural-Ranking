import math
import random

import pytest

from searchengine.eval.metrics import evaluate, ndcg, parse_metric, rank_documents


def test_hand_computed_example():
    # ranking: a, b, c, d ; relevant: b(1), d(2), e(1, never retrieved)
    run = {"q": {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}}
    qrels = {"q": {"b": 1, "d": 2, "e": 1, "c": 0}}
    r = evaluate(run, qrels, ["mrr@10", "recall@2", "precision@4", "map@10", "ndcg@4"]).mean
    assert r["mrr@10"] == pytest.approx(1 / 2)
    assert r["recall@2"] == pytest.approx(1 / 3)
    assert r["precision@4"] == pytest.approx(2 / 4)
    assert r["map@10"] == pytest.approx((1 / 2 + 2 / 4) / 3)
    dcg = 1 / math.log2(3) + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert r["ndcg@4"] == pytest.approx(dcg / idcg)


def test_threshold_changes_binary_metrics_only():
    run = {"q": {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}}
    qrels = {"q": {"b": 1, "d": 2}}
    r1 = evaluate(run, qrels, ["mrr@10", "ndcg@10"], rel_threshold=1).mean
    r2 = evaluate(run, qrels, ["mrr@10", "ndcg@10"], rel_threshold=2).mean
    assert r1["mrr@10"] == pytest.approx(1 / 2)
    assert r2["mrr@10"] == pytest.approx(1 / 4)
    assert r1["ndcg@10"] == pytest.approx(r2["ndcg@10"])


def test_exponential_gain():
    assert ndcg(["x", "y"], {"x": 1, "y": 3}, 2, "exponential") == pytest.approx(
        (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    )


def test_missing_query_scores_zero_and_queries_without_relevant_are_skipped():
    run = {"q1": {"a": 1.0}}
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 0}}
    res = evaluate(run, qrels, ["mrr@10"])
    assert res.n_queries == 2  # q3 has no relevant doc
    assert res.n_missing_from_run == 1
    assert res.mean["mrr@10"] == pytest.approx(0.5)


def test_tie_break_is_docid_descending():
    assert rank_documents({"a": 1.0, "c": 1.0, "b": 2.0}) == ["b", "c", "a"]


def test_parse_metric():
    assert parse_metric("P@5") == ("precision", 5)
    with pytest.raises(ValueError):
        parse_metric("ndcg10")


def _random_case(seed):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(60)]
    qrels, run = {}, {}
    for q in range(40):
        qid = f"q{q}"
        judged = rng.sample(docs, rng.randint(1, 15))
        qrels[qid] = {d: rng.choice([0, 0, 1, 2, 3]) for d in judged}
        # rounded scores create ties, exercising the tie-break rule
        run[qid] = {d: round(rng.random(), 1) for d in rng.sample(docs, rng.randint(5, 50))}
    return run, qrels


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("threshold", [1, 2])
def test_matches_pytrec_eval(seed, threshold):
    pytrec_eval = pytest.importorskip("pytrec_eval")
    run, qrels = _random_case(seed)
    qrels = {q: r for q, r in qrels.items() if any(v >= threshold for v in r.values())}
    run = {q: run[q] for q in qrels}

    ours = evaluate(run, qrels, ["recall@20", "precision@5", "ndcg@10", "map@30"], rel_threshold=threshold).per_query
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"recall.20", "P.5", "ndcg_cut.10", "map_cut.30"}, relevance_level=threshold)
    ref = ev.evaluate(run)
    for q in qrels:
        assert ours["recall@20"][q] == pytest.approx(ref[q]["recall_20"], abs=1e-9)
        assert ours["precision@5"][q] == pytest.approx(ref[q]["P_5"], abs=1e-9)
        assert ours["ndcg@10"][q] == pytest.approx(ref[q]["ndcg_cut_10"], abs=1e-9)
        assert ours["map@30"][q] == pytest.approx(ref[q]["map_cut_30"], abs=1e-9)

    # trec_eval's recip_rank has no cutoff, so compare MRR@10 on runs truncated to 10.
    run10 = {q: {d: run[q][d] for d in rank_documents(run[q])[:10]} for q in run}
    ref_rr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}, relevance_level=threshold).evaluate(run10)
    ours_rr = evaluate(run, qrels, ["mrr@10"], rel_threshold=threshold).per_query["mrr@10"]
    for q in qrels:
        assert ours_rr[q] == pytest.approx(ref_rr[q]["recip_rank"], abs=1e-9)
