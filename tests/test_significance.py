import numpy as np
import pytest

from searchengine.eval.significance import paired_randomization_test


def test_identical_systems_not_significant():
    rng = np.random.default_rng(0)
    a = {f"q{i}": float(v) for i, v in enumerate(rng.random(200))}
    res = paired_randomization_test(a, dict(a), n_resamples=2000, n_bootstrap=2000)
    assert res.diff == 0.0
    assert res.p_value == pytest.approx(1.0)


def test_consistent_improvement_is_significant_and_ci_covers_effect():
    rng = np.random.default_rng(1)
    base = rng.random(300)
    a = {f"q{i}": float(v) for i, v in enumerate(base)}
    b = {f"q{i}": float(v + 0.05 + rng.normal(0, 0.05)) for i, v in enumerate(base)}
    res = paired_randomization_test(a, b, n_resamples=2000, n_bootstrap=2000)
    assert res.p_value < 0.01
    assert res.ci_low < 0.05 < res.ci_high


def test_noise_only_usually_not_significant():
    rng = np.random.default_rng(2)
    base = rng.random(100)
    a = {f"q{i}": float(v) for i, v in enumerate(base)}
    b = {f"q{i}": float(v + rng.normal(0, 0.1)) for i, v in enumerate(base)}
    res = paired_randomization_test(a, b, n_resamples=2000, n_bootstrap=2000)
    assert res.p_value > 0.01


def test_only_common_queries_used():
    res = paired_randomization_test({"a": 1.0, "b": 0.0}, {"b": 1.0, "c": 0.0}, n_resamples=100, n_bootstrap=100)
    assert res.n == 1
