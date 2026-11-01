r"""Paired significance testing for comparing two systems on the same queries.

Why paired: query difficulty varies far more than the difference between two
systems, so we test the per-query differences d_q = B_q - A_q, not the two means
independently.

Paired randomization (sign-flip permutation) test
-------------------------------------------------
H0: systems A and B are exchangeable, i.e. for every query the label "A" vs "B"
could be swapped without changing the distribution. Under H0 each d_q is equally
likely to be +|d_q| or -|d_q|. So we draw random signs s_q in {-1, +1}, compute
mean(s * d), and count how often its magnitude reaches the observed |mean(d)|:

    p = (1 + #{ |mean(s * d)| >= |mean(d)| }) / (1 + n_resamples)

The +1 terms make the estimate valid (never exactly 0). No normality assumption,
which matters because per-query RR/nDCG values are bounded and very skewed.
Smucker et al. (CIKM 2007) recommend this test for IR evaluation.

A percentile bootstrap 95% CI on the mean difference is reported alongside, since
a p-value alone says nothing about effect size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedTestResult:
    n: int
    mean_a: float
    mean_b: float
    diff: float
    p_value: float
    ci_low: float
    ci_high: float

    def format(self, alpha: float = 0.05) -> str:
        sig = "significant" if self.p_value < alpha else "not significant"
        return (
            f"A={self.mean_a:.4f}  B={self.mean_b:.4f}  diff={self.diff:+.4f}  "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  p={self.p_value:.4g} ({sig} at {alpha})  n={self.n}"
        )


def paired_randomization_test(
    a: dict[str, float],
    b: dict[str, float],
    n_resamples: int = 10_000,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    chunk: int = 500,
) -> PairedTestResult:
    """Test B vs A on the queries both were scored on. Inputs: {qid: metric value}."""
    qids = sorted(set(a) & set(b))
    if not qids:
        raise ValueError("No common queries between the two systems")
    xa = np.array([a[q] for q in qids], dtype=np.float64)
    xb = np.array([b[q] for q in qids], dtype=np.float64)
    d = xb - xa
    n = d.size
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)

    # Chunked so memory stays O(chunk * n) even for ~7K queries x 10K resamples.
    extreme = 0
    for start in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - start)
        signs = rng.integers(0, 2, size=(m, n), dtype=np.int8) * 2 - 1
        perm_means = np.abs((signs * d).mean(axis=1))
        extreme += int(np.sum(perm_means >= obs - 1e-12))
    p = (extreme + 1) / (n_resamples + 1)

    boot = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, chunk):
        m = min(chunk, n_bootstrap - start)
        idx = rng.integers(0, n, size=(m, n))
        boot[start : start + m] = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return PairedTestResult(n, float(xa.mean()), float(xb.mean()), float(d.mean()), float(p), float(lo), float(hi))
