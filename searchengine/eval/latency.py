"""Per-query latency measurement.

Batch throughput hides tail latency, and users feel the tail. So latency is
measured by issuing queries one at a time (after a warmup to fill caches and JIT
paths) and reporting percentiles, not the mean alone.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def measure_latency(search_fn: Callable[[str], object], queries: list[str], warmup: int = 10) -> dict[str, float]:
    if not queries:
        return {}
    for q in queries[:warmup]:
        search_fn(q)
    times = np.empty(len(queries))
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        search_fn(q)
        times[i] = (time.perf_counter() - t0) * 1000.0
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {
        "n": int(len(queries)),
        "mean_ms": float(times.mean()),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(times.max()),
    }
