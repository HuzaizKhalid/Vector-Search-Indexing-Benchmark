"""
Benchmark metrics: recall@k and latency statistics.
"""

import numpy as np
import time
from typing import Callable, List


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Fraction of true top-k neighbours that appear in the predicted top-k.

    predicted:    (n_queries, k_returned) integer array of returned IDs
    ground_truth: (n_queries, k_true)    integer array of true nearest IDs
    """
    n_queries = len(ground_truth)
    hits = 0
    for pred_row, gt_row in zip(predicted, ground_truth):
        # count overlap between returned set and true set, up to k
        hits += len(set(pred_row[:k]) & set(gt_row[:k]))
    return hits / (n_queries * k)


def measure_latencies(
    query_fn: Callable[[np.ndarray], np.ndarray],
    queries: np.ndarray,
    warmup_runs: int = 1,
    timing_runs: int = 3,
) -> dict:
    """
    Time `query_fn` on each query individually, return latency stats and QPS.

    query_fn receives a single query vector shaped (1, dim) and returns
    neighbour IDs (we discard the return value here — we only time the call).
    """
    # Warmup passes (caches, JIT, etc.)
    for _ in range(warmup_runs):
        for q in queries:
            query_fn(q[np.newaxis])

    all_times_ms: List[float] = []
    for _ in range(timing_runs):
        for q in queries:
            t0 = time.perf_counter()
            query_fn(q[np.newaxis])
            all_times_ms.append((time.perf_counter() - t0) * 1_000)

    times = np.array(all_times_ms)
    total_queries = len(queries) * timing_runs
    total_seconds = times.sum() / 1_000
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "qps": total_queries / total_seconds,
    }
