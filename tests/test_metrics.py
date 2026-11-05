"""
Unit tests for utils/metrics.py.

Run with:
    python -m pytest tests/ -v
"""

import numpy as np
import pytest

from utils.metrics import recall_at_k, measure_latencies


# ─── recall_at_k ──────────────────────────────────────────────────────────────

def test_perfect_recall():
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    pred = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    assert recall_at_k(pred, gt, k=10) == pytest.approx(1.0)


def test_zero_recall():
    gt = np.array([[0, 1, 2]])
    pred = np.array([[10, 11, 12]])
    assert recall_at_k(pred, gt, k=3) == pytest.approx(0.0)


def test_half_recall():
    gt = np.array([[0, 1, 2, 3]])
    pred = np.array([[0, 1, 10, 11]])   # 2 out of 4 correct
    assert recall_at_k(pred, gt, k=4) == pytest.approx(0.5)


def test_multiple_queries():
    gt = np.array([[0, 1], [2, 3]])
    pred = np.array([[0, 1], [2, 99]])  # first perfect, second 1/2
    assert recall_at_k(pred, gt, k=2) == pytest.approx(0.75)


def test_recall_truncates_to_k():
    # Even if more results are returned we only look at top-k
    gt = np.array([[0, 1, 2]])
    pred = np.array([[99, 0, 1]])   # first result wrong, hits at positions 2&3
    # For k=3: set(pred[:3])={99,0,1}  &  set(gt[:3])={0,1,2}  → 2 hits / 3 = 0.667
    assert recall_at_k(pred, gt, k=3) == pytest.approx(2 / 3)


# ─── measure_latencies ────────────────────────────────────────────────────────

def test_measure_latencies_returns_expected_keys():
    queries = np.random.rand(20, 8).astype(np.float32)
    stats = measure_latencies(lambda q: q, queries, warmup_runs=1, timing_runs=2)
    assert set(stats.keys()) == {"p50_ms", "p95_ms", "mean_ms", "qps"}


def test_measure_latencies_qps_positive():
    queries = np.random.rand(10, 4).astype(np.float32)
    stats = measure_latencies(lambda q: q, queries, warmup_runs=0, timing_runs=1)
    assert stats["qps"] > 0


def test_measure_latencies_percentile_ordering():
    queries = np.random.rand(50, 4).astype(np.float32)
    stats = measure_latencies(lambda q: q, queries, warmup_runs=0, timing_runs=1)
    assert stats["p50_ms"] <= stats["p95_ms"]
