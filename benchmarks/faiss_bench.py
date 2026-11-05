"""
FAISS benchmark: Flat (exact), HNSW, IVF-Flat, IVF-PQ.

Each public function returns a list of dicts compatible with the shared
results CSV schema used by run_benchmark.py.
"""

import logging
import time
from typing import Any

import faiss
import numpy as np

from utils.memory_tracker import MemoryTracker
from utils.metrics import measure_latencies, recall_at_k

logger = logging.getLogger(__name__)

# Use one thread for single-query latency measurements so timings are
# comparable across systems that don't support multi-threading.
faiss.omp_set_num_threads(1)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _search_fn(index: Any, k: int, nprobe: int | None = None, ef_search: int | None = None):
    """Return a closure suitable for measure_latencies."""
    if nprobe is not None:
        index.nprobe = nprobe
    if ef_search is not None:
        index.hnsw.efSearch = ef_search

    def _fn(q: np.ndarray) -> np.ndarray:
        _, ids = index.search(q, k)
        return ids
    return _fn


def _run_queries(index: Any, queries: np.ndarray, k: int,
                 nprobe: int | None = None, ef_search: int | None = None,
                 warmup: int = 1, runs: int = 3) -> dict:
    fn = _search_fn(index, k, nprobe=nprobe, ef_search=ef_search)
    return measure_latencies(fn, queries, warmup_runs=warmup, timing_runs=runs)


def _get_all_neighbours(index: Any, queries: np.ndarray, k: int) -> np.ndarray:
    _, ids = index.search(queries, k)
    return ids


# ─── Flat (brute-force exact) ─────────────────────────────────────────────────

def bench_flat(train: np.ndarray, queries: np.ndarray,
               ground_truth: np.ndarray, cfg: dict) -> list[dict]:
    dim = train.shape[1]
    k = cfg["benchmark"]["k"]
    bm = cfg["benchmark"]

    logger.info("[FAISS Flat] building …")
    with MemoryTracker() as mem:
        t0 = time.perf_counter()
        index = faiss.IndexFlatL2(dim)
        index.add(train)
        build_time = time.perf_counter() - t0
        mem.snapshot()

    predicted = _get_all_neighbours(index, queries, k)
    rc = recall_at_k(predicted, ground_truth, k)
    lat = _run_queries(index, queries, k,
                       warmup=bm["warmup_runs"], runs=bm["n_timing_runs"])

    return [dict(
        system="FAISS", index_type="Flat", params="{}",
        recall_at_k=round(rc, 4),
        qps=round(lat["qps"], 2),
        p50_ms=round(lat["p50_ms"], 3),
        p95_ms=round(lat["p95_ms"], 3),
        build_time_s=round(build_time, 2),
        peak_memory_mb=round(mem.peak_increase_mb, 1),
    )]


# ─── HNSW ─────────────────────────────────────────────────────────────────────

def bench_hnsw(train: np.ndarray, queries: np.ndarray,
               ground_truth: np.ndarray, cfg: dict) -> list[dict]:
    """
    HNSW works by building a multi-layer proximity graph.  M controls the
    number of bi-directional links per node (more = better recall, more memory,
    slower build).  efSearch controls the beam-search width at query time
    (larger = higher recall, slower).
    """
    dim = train.shape[1]
    k = cfg["benchmark"]["k"]
    bm = cfg["benchmark"]
    hnsw_cfg = cfg["faiss"]["hnsw"]
    ef_construction = hnsw_cfg["ef_construction"]
    results = []

    for M in hnsw_cfg["M_values"]:
        logger.info("[FAISS HNSW] M=%d, ef_construction=%d — building …", M, ef_construction)
        with MemoryTracker() as mem:
            t0 = time.perf_counter()
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = ef_construction
            index.add(train)
            build_time = time.perf_counter() - t0
            mem.snapshot()

        for ef in hnsw_cfg["ef_search_values"]:
            index.hnsw.efSearch = ef
            predicted = _get_all_neighbours(index, queries, k)
            rc = recall_at_k(predicted, ground_truth, k)
            lat = _run_queries(index, queries, k, ef_search=ef,
                               warmup=bm["warmup_runs"], runs=bm["n_timing_runs"])
            params = f"M={M},efSearch={ef},efConstruct={ef_construction}"
            logger.info("  ef_search=%d → recall=%.3f  p50=%.2f ms", ef, rc, lat["p50_ms"])
            results.append(dict(
                system="FAISS", index_type="HNSW", params=params,
                recall_at_k=round(rc, 4),
                qps=round(lat["qps"], 2),
                p50_ms=round(lat["p50_ms"], 3),
                p95_ms=round(lat["p95_ms"], 3),
                build_time_s=round(build_time, 2),
                peak_memory_mb=round(mem.peak_increase_mb, 1),
            ))

    return results


# ─── IVF-Flat ─────────────────────────────────────────────────────────────────

def bench_ivf_flat(train: np.ndarray, queries: np.ndarray,
                   ground_truth: np.ndarray, cfg: dict) -> list[dict]:
    """
    IVF partitions the space into nlist Voronoi cells via k-means.  At query
    time only nprobe cells are searched.  Higher nprobe → better recall,
    higher latency.
    """
    dim = train.shape[1]
    k = cfg["benchmark"]["k"]
    bm = cfg["benchmark"]
    ivf_cfg = cfg["faiss"]["ivf_flat"]
    results = []

    for nlist in ivf_cfg["nlist_values"]:
        logger.info("[FAISS IVF-Flat] nlist=%d — training + building …", nlist)
        with MemoryTracker() as mem:
            t0 = time.perf_counter()
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(train)
            index.add(train)
            build_time = time.perf_counter() - t0
            mem.snapshot()

        for nprobe in ivf_cfg["nprobe_values"]:
            index.nprobe = nprobe
            predicted = _get_all_neighbours(index, queries, k)
            rc = recall_at_k(predicted, ground_truth, k)
            lat = _run_queries(index, queries, k, nprobe=nprobe,
                               warmup=bm["warmup_runs"], runs=bm["n_timing_runs"])
            params = f"nlist={nlist},nprobe={nprobe}"
            logger.info("  nprobe=%d → recall=%.3f  p50=%.2f ms", nprobe, rc, lat["p50_ms"])
            results.append(dict(
                system="FAISS", index_type="IVF-Flat", params=params,
                recall_at_k=round(rc, 4),
                qps=round(lat["qps"], 2),
                p50_ms=round(lat["p50_ms"], 3),
                p95_ms=round(lat["p95_ms"], 3),
                build_time_s=round(build_time, 2),
                peak_memory_mb=round(mem.peak_increase_mb, 1),
            ))

    return results


# ─── IVF-PQ ───────────────────────────────────────────────────────────────────

def bench_ivf_pq(train: np.ndarray, queries: np.ndarray,
                 ground_truth: np.ndarray, cfg: dict) -> list[dict]:
    """
    IVF-PQ compresses each vector into M_pq × nbits product-quantizer codes,
    giving ~8× smaller storage than IVF-Flat at the cost of approximation error.
    """
    dim = train.shape[1]
    k = cfg["benchmark"]["k"]
    bm = cfg["benchmark"]
    pq_cfg = cfg["faiss"]["ivf_pq"]
    nlist = pq_cfg["nlist"]
    nbits = pq_cfg["nbits"]

    # Choose M_pq such that dim is divisible and sub-vectors are ≥ 4-dim
    M_pq = dim // 8
    while dim % M_pq != 0 and M_pq > 1:
        M_pq -= 1
    logger.info("[FAISS IVF-PQ] dim=%d → M_pq=%d (sub-vectors of %d dims)",
                dim, M_pq, dim // M_pq)

    results = []
    logger.info("[FAISS IVF-PQ] nlist=%d, M_pq=%d, nbits=%d — training + building …",
                nlist, M_pq, nbits)
    with MemoryTracker() as mem:
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, M_pq, nbits)
        index.train(train)
        index.add(train)
        build_time = time.perf_counter() - t0
        mem.snapshot()

    for nprobe in pq_cfg["nprobe_values"]:
        index.nprobe = nprobe
        predicted = _get_all_neighbours(index, queries, k)
        rc = recall_at_k(predicted, ground_truth, k)
        lat = _run_queries(index, queries, k, nprobe=nprobe,
                           warmup=bm["warmup_runs"], runs=bm["n_timing_runs"])
        params = f"nlist={nlist},nprobe={nprobe},M_pq={M_pq},nbits={nbits}"
        logger.info("  nprobe=%d → recall=%.3f  p50=%.2f ms", nprobe, rc, lat["p50_ms"])
        results.append(dict(
            system="FAISS", index_type="IVF-PQ", params=params,
            recall_at_k=round(rc, 4),
            qps=round(lat["qps"], 2),
            p50_ms=round(lat["p50_ms"], 3),
            p95_ms=round(lat["p95_ms"], 3),
            build_time_s=round(build_time, 2),
            peak_memory_mb=round(mem.peak_increase_mb, 1),
        ))

    return results


# ─── entry ────────────────────────────────────────────────────────────────────

def run_all(train: np.ndarray, queries: np.ndarray,
            ground_truth: np.ndarray, cfg: dict) -> list[dict]:
    rows = []
    rows += bench_flat(train, queries, ground_truth, cfg)
    rows += bench_hnsw(train, queries, ground_truth, cfg)
    rows += bench_ivf_flat(train, queries, ground_truth, cfg)
    rows += bench_ivf_pq(train, queries, ground_truth, cfg)
    return rows
