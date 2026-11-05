"""
ChromaDB benchmark — ephemeral (in-memory) client, no server.

ChromaDB 1.x holds file locks on Windows that survive gc.collect(), so we use
EphemeralClient (in-memory) instead of PersistentClient.  This is fair for a
CPU latency benchmark: the HNSW query path is identical; persistent I/O would
only matter if we were benchmarking insert throughput to disk.

HNSW parameters exposed via collection metadata:
  hnsw:M, hnsw:construction_ef, hnsw:search_ef
"""

import logging
import time

import numpy as np

from utils.memory_tracker import MemoryTracker
from utils.metrics import measure_latencies, recall_at_k

logger = logging.getLogger(__name__)

BATCH_SIZE = 5_000


def _ids_for(start: int, end: int) -> list[str]:
    return [str(i) for i in range(start, end)]


def _add_in_batches(collection, vectors: np.ndarray):
    n = len(vectors)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        collection.add(
            embeddings=vectors[start:end].tolist(),
            ids=_ids_for(start, end),
        )


def _run_config(
    train: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search: int,
    k: int,
    bm_cfg: dict,
) -> dict:
    import chromadb

    client = chromadb.EphemeralClient()

    logger.info("[Chroma] M=%d ef_construction=%d ef_search=%d — building …",
                M, ef_construction, ef_search)
    with MemoryTracker() as mem:
        t0 = time.perf_counter()
        collection = client.create_collection(
            name="bench",
            metadata={
                "hnsw:space": "l2",
                "hnsw:M": M,
                "hnsw:construction_ef": ef_construction,
                "hnsw:search_ef": ef_search,
            },
        )
        _add_in_batches(collection, train)
        build_time = time.perf_counter() - t0
        mem.snapshot()

    # Collect predicted neighbours for recall
    predicted = []
    for q in queries:
        res = collection.query(
            query_embeddings=[q.tolist()],
            n_results=k,
            include=["distances"],
        )
        predicted.append([int(x) for x in res["ids"][0]])
    predicted = np.array(predicted, dtype=np.int64)
    rc = recall_at_k(predicted, ground_truth, k)

    def _query_fn(q: np.ndarray):
        return collection.query(
            query_embeddings=q.tolist(),
            n_results=k,
            include=["distances"],
        )

    lat = measure_latencies(
        _query_fn, queries,
        warmup_runs=bm_cfg["warmup_runs"],
        timing_runs=bm_cfg["n_timing_runs"],
    )
    logger.info("  → recall=%.3f  p50=%.2f ms", rc, lat["p50_ms"])

    params = f"M={M},ef_construction={ef_construction},ef_search={ef_search}"
    return dict(
        system="Chroma", index_type="HNSW", params=params,
        recall_at_k=round(rc, 4),
        qps=round(lat["qps"], 2),
        p50_ms=round(lat["p50_ms"], 3),
        p95_ms=round(lat["p95_ms"], 3),
        build_time_s=round(build_time, 2),
        peak_memory_mb=round(mem.peak_increase_mb, 1),
    )


def run_all(train: np.ndarray, queries: np.ndarray,
            ground_truth: np.ndarray, cfg: dict) -> list[dict]:
    chroma_cfg = cfg["chroma"]
    bm_cfg = cfg["benchmark"]
    k = bm_cfg["k"]
    results = []

    for M in chroma_cfg["M_values"]:
        for ef_c in chroma_cfg["ef_construction_values"]:
            for ef_s in chroma_cfg["ef_search_values"]:
                try:
                    row = _run_config(
                        train, queries, ground_truth,
                        M=M, ef_construction=ef_c, ef_search=ef_s,
                        k=k, bm_cfg=bm_cfg,
                    )
                    results.append(row)
                except Exception as exc:
                    logger.error("[Chroma] M=%d ef_c=%d ef_s=%d failed: %s",
                                 M, ef_c, ef_s, exc)

    return results
