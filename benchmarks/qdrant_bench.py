"""
Qdrant benchmark — local embedded mode (no server, no Docker).

qdrant-client >= 1.7 ships a pure-Python embedded backend activated by:
    QdrantClient(path="./qdrant_store")

IMPORTANT LIMITATION (discovered at runtime):
    The Python embedded backend performs EXACT brute-force search —
    search_params (hnsw_ef) has no effect.  This is documented behaviour:
    "Local mode performs exact (brute-force) search."
    Real HNSW is only available in the full Qdrant Rust server.

For this benchmark we therefore run a single Qdrant config (no ef_search
sweep) and label it "Exact" to reflect what it actually does.  It serves as
a useful comparison point: Qdrant overhead vs FAISS Flat on the same data.
"""

import logging
import shutil
import time
from pathlib import Path

import numpy as np

from utils.memory_tracker import MemoryTracker
from utils.metrics import measure_latencies, recall_at_k

logger = logging.getLogger(__name__)

COLLECTION_NAME = "bench"
UPSERT_BATCH = 2_000   # points per batch; keep low for the pure-Python backend


def _make_client(storage_path: str):
    from qdrant_client import QdrantClient
    store = Path(storage_path)
    if store.exists():
        shutil.rmtree(store)
    store.mkdir(parents=True)
    return QdrantClient(path=str(store))


def _build_collection(client, dim: int, M: int, ef_construct: int, distance_str: str = "Euclid"):
    from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

    dist_map = {"Euclid": Distance.EUCLID, "Cosine": Distance.COSINE, "Dot": Distance.DOT}
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=dist_map[distance_str]),
        hnsw_config=HnswConfigDiff(m=M, ef_construct=ef_construct),
    )


def _add_points(client, vectors: np.ndarray):
    from qdrant_client.models import PointStruct

    n = len(vectors)
    for start in range(0, n, UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, n)
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload={})
            for i in range(start, end)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.debug("  upserted %d points", n)


def _run_config(
    train: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    M: int,
    ef_construct: int,
    ef_search: int,
    k: int,
    storage_path: str,
    bm_cfg: dict,
) -> dict:
    from qdrant_client.models import SearchParams

    client = _make_client(storage_path)
    dim = train.shape[1]

    logger.info("[Qdrant] M=%d ef_construct=%d ef_search=%d — building …",
                M, ef_construct, ef_search)
    with MemoryTracker() as mem:
        t0 = time.perf_counter()
        _build_collection(client, dim, M, ef_construct)
        _add_points(client, train)
        build_time = time.perf_counter() - t0
        mem.snapshot()

    sp = SearchParams(hnsw_ef=ef_search, exact=False)

    # qdrant-client >= 1.7 uses query_points(); search() was removed in 1.20
    def _query_fn(q: np.ndarray):
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=q[0].tolist(),
            limit=k,
            search_params=sp,
        )

    # Collect IDs for recall
    predicted = []
    for q in queries:
        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q.tolist(),
            limit=k,
            search_params=sp,
        )
        predicted.append([pt.id for pt in resp.points])
    predicted = np.array(predicted, dtype=np.int64)
    rc = recall_at_k(predicted, ground_truth, k)

    lat = measure_latencies(
        _query_fn, queries,
        warmup_runs=bm_cfg["warmup_runs"],
        timing_runs=bm_cfg["n_timing_runs"],
    )

    # Must close before shutil.rmtree on Windows (releases the lock file)
    client.close()

    logger.info("  → recall=%.3f  p50=%.2f ms", rc, lat["p50_ms"])

    # Local mode is brute-force; ef_search has no effect — label it "Exact"
    params = f"M={M},ef_construct={ef_construct},mode=exact_brute_force"
    return dict(
        system="Qdrant", index_type="Exact", params=params,
        recall_at_k=round(rc, 4),
        qps=round(lat["qps"], 2),
        p50_ms=round(lat["p50_ms"], 3),
        p95_ms=round(lat["p95_ms"], 3),
        build_time_s=round(build_time, 2),
        peak_memory_mb=round(mem.peak_increase_mb, 1),
    )


def run_all(train: np.ndarray, queries: np.ndarray,
            ground_truth: np.ndarray, cfg: dict) -> list[dict]:
    """
    Run one Qdrant config per (M, ef_construct) pair.
    ef_search is irrelevant because the embedded backend is brute-force.
    """
    import warnings
    q_cfg = cfg["qdrant"]
    bm_cfg = cfg["benchmark"]
    k = bm_cfg["k"]
    storage_path = q_cfg["storage_path"]
    results = []

    logger.warning(
        "[Qdrant] NOTE: qdrant-client embedded mode performs EXACT brute-force "
        "search — ef_search has no effect.  Results labelled index_type='Exact'."
    )

    for M in q_cfg["M_values"]:
        for ef_c in q_cfg["ef_construct_values"]:
            # ef_search=64 is passed but silently ignored by the embedded backend
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    row = _run_config(
                        train, queries, ground_truth,
                        M=M, ef_construct=ef_c, ef_search=64,
                        k=k, storage_path=storage_path, bm_cfg=bm_cfg,
                    )
                    results.append(row)
                except Exception as exc:
                    logger.error("[Qdrant] M=%d ef_c=%d failed: %s", M, ef_c, exc)

    store = Path(storage_path)
    if store.exists():
        shutil.rmtree(store)

    return results
