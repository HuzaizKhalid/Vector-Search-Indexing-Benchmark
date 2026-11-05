"""
run_benchmark.py — master orchestrator.

Usage:
    python run_benchmark.py                     # run everything
    python run_benchmark.py --only faiss        # run only FAISS
    python run_benchmark.py --only chroma qdrant
    python run_benchmark.py --config my.yaml    # custom config
    python run_benchmark.py --redownload        # force data re-download

Each system's results are appended to results/benchmark_results.csv as they
complete, so a crash does not lose earlier results.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import yaml

from data.prepare_data import prepare

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLUMNS = [
    "system", "index_type", "params",
    "recall_at_k", "qps", "p50_ms", "p95_ms",
    "build_time_s", "peak_memory_mb",
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_rows(rows: list[dict], csv_path: Path, write_header: bool):
    csv_path.parent.mkdir(exist_ok=True)
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="+", choices=["faiss", "chroma", "qdrant"],
        default=["faiss", "chroma", "qdrant"],
        help="Which systems to benchmark (default: all)",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--redownload", action="store_true",
                        help="Force re-download / re-generate data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = Path(cfg["benchmark"]["output_csv"])

    # ── data ──────────────────────────────────────────────────────────────────
    logger.info("=== Stage 1: Prepare data ===")
    train, queries, ground_truth = prepare(cfg["dataset"], force=args.redownload)
    dim = train.shape[1]
    logger.info("Vectors: train=%s  queries=%s  dim=%d", train.shape, queries.shape, dim)
    cfg["dataset"]["dimension"] = dim   # propagate to index builders if needed

    all_results: list[dict] = []
    write_header = True

    # ── FAISS ─────────────────────────────────────────────────────────────────
    if "faiss" in args.only:
        logger.info("=== Stage 2: FAISS benchmarks ===")
        from benchmarks.faiss_bench import run_all as faiss_run
        t0 = time.perf_counter()
        rows = faiss_run(train, queries, ground_truth, cfg)
        logger.info("FAISS done in %.1f s — %d configs", time.perf_counter() - t0, len(rows))
        save_rows(rows, csv_path, write_header=write_header)
        write_header = False
        all_results += rows

    # ── Chroma ────────────────────────────────────────────────────────────────
    if "chroma" in args.only:
        logger.info("=== Stage 3: ChromaDB benchmarks ===")
        try:
            from benchmarks.chroma_bench import run_all as chroma_run
            t0 = time.perf_counter()
            rows = chroma_run(train, queries, ground_truth, cfg)
            logger.info("Chroma done in %.1f s — %d configs", time.perf_counter() - t0, len(rows))
            save_rows(rows, csv_path, write_header=write_header)
            write_header = False
            all_results += rows
        except ImportError as exc:
            logger.warning("chromadb not installed, skipping (%s)", exc)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    if "qdrant" in args.only:
        logger.info("=== Stage 4: Qdrant benchmarks ===")
        try:
            from benchmarks.qdrant_bench import run_all as qdrant_run
            t0 = time.perf_counter()
            rows = qdrant_run(train, queries, ground_truth, cfg)
            logger.info("Qdrant done in %.1f s — %d configs", time.perf_counter() - t0, len(rows))
            save_rows(rows, csv_path, write_header=write_header)
            all_results += rows
        except ImportError as exc:
            logger.warning("qdrant-client not installed, skipping (%s)", exc)

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("=== Summary: %d configurations benchmarked ===", len(all_results))
    if all_results:
        best = max(all_results, key=lambda r: r["recall_at_k"])
        fastest = min(all_results, key=lambda r: r["p50_ms"])
        logger.info("Best recall  : %s  %s  %s  → %.4f",
                    best["system"], best["index_type"], best["params"], best["recall_at_k"])
        logger.info("Lowest p50   : %s  %s  %s  → %.3f ms",
                    fastest["system"], fastest["index_type"], fastest["params"], fastest["p50_ms"])
        logger.info("Results saved to %s", csv_path)
        logger.info("Run  python analyze_results.py  to generate plots.")


if __name__ == "__main__":
    main()
