"""
Data preparation: download (or generate) vectors, compute ground truth.

Outputs written to disk:
  data/vectors_train.npy   – (n_train, dim) float32
  data/vectors_query.npy   – (n_queries, dim) float32
  data/ground_truth.npy    – (n_queries, k) int64  (exact top-k IDs)

The ground truth is always recomputed with FAISS IndexFlatL2 so it reflects
the exact nearest neighbours within the 100 k training subset — not the full
1 M set used by ANN-Benchmarks.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ─── loaders ──────────────────────────────────────────────────────────────────

def _load_ann_benchmarks(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_vectors, query_vectors) from the ANN-Benchmarks HDF5."""
    import h5py
    import urllib.request

    hdf5_path = Path(cfg["local_hdf5_path"])
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    if not hdf5_path.exists():
        url = cfg["ann_benchmarks_url"]
        logger.info("Downloading %s  (~468 MB) …", url)
        logger.info("This may take a few minutes on a slow connection.")
        try:
            urllib.request.urlretrieve(url, hdf5_path, _download_progress)
            print()   # newline after progress display
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            hdf5_path.unlink(missing_ok=True)
            raise

    logger.info("Loading HDF5 from %s", hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        n_train = cfg["n_train"]
        n_queries = cfg["n_queries"]
        train = f["train"][:n_train].astype(np.float32)
        test = f["test"][:n_queries].astype(np.float32)
    return train, test


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 ** 2)
        print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)


def _load_sentence_transformers(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Embed AG-News texts with all-MiniLM-L6-v2 as a 384-dim fallback."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    n_total = cfg["n_train"] + cfg["n_queries"]
    dataset_name = cfg["st_dataset"]   # e.g. "fancyzhx/ag_news"
    logger.info("Loading %s dataset …", dataset_name)
    # trust_remote_code=False is safe; no custom script in this dataset
    ds = load_dataset(dataset_name, split="train", trust_remote_code=False)
    texts = [row["text"] for row in ds][:n_total]
    logger.info("Loaded %d texts from %s", len(texts), dataset_name)

    model_name = cfg["st_model"]
    logger.info("Encoding %d texts with %s …", len(texts), model_name)
    logger.info("(This downloads ~90 MB model on first run, then encodes — expect 5-15 min on CPU)")
    model = SentenceTransformer(model_name)
    # batch_size=64 to keep memory low on 8 GB RAM
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)

    train = embeddings[: cfg["n_train"]]
    queries = embeddings[cfg["n_train"] :]
    return train, queries


# ─── ground truth ─────────────────────────────────────────────────────────────

def _compute_ground_truth(train: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    """Exact brute-force nearest neighbours via FAISS IndexFlatL2."""
    import faiss

    logger.info("Computing exact ground truth with FAISS Flat (k=%d) …", k)
    dim = train.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(train)
    _, indices = index.search(queries, k)
    return indices.astype(np.int64)


# ─── public API ───────────────────────────────────────────────────────────────

def prepare(cfg: dict, force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load or build train/query/ground-truth arrays.

    Returns:
        train      – (n_train, dim) float32
        queries    – (n_queries, dim) float32
        gt         – (n_queries, 100) int64  (top-100 exact neighbours)
    """
    data_dir = Path("data")
    train_path = data_dir / "vectors_train.npy"
    query_path = data_dir / "vectors_query.npy"
    gt_path    = data_dir / "ground_truth.npy"

    if not force and train_path.exists() and query_path.exists() and gt_path.exists():
        logger.info("Loading cached data from data/ …")
        return (
            np.load(train_path),
            np.load(query_path),
            np.load(gt_path),
        )

    # ── load vectors ──
    source = cfg.get("source", "ann_benchmarks")
    if source == "ann_benchmarks":
        try:
            train, queries = _load_ann_benchmarks(cfg)
        except Exception as exc:
            logger.warning("ANN-Benchmarks download failed (%s). Falling back to sentence-transformers.", exc)
            train, queries = _load_sentence_transformers(cfg)
    elif source == "sentence_transformers":
        train, queries = _load_sentence_transformers(cfg)
    else:
        raise ValueError(f"Unknown dataset source: {source!r}")

    logger.info("train shape: %s   queries shape: %s", train.shape, queries.shape)

    # ── ground truth ──
    gt = _compute_ground_truth(train, queries, k=100)

    # ── persist ──
    data_dir.mkdir(exist_ok=True)
    np.save(train_path, train)
    np.save(query_path, queries)
    np.save(gt_path, gt)
    logger.info("Saved to data/vectors_train.npy, vectors_query.npy, ground_truth.npy")

    return train, queries, gt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    import yaml
    with open("config.yaml") as f:
        full_cfg = yaml.safe_load(f)
    prepare(full_cfg["dataset"], force="--force" in sys.argv)
