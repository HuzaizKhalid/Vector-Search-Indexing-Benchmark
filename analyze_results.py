"""
analyze_results.py — load results CSV, generate plots and print tables.

Outputs written to results/:
  recall_vs_latency.png   – scatter: p50 latency vs recall@10 for all configs
  recall_vs_qps.png       – scatter: QPS vs recall@10
  build_time.png          – bar chart per system+index_type
  memory_usage.png        – bar chart per system+index_type
  hnsw_param_sweep.png    – FAISS HNSW: recall vs ef_search per M value
  ivf_param_sweep.png     – FAISS IVF-Flat: recall vs nprobe per nlist value
  summary_table.csv       – best config per (system, index_type)

Usage:
    python analyze_results.py
    python analyze_results.py --csv results/benchmark_results.csv
"""

import argparse
import ast
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless rendering on Windows
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
SYSTEM_COLORS = {
    "FAISS": "#2563EB",   # blue
    "Chroma": "#16A34A",  # green
    "Qdrant": "#DC2626",  # red
}
INDEX_MARKERS = {
    "Flat": "D",
    "HNSW": "o",
    "IVF-Flat": "s",
    "IVF-PQ": "^",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["color"] = df["system"].map(SYSTEM_COLORS).fillna("#888888")
    df["marker"] = df["index_type"].map(INDEX_MARKERS).fillna("x")
    return df


def _save(fig, name: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── plots ────────────────────────────────────────────────────────────────────

def plot_recall_vs_latency(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Recall@10 vs p50 Latency", fontsize=13)

    for (sys, idx), grp in df.groupby(["system", "index_type"]):
        ax.scatter(
            grp["p50_ms"], grp["recall_at_k"],
            color=SYSTEM_COLORS.get(sys, "#888"),
            marker=INDEX_MARKERS.get(idx, "x"),
            label=f"{sys} {idx}",
            s=60, alpha=0.85, edgecolors="white", linewidths=0.4,
        )

    ax.set_xlabel("p50 Latency (ms) — lower is better", fontsize=11)
    ax.set_ylabel("Recall@10 — higher is better", fontsize=11)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3g}"))
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    _save(fig, "recall_vs_latency.png")


def plot_recall_vs_qps(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Recall@10 vs Throughput (QPS)", fontsize=13)

    for (sys, idx), grp in df.groupby(["system", "index_type"]):
        ax.scatter(
            grp["qps"], grp["recall_at_k"],
            color=SYSTEM_COLORS.get(sys, "#888"),
            marker=INDEX_MARKERS.get(idx, "x"),
            label=f"{sys} {idx}",
            s=60, alpha=0.85, edgecolors="white", linewidths=0.4,
        )

    ax.set_xlabel("Queries Per Second — higher is better", fontsize=11)
    ax.set_ylabel("Recall@10 — higher is better", fontsize=11)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.4g}"))
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    _save(fig, "recall_vs_qps.png")


def _bar_chart(df_grp: pd.DataFrame, col: str, title: str, ylabel: str, fname: str):
    labels = df_grp.index.map(lambda x: f"{x[0]}\n{x[1]}")
    colors = [SYSTEM_COLORS.get(sys, "#888") for sys, _ in df_grp.index]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    bars = ax.bar(range(len(labels)), df_grp[col], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    for bar, val in zip(bars, df_grp[col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * df_grp[col].max(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    _save(fig, fname)


def plot_build_times(df: pd.DataFrame):
    summary = (
        df.groupby(["system", "index_type"])["build_time_s"]
        .median()
        .reset_index()
        .set_index(["system", "index_type"])
    )
    _bar_chart(summary, "build_time_s",
               "Median Index Build Time", "Build Time (s)", "build_time.png")


def plot_memory(df: pd.DataFrame):
    summary = (
        df.groupby(["system", "index_type"])["peak_memory_mb"]
        .median()
        .reset_index()
        .set_index(["system", "index_type"])
    )
    _bar_chart(summary, "peak_memory_mb",
               "Median Peak Memory Increase During Build", "Memory (MB)", "memory_usage.png")


def _extract_param(params_str: str, key: str) -> float | None:
    """Parse 'key=val,...' strings and return the numeric value for key."""
    for part in str(params_str).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() == key:
                try:
                    return float(v.strip())
                except ValueError:
                    return None
    return None


def plot_hnsw_param_sweep(df: pd.DataFrame):
    hnsw = df[(df["system"] == "FAISS") & (df["index_type"] == "HNSW")].copy()
    if hnsw.empty:
        return
    hnsw["M"] = hnsw["params"].apply(lambda p: _extract_param(p, "M"))
    hnsw["ef_search"] = hnsw["params"].apply(lambda p: _extract_param(p, "efSearch"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("FAISS HNSW: Recall@10 vs ef_search  (per M value)", fontsize=13)

    for M_val, grp in hnsw.groupby("M"):
        grp = grp.sort_values("ef_search")
        ax.plot(grp["ef_search"], grp["recall_at_k"],
                marker="o", label=f"M={int(M_val)}")

    ax.set_xlabel("ef_search", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, "hnsw_param_sweep.png")


def plot_ivf_param_sweep(df: pd.DataFrame):
    ivf = df[(df["system"] == "FAISS") & (df["index_type"] == "IVF-Flat")].copy()
    if ivf.empty:
        return
    ivf["nlist"] = ivf["params"].apply(lambda p: _extract_param(p, "nlist"))
    ivf["nprobe"] = ivf["params"].apply(lambda p: _extract_param(p, "nprobe"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("FAISS IVF-Flat: Recall@10 vs nprobe  (per nlist value)", fontsize=13)

    for nlist_val, grp in ivf.groupby("nlist"):
        grp = grp.sort_values("nprobe")
        ax.plot(grp["nprobe"], grp["recall_at_k"],
                marker="s", label=f"nlist={int(nlist_val)}")

    ax.set_xlabel("nprobe", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    _save(fig, "ivf_param_sweep.png")


# ─── summary table ────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    best = (
        df.sort_values("recall_at_k", ascending=False)
        .groupby(["system", "index_type"])
        .first()
        .reset_index()
        [["system", "index_type", "params", "recall_at_k", "qps", "p50_ms", "p95_ms",
          "build_time_s", "peak_memory_mb"]]
    )
    best.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print("\n=== Best config per (system, index_type) ===")
    print(best.to_string(index=False))
    print("\nSaved to results/summary_table.csv")


def print_scenario_recommendations(df: pd.DataFrame):
    print("\n=== Scenario recommendations (from actual results) ===")

    # Highest recall regardless of speed
    top = df.sort_values("recall_at_k", ascending=False).iloc[0]
    print(f"\n HIGH RECALL: {top.system} {top.index_type} ({top.params})")
    print(f"   recall={top.recall_at_k:.4f}  p50={top.p50_ms:.2f} ms")

    # Lowest latency that still hits recall ≥ 0.9
    fast = df[df["recall_at_k"] >= 0.9].sort_values("p50_ms").iloc[0] if (df["recall_at_k"] >= 0.9).any() else None
    if fast is not None:
        print(f"\n LOW LATENCY (recall≥0.9): {fast.system} {fast.index_type} ({fast.params})")
        print(f"   recall={fast.recall_at_k:.4f}  p50={fast.p50_ms:.2f} ms")

    # Lowest memory
    mem_min = df.sort_values("peak_memory_mb").iloc[0]
    print(f"\n MEMORY EFFICIENT: {mem_min.system} {mem_min.index_type} ({mem_min.params})")
    print(f"   memory={mem_min.peak_memory_mb:.1f} MB  recall={mem_min.recall_at_k:.4f}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/benchmark_results.csv")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"Results file not found: {args.csv}")
        print("Run  python run_benchmark.py  first.")
        return

    print(f"Loading {args.csv} …")
    df = load_results(args.csv)
    print(f"  {len(df)} rows, systems: {sorted(df['system'].unique())}")

    print("\nGenerating plots …")
    plot_recall_vs_latency(df)
    plot_recall_vs_qps(df)
    plot_build_times(df)
    plot_memory(df)
    plot_hnsw_param_sweep(df)
    plot_ivf_param_sweep(df)

    print_summary_table(df)
    print_scenario_recommendations(df)


if __name__ == "__main__":
    main()
