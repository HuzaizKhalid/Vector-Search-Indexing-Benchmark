"""
Lightweight RSS memory tracker using psutil.

psutil measures the OS-level Resident Set Size, which captures both Python
objects and C/C++ heap allocations (e.g. FAISS index storage).  This is more
accurate than tracemalloc for native-extension-heavy workloads.
"""

import os
import psutil


def _rss_mb() -> float:
    """Current RSS of this process in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


class MemoryTracker:
    """
    Context manager that records the peak memory increase during a block.

    Usage:
        with MemoryTracker() as mem:
            build_index(...)
        print(f"Index used {mem.peak_increase_mb:.1f} MB extra")
    """

    def __init__(self):
        self.baseline_mb: float = 0.0
        self.peak_mb: float = 0.0
        self.peak_increase_mb: float = 0.0

    def __enter__(self):
        self.baseline_mb = _rss_mb()
        self.peak_mb = self.baseline_mb
        return self

    def snapshot(self):
        """Call periodically inside the block to capture intermediate peaks."""
        current = _rss_mb()
        if current > self.peak_mb:
            self.peak_mb = current

    def __exit__(self, *_):
        self.snapshot()
        self.peak_increase_mb = max(0.0, self.peak_mb - self.baseline_mb)
