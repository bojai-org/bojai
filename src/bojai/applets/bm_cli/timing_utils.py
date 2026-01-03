# pbm_cln/timing_utils.py
import time
from collections import defaultdict

# global registry (shared across all imports)
_TIMES = defaultdict(float)

class Timer:
    """
    Usage:
        with Timer("data_loading"):
            ... code ...
    """
    def __init__(self, label: str):
        self.label = label
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is None:
            return
        dt = time.perf_counter() - self._start
        _TIMES[self.label] += dt

def add_time(label: str, dt: float):
    """Optional: manually add time if not using the context manager."""
    _TIMES[label] += dt

def print_timings():
    print("\n=== Bojai timing summary ===")
    for label, total in sorted(_TIMES.items()):
        print(f"{label:30s}: {total:.3f} s")
    print("=== end timing summary ===\n")