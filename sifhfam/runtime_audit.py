"""Runtime, memory and environment instrumentation (prompt section 13)."""
from __future__ import annotations

import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def environment_snapshot() -> Dict[str, object]:
    snap: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
    }
    for mod in ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "psutil"):
        try:
            m = __import__(mod)
            snap[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            snap[mod] = f"unavailable: {exc}"
    try:
        import psutil
        vm = psutil.virtual_memory()
        snap["ram_total_bytes"] = int(vm.total)
        snap["cpu_percent_note"] = "recorded on demand"
        try:
            snap["cpu_model"] = ""
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        snap["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    except Exception:
        snap["ram_total_bytes"] = None
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts")
        snap["blas_config"] = cfg.get("Build Dependencies", cfg) if isinstance(cfg, dict) else str(cfg)
    except Exception as exc:
        snap["blas_config"] = f"unavailable: {exc}"
    return snap


@dataclass
class StageTimer:
    """Named stage timers for a single feature-selection run."""
    stages: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0)


def current_rss_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def tracemalloc_peak(fn, *args, **kwargs):
    """Run fn and return (result, peak_bytes or None)."""
    try:
        import tracemalloc
    except Exception:
        return fn(*args, **kwargs), None
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        peak = tracemalloc.get_traced_memory()[1]
        return result, int(peak)
    finally:
        try:
            tracemalloc.stop()
        except Exception:
            pass
