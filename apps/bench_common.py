"""Shared pieces of the FlashAttention benchmarks: timing, peak-memory tracking, result files.

A benchmark result file is JSON, {"meta": {...}, "records": [{...}, ...]}, with one record per
measurement; apps/plot_flash_benchmarks.py turns these files into figures.
"""
import gc
import json
import os
import platform
import subprocess
import sys
import time
import weakref

sys.path.append("./python")

import numpy as np

import needle as ndl
from needle.backend_ndarray import ndarray as nd

ELEM_BYTES = 4  # needle arrays are float32
MB = 2**20
GB = 2**30


class PeakMemory:
    """Peak number of bytes held in needle array buffers allocated inside the `with` block.

    Every buffer needle allocates goes through NDArray.make().  It is counted from there until
    the backend frees it, which a weakref finalizer on its handle reports, so the peak includes
    everything the autograd graph keeps alive.  Not counted: buffers that existed before the
    block (the inputs), and the fixed-size scratch space inside the native kernels.
    """

    def __enter__(self):
        gc.collect()
        self.live = self.peak = 0
        self._make = nd.NDArray.make

        def tracking_make(shape, strides=None, device=None, handle=None, offset=0):
            array = self._make(shape, strides=strides, device=device, handle=handle, offset=offset)
            if handle is None:  # a new buffer, not a view of an existing one
                nbytes = nd.prod(shape) * ELEM_BYTES
                self.live += nbytes
                self.peak = max(self.peak, self.live)
                weakref.finalize(array._handle, self._release, nbytes)
            return array

        nd.NDArray.make = staticmethod(tracking_make)
        return self

    def _release(self, nbytes):
        self.live -= nbytes

    def __exit__(self, *exc):
        nd.NDArray.make = staticmethod(self._make)


def is_out_of_memory(error):
    """Whether an exception raised by needle, the cuda runtime or torch is an allocation failure."""
    return isinstance(error, MemoryError) or "memory" in str(error).lower()


def measure(run, repeats, peak_memory=None):
    """Time `run` and record its peak memory.

    Needle's autograd graphs contain reference cycles (a node's gradient is computed from the
    node), so the arrays of a finished run are only freed by the garbage collector.  It is run
    before every measurement, outside the timed region; otherwise consecutive runs pile up
    and exhaust a GPU that a single run fits in several times over.

    Args:
        run: does the work once and returns only when it has finished (i.e. it synchronizes
            the device itself)
        repeats: number of timed runs; the median is reported
        peak_memory: optional callable(run) -> peak bytes or None, replacing the default
            PeakMemory tracking of needle buffers (used for PyTorch)

    Returns:
        dict(status, time_ms, peak_mb); status is "ok" or "oom", and the numbers are None on "oom"
    """
    out_of_memory = False
    try:
        # the memory run doubles as the warm-up, so the tracking overhead is never timed
        gc.collect()
        if peak_memory is not None:
            peak = peak_memory(run)
        else:
            with PeakMemory() as tracker:
                run()
            peak = tracker.peak
        times = []
        for _ in range(repeats):
            gc.collect()
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
    except (MemoryError, RuntimeError) as error:
        if not is_out_of_memory(error):
            raise
        out_of_memory = True
    gc.collect()  # after the except block: its traceback kept the failed run's arrays alive
    if out_of_memory:
        return dict(status="oom", time_ms=None, peak_mb=None)
    return dict(status="ok", time_ms=float(np.median(times)) * 1e3,
                peak_mb=None if peak is None else peak / MB)


def get_device(name):
    device = getattr(ndl, name)()
    if not device.enabled():
        sys.exit(f"needle device '{name}' is not available (is the backend built?)")
    return device


def describe_hardware(device_name):
    """Human-readable name of the processor the benchmark runs on, for figure captions."""
    if device_name == "cuda":
        try:
            return subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            return "CUDA GPU"
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def save_results(path, meta, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=1)
    print(f"\nwrote {len(records)} records to {path}")


def format_cell(record, key, fmt):
    """One table cell: the number, or why there is none."""
    if record is None:
        return "-"
    if record["status"] != "ok":
        return record["status"]
    value = record.get(key)
    return "n/a" if value is None else format(value, fmt)
