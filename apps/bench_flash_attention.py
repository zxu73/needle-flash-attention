"""Operator benchmark: needle.ops.flash_attention against standard attention and PyTorch.

Run from the project root:
    python apps/bench_flash_attention.py --device cuda --causal --out results/attention_cuda.json

Implementations compared on (B, H, N, D) self-attention inputs:
    flash   needle.ops.flash_attention (this project)
    naive   needle.nn.MultiHeadAttention, the standard attention: it materializes the
            (B, H, N, N) scores and, inside its batched matmul, (B, H, N, N, D) products
    torch   torch.nn.functional.scaled_dot_product_attention (skipped if torch is missing)

Two modes are measured for each sequence length N:
    forward            inference; q, k, v do not require gradients
    forward+backward   the forward pass plus the gradients of q, k, v for a given adjoint of
                       the output (passed to backward() directly, so no loss op is timed)

and for each the median wall time, the peak memory allocated on top of the inputs, and the
largest absolute deviation from flash_attention's output and gradients.  Timed runs end with
one copy of a result to the host, which is what synchronizes the cuda backend.
"""
import argparse
import sys

sys.path.append("./python")

import numpy as np

import needle as ndl
from bench_common import (GB, PeakMemory, describe_hardware, format_cell, get_device, measure,
                          save_results)

MODES = ["forward", "forward+backward"]
LABELS = {"flash": "needle flash_attention (ours)",
          "naive": "needle nn.MultiHeadAttention (standard)",
          "torch": "PyTorch scaled_dot_product_attention"}

# Peak memory of the standard implementations relative to their largest intermediate, measured
# with this script at small N.  Only used to decide which runs to skip (--budget-gb): a run that
# overshoots physical memory on a cpu is killed by the OS instead of failing cleanly.
NAIVE_PEAK_FACTOR = {"forward": 3.1, "forward+backward": 8.5}    # x the (B, H, N, N, D) array
TORCH_PEAK_FACTOR = {"forward": 3.0, "forward+backward": 5.0}    # x the (B, H, N, N) array


class NeedleAttention:
    """flash_attention or nn.MultiHeadAttention applied to needle tensors."""

    def __init__(self, device, causal, flash):
        self.device = device
        if flash:
            self.attend = lambda q, k, v: ndl.ops.flash_attention(q, k, v, causal=causal)
        else:
            mha = ndl.nn.MultiHeadAttention(dropout=0.0, causal=causal, device=device)
            self.attend = lambda q, k, v: mha(q, k, v)[0]
        self.peak_memory = None  # measure() tracks needle buffers itself

    def load(self, q, k, v, dout):
        tensor = lambda x, grad: ndl.Tensor(x, device=self.device, requires_grad=grad)
        self.constants = [tensor(x, False) for x in (q, k, v)]
        self.variables = [tensor(x, True) for x in (q, k, v)]
        self.dout = tensor(dout, False)
        self.last = None

    def forward(self):
        self.last = None  # drop the previous run's arrays before allocating new ones
        out = self.attend(*self.constants)
        out.numpy()  # synchronizes cuda
        self.last = [out]

    def forward_backward(self):
        self.last = None
        for x in self.variables:
            x.grad = None
        out = self.attend(*self.variables)
        out.backward(self.dout)
        self.variables[0].grad.numpy()  # synchronizes cuda
        self.last = [out] + [x.grad for x in self.variables]

    def outputs(self):
        """What the last run computed, as numpy arrays: [out] or [out, dq, dk, dv]."""
        return [t.numpy() for t in self.last]

    def unload(self):
        self.constants = self.variables = self.dout = self.last = None

    def estimated_peak(self, mode, B, H, N, D):
        return 0  # overridden for the materializing implementation in build_implementations


class TorchAttention:
    """torch.nn.functional.scaled_dot_product_attention on the matching torch device.

    A torch build without CUDA support runs on the cpu even when needle runs on cuda; the
    series label says so (see main).
    """

    def __init__(self, device_name, causal):
        import torch

        self.torch = torch
        self.causal = causal
        self.device = "cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu"

    def _attend(self, q, k, v):
        return self.torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal)

    def _synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def load(self, q, k, v, dout):
        tensor = lambda x, grad: self.torch.tensor(x, device=self.device, requires_grad=grad)
        self.constants = [tensor(x, False) for x in (q, k, v)]
        self.variables = [tensor(x, True) for x in (q, k, v)]
        self.dout = tensor(dout, False)
        self.last = None

    def forward(self):
        self.last = None
        with self.torch.no_grad():
            out = self._attend(*self.constants)
        self._synchronize()
        self.last = [out]

    def forward_backward(self):
        self.last = None
        for x in self.variables:
            x.grad = None
        out = self._attend(*self.variables)
        out.backward(self.dout)
        self._synchronize()
        self.last = [out] + [x.grad for x in self.variables]

    def outputs(self):
        return [t.detach().cpu().numpy() for t in self.last]

    def unload(self):
        self.constants = self.variables = self.dout = self.last = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def peak_memory(self, run):
        """Peak bytes allocated by `run` on top of what is already held; None on the cpu."""
        if self.device != "cuda":
            run()
            return None
        cuda = self.torch.cuda
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
        before = cuda.memory_allocated()
        run()
        return cuda.max_memory_allocated() - before

    def estimated_peak(self, mode, B, H, N, D):
        return TORCH_PEAK_FACTOR[mode] * 4 * B * H * N * N


def build_implementations(names, device, device_name, causal):
    implementations = {}
    for name in names:
        if name == "torch":
            try:
                implementations[name] = TorchAttention(device_name, causal)
            except ImportError:
                print("torch is not installed: skipping the PyTorch comparison")
        else:
            implementations[name] = NeedleAttention(device, causal, flash=(name == "flash"))
    if "naive" in implementations:
        implementations["naive"].estimated_peak = (
            lambda mode, B, H, N, D: NAIVE_PEAK_FACTOR[mode] * 4 * B * H * N * N * D)
    return implementations


def max_abs_diff(arrays, reference):
    return float(max(np.abs(a - r).max() for a, r in zip(arrays, reference)))


def print_table(records, mode, names):
    rows = sorted({r["N"] for r in records})
    find = lambda N, name: next((r for r in records if (r["N"], r["mode"], r["impl"])
                                 == (N, mode, name)), None)
    others = [n for n in names if n != "flash"]
    header = (f"{'N':>6} |" + "".join(f" {n + ' ms':>10}" for n in names) + " |"
              + "".join(f" {n + ' MB':>10}" for n in names) + " |"
              + "".join(f" {n + ' diff':>11}" for n in others))
    print(f"\n{mode}\n{header}\n{'-' * len(header)}")
    for N in rows:
        cells = [find(N, name) for name in names]
        print(f"{N:>6} |" + "".join(f" {format_cell(c, 'time_ms', '.2f'):>10}" for c in cells)
              + " |" + "".join(f" {format_cell(c, 'peak_mb', '.1f'):>10}" for c in cells)
              + " |" + "".join(f" {format_cell(find(N, n), 'max_abs_diff', '.1e'):>11}"
                               for n in others))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "cpu_numpy"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64, help="head dimension D")
    parser.add_argument("--lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--impls", nargs="+", default=["flash", "naive", "torch"],
                        choices=["flash", "naive", "torch"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget-gb", type=float, default=4.0,
                        help="skip a standard implementation when its estimated peak memory "
                             "exceeds this")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="cpu threads for PyTorch (needle's cpu backend uses one)")
    parser.add_argument("--out", default=None, help="write the records to this JSON file")
    args = parser.parse_args()

    device = get_device(args.device)
    names = ["flash"] + [n for n in args.impls if n != "flash"]  # flash is the reference
    implementations = build_implementations(names, device, args.device, args.causal)
    names = [n for n in names if n in implementations]
    if "torch" in implementations and args.torch_threads:
        implementations["torch"].torch.set_num_threads(args.torch_threads)

    B, H, D = args.batch, args.heads, args.dim
    meta = dict(benchmark="attention", device=args.device, hardware=describe_hardware(args.device),
                B=B, H=H, D=D, causal=args.causal, repeats=args.repeats, budget_gb=args.budget_gb,
                labels={n: LABELS[n] for n in names})
    if "torch" in implementations:
        torch = implementations["torch"].torch
        torch_device = implementations["torch"].device
        meta.update(torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                    torch_device=torch_device)
        if torch_device != args.device.replace("cpu_numpy", "cpu"):
            print(f"this torch build has no CUDA support: PyTorch runs on the {torch_device}")
            meta["labels"]["torch"] += f" (on {torch_device})"
    print(f"device={args.device} ({meta['hardware']})  B={B} H={H} D={D} causal={args.causal}")

    records = []
    for N in args.lens:
        rng = np.random.default_rng(N)
        q, k, v, dout = (rng.standard_normal((B, H, N, D)).astype(np.float32) for _ in range(4))
        reference = {}
        for name in names:
            impl = implementations[name]
            impl.load(q, k, v, dout)
            for mode in args.modes:
                record = dict(N=N, mode=mode, impl=name, time_ms=None, peak_mb=None,
                              max_abs_diff=None)
                if impl.estimated_peak(mode, B, H, N, D) > args.budget_gb * GB:
                    record["status"] = "skipped"
                else:
                    run = impl.forward if mode == "forward" else impl.forward_backward
                    record.update(measure(run, args.repeats, impl.peak_memory))
                if record["status"] == "ok":
                    outputs = impl.outputs()
                    if name == "flash":
                        reference[mode] = outputs
                    elif mode in reference:  # nothing to compare with if flash itself failed
                        record["max_abs_diff"] = max_abs_diff(outputs, reference[mode])
                records.append(record)
                print(f"  N={N:<5} {mode:<17} {name:<5} "
                      f"{format_cell(record, 'time_ms', '.2f'):>10} ms "
                      f"{format_cell(record, 'peak_mb', '.1f'):>10} MB", flush=True)
            impl.unload()

    for mode in args.modes:
        print_table(records, mode, names)
    if args.out:
        save_results(args.out, meta, records)


if __name__ == "__main__":
    main()
