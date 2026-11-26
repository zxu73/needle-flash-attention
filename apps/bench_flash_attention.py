"""Forward-pass benchmark: needle.ops.flash_attention vs nn.MultiHeadAttention.

Run from the project root:  python apps/bench_flash_attention.py [--device cuda] [--causal]

"score bytes" is the largest attention-sized intermediate each method allocates:
nn.MultiHeadAttention.matmul broadcasts to (B, H, Nq, Nk, D) before reducing, then keeps the
(B, H, Nq, Nk) logits / probabilities; flash_attention allocates only out (B, H, Nq, D) and
lse (B, H, Nq).
"""
import argparse
import sys
import time

sys.path.append("./python")

import numpy as np

import needle as ndl


def _time(fn, repeats):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn().numpy()  # .numpy() also synchronizes the cuda backend
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "cpu_numpy"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--naive-budget-gb", type=float, default=4.0,
                        help="skip MultiHeadAttention when its intermediates exceed this")
    args = parser.parse_args()

    device = getattr(ndl, args.device)()
    if not device.enabled():
        sys.exit(f"device {args.device} is not available")
    B, H, D = args.batch, args.heads, args.dim
    mha = ndl.nn.MultiHeadAttention(dropout=0.0, causal=args.causal, device=device)

    print(f"device={args.device} B={B} H={H} D={D} causal={args.causal}")
    print(f"{'N':>6} | {'flash ms':>9} | {'naive ms':>9} | {'speedup':>7} | "
          f"{'flash bytes':>11} | {'naive bytes':>11} | max |diff|")
    for N in args.lens:
        rng = np.random.default_rng(N)
        q, k, v = (ndl.Tensor(rng.standard_normal((B, H, N, D)).astype(np.float32), device=device)
                   for _ in range(3))
        flash_bytes = 4 * (B * H * N * D + B * H * N)
        naive_bytes = 4 * B * H * N * N * D

        t_flash = _time(lambda: ndl.ops.flash_attention(q, k, v, causal=args.causal), args.repeats)

        if naive_bytes <= args.naive_budget_gb * 2**30:
            t_naive = _time(lambda: mha(q, k, v)[0], args.repeats)
            diff = np.abs(ndl.ops.flash_attention(q, k, v, causal=args.causal).numpy()
                          - mha(q, k, v)[0].numpy()).max()
            naive_cols = f"{t_naive * 1e3:9.2f} | {t_naive / t_flash:6.1f}x"
            diff_col = f"{diff:.1e}"
        else:
            naive_cols, diff_col = f"{'skipped':>9} | {'-':>7}", "-"

        print(f"{N:>6} | {t_flash * 1e3:9.2f} | {naive_cols} | "
              f"{flash_bytes / 2**20:9.1f}MB | {naive_bytes / 2**20:9.1f}MB | {diff_col}")


if __name__ == "__main__":
    main()
