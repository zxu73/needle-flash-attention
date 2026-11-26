# FlashAttention in Needle

A memory-efficient, FlashAttention-style scaled dot-product attention operator for the
[Needle](https://dlsyscourse.org) deep learning framework (CMU 10-414/714 final project).
Built on top of the HW4 version of Needle.

```
out = softmax(scale * Q K^T + causal_mask) V
```

The operator tiles the queries and keys and uses an **online softmax**, so the
`(N x N)` attention matrix is never materialized: extra memory is `O(N * D)` instead of
`O(N^2)`.

## Status

- [x] Forward pass — native CPU and CUDA kernels, plus a device-agnostic NDArray fallback
- [ ] Backward pass (recompute attention tile by tile from the saved log-sum-exp)
- [ ] Integration into `nn.AttentionLayer` / `nn.Transformer`

## Usage

```python
import needle as ndl

# q: (B, H, Nq, D), k / v: (B, H, Nk, D)
out = ndl.ops.flash_attention(q, k, v, causal=True)  # scale defaults to 1/sqrt(D)
```

`ndl.ops.FlashAttention(...)` is the underlying `TensorOp`; after a forward pass its `lse`
attribute holds the `(B, H, Nq)` row log-sum-exp kept for the backward pass.

- **Causal mask** is bottom-right aligned: query `i` sees keys `j <= i + (Nk - Nq)`, matching
  `nn.MultiHeadAttention(causal=True)`. Rows that see no key return `out = 0`, `lse = -inf`.
- **Dispatch**: `cpu` and `cuda` use the native `flash_attention_fwd` kernel; other devices
  (e.g. `cpu_numpy`), and CUDA with head dim > 128, use `flash_attention_forward_tiled`.
- No dropout on the attention probabilities.

## Layout

| Path | Contents |
|---|---|
| `python/needle/ops/ops_attention.py` | `FlashAttention` op, dispatch, NDArray tiled fallback |
| `src/ndarray_backend_cpu.cc` | `FlashAttentionFwd` CPU kernel |
| `src/ndarray_backend_cuda.cu` | `FlashAttentionFwdKernel` CUDA kernel |
| `tests/flash_attention/` | forward-pass tests |
| `apps/bench_flash_attention.py` | benchmark vs `nn.MultiHeadAttention` |

## Build & test

Compiled backends are not checked in; build them first.

```bash
make            # Linux / macOS (cmake)
build_windows.bat && build_windows_cuda.bat   # Windows (MSVC + nvcc)

python -m pytest tests/flash_attention -q
python apps/bench_flash_attention.py --device cuda --causal
```

## Results (forward)

RTX 5070 Ti Laptop GPU, B=1, H=8, D=64, causal:

| N | flash_attention | nn.MultiHeadAttention | extra memory (flash / naive) |
|---|---|---|---|
| 512 | 0.85 ms | 99.9 ms | 1 MB / 512 MB |
| 1024 | 1.44 ms | 376 ms | 2 MB / 2 GB |
| 4096 | 4.73 ms | out of memory | 8 MB / 32 GB |

Outputs agree with the naive implementation to within ~1e-6.
