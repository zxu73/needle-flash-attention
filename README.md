# FlashAttention in Needle

A memory-efficient, FlashAttention-style scaled dot-product attention operator for the
[Needle](https://dlsyscourse.org) deep learning framework (CMU 10-414/714 final project).
Built on top of the HW4 version of Needle.

```
out = softmax(scale * Q K^T + causal_mask) V
```

The operator tiles the queries and keys and uses an **online softmax**, so the
`(N x N)` attention matrix is never materialized: extra memory is `O(N * D)` instead of
`O(N^2)`. The same holds for its gradient, which recomputes the attention probabilities
tile by tile instead of storing them.

## Status

- [x] Forward pass — native CPU and CUDA kernels, plus a device-agnostic NDArray fallback
- [x] Backward pass — native CPU and CUDA kernels, plus a device-agnostic NDArray fallback
- [x] Integration into `nn.MultiHeadAttention` / `nn.AttentionLayer` / `nn.TransformerLayer` /
  `nn.Transformer` (`flash=True`)

## Usage

```python
import needle as ndl

# q: (B, H, Nq, D), k / v: (B, H, Nk, D)
out = ndl.ops.flash_attention(q, k, v, causal=True)  # scale defaults to 1/sqrt(D)
out.sum().backward()                                  # q.grad, k.grad, v.grad

# or as a drop-in for the attention inside the existing modules
model = ndl.nn.Transformer(embedding_size, hidden_size, num_layers, causal=True, flash=True)
```

`ndl.ops.FlashAttention(...)` is the underlying `TensorOp`; after a forward pass its `lse`
attribute holds the `(B, H, Nq)` row log-sum-exp kept for the backward pass. Its gradient is
one more fused node, `ndl.ops.FlashAttentionBackward`, a `TensorTupleOp` that returns
`(dq, dk, dv)` from a single kernel call.

- **Causal mask** is bottom-right aligned: query `i` sees keys `j <= i + (Nk - Nq)`, matching
  `nn.MultiHeadAttention(causal=True)`. Rows that see no key return `out = 0`, `lse = -inf`.
- **Dispatch**: `cpu` and `cuda` use the native `flash_attention_fwd` / `flash_attention_bwd`
  kernels; other devices (e.g. `cpu_numpy`), and CUDA with head dim > 128, use
  `flash_attention_forward_tiled` / `flash_attention_backward_tiled`.
- No dropout on the attention probabilities, and no second-order gradients.

## Backward pass

With `S = scale * Q K^T + mask`, `P = softmax(S)`, `out = P V` and `dout` the adjoint of `out`:

```
dV = P^T dout
dS = P * (dout V^T - delta)        delta_i = sum_j P_ij (dout_i . v_j)
dQ = scale * dS K
dK = scale * dS^T Q
```

Nothing of size `(Nq x Nk)` has to be stored to evaluate this:

- `P_ij = exp(S_ij - lse_i)` is recomputed for one tile at a time from the `lse` saved by the
  forward pass, so no second online-softmax pass is needed.
- `delta_i = dout_i . out_i`, because `sum_j P_ij v_j` is the forward output. The softmax
  Jacobian's row sums therefore cost `O(N * D)` and need no pass over the keys.

Each `(query tile, key tile)` block then adds its share to `dQ_i`, `dK_j` and `dV_j`
independently. The CPU kernel does this in one sweep over the blocks. On CUDA, `dQ` is a sum
over keys per query row while `dK`, `dV` are sums over queries per key row, so two kernels
are launched, one thread per query row and one thread per key row: every thread owns the rows
it writes, which needs no atomics and keeps the result deterministic.

## In a model

`flash=True` on `nn.MultiHeadAttention`, `nn.AttentionLayer`, `nn.TransformerLayer` or
`nn.Transformer` routes the attention through `ops.flash_attention`. It is off by default, and
the parameters, outputs and gradients are the same either way. Two differences follow from the
probabilities never being formed: `MultiHeadAttention.forward` returns `None` for them (so
`AttentionLayer.probs` is `None`), and the dropout on them is skipped; the residual and MLP
dropouts of `TransformerLayer` still apply.

## Layout

| Path | Contents |
|---|---|
| `python/needle/ops/ops_attention.py` | `FlashAttention` / `FlashAttentionBackward` ops, dispatch, NDArray tiled fallbacks |
| `python/needle/nn/nn_transformer.py` | `flash=True` option of the attention and transformer modules |
| `src/ndarray_backend_cpu.cc` | `FlashAttentionFwd`, `FlashAttentionBwd` CPU kernels |
| `src/ndarray_backend_cuda.cu` | `FlashAttentionFwdKernel`, `FlashAttentionBwdQueryKernel`, `FlashAttentionBwdKeyKernel` CUDA kernels |
| `tests/flash_attention/` | forward, backward and module-integration tests |
| `apps/bench_flash_attention.py` | operator benchmark vs `nn.MultiHeadAttention` and PyTorch: time, peak memory, max deviation |
| `apps/bench_flash_transformer.py` | model benchmark: Transformer training step and Penn Treebank loss curves, with and without `flash=True` |
| `apps/plot_flash_benchmarks.py` | figures from the benchmark results |
| `apps/bench_common.py` | timing, peak-memory tracking and result files shared by the benchmarks |
| `colab_benchmarks.ipynb` | Colab notebook: build, test and benchmark on a GPU, with the figures |

## Build & test

Compiled backends are not checked in; build them first.

```bash
make            # Linux / macOS (cmake)
build_windows.bat && build_windows_cuda.bat   # Windows (MSVC + nvcc)

python -m pytest tests/flash_attention -q
```

## Benchmarks

```bash
# operator: forward and forward+backward, vs nn.MultiHeadAttention and PyTorch
python apps/bench_flash_attention.py --device cuda --causal --out results/attention_cuda.json

# model: a Transformer language model with and without flash=True
python apps/bench_flash_transformer.py --device cuda --ptb-dir data/ptb \
    --out results/transformer_cuda.json

# figures (results/figures/*.png)
python apps/plot_flash_benchmarks.py --attention results/attention_cuda.json \
    --transformer results/transformer_cuda.json
```

- **Time** is the median wall time over repeated runs, each ending with a copy of a result to
  the host (which is what synchronizes the CUDA backend).
- **Peak memory** is measured, not estimated: every buffer needle allocates is counted from
  `NDArray.make` until the backend frees it, so the peak includes whatever the autograd graph
  keeps alive. It is the memory on top of the inputs. For PyTorch it is
  `torch.cuda.max_memory_allocated` (CUDA only).
- **Max deviation** is the largest absolute difference from `flash_attention`'s output and
  gradients.
- The standard implementations are skipped where their estimated peak memory exceeds
  `--budget-gb` (default 4), since running out of host memory kills the process; on a GPU a
  larger budget is safe, an allocation failure there is caught and reported as `oom`.
- The Penn Treebank experiment reads `train.txt` and `test.txt` from `--ptb-dir` (the files of
  homework 4) and is skipped without them.

PyTorch is a reference point rather than a target: its attention runs on tuned, multithreaded
BLAS / cuBLAS matmuls, while needle's CPU backend is single-threaded.

## Results (forward)

RTX 5070 Ti Laptop GPU, B=1, H=8, D=64, causal:

| N | flash_attention | nn.MultiHeadAttention | extra memory (flash / naive) |
|---|---|---|---|
| 512 | 0.85 ms | 99.9 ms | 1 MB / 512 MB |
| 1024 | 1.44 ms | 376 ms | 2 MB / 2 GB |
| 4096 | 4.73 ms | out of memory | 8 MB / 32 GB |

Outputs agree with the naive implementation to within ~1e-6.

## Correctness of the gradients

`tests/flash_attention/test_flash_attention_backward.py` checks `dq`, `dk`, `dv` on every
device against closed-form numpy gradients of naive attention, central finite differences,
needle's autograd through `nn.MultiHeadAttention`, and PyTorch's
`scaled_dot_product_attention` (skipped if `torch` is not installed).
`test_flash_attention_nn.py` checks that the modules give the same outputs and parameter
gradients with and without `flash=True`, and that a small causal language model trains to the
same loss curve.
