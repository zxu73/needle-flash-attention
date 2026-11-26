"""Memory-efficient (FlashAttention-style) scaled dot-product attention.

    out = softmax(scale * Q K^T + mask) V

computed over (query tile, key tile) blocks with an online softmax, so the
(Nq x Nk) attention matrix is never materialized.  Besides the output, the
forward pass produces the row log-sum-exp of the scores,

    lse[i] = log sum_j exp(scale * q_i . k_j)      (over visible keys j),

which is all the backward pass needs to recompute the softmax probabilities
tile by tile (the probabilities themselves are never stored).

Dispatch: devices whose backend provides a native ``flash_attention_fwd``
kernel (cpu, cuda) use it; anything else (e.g. cpu_numpy, or a stale compiled
backend) falls back to ``flash_attention_forward_tiled``, the same algorithm
written with NDArray operations.
"""

import math

import numpy

from ..autograd import NDArray, Tensor, TensorOp
from ..backend_ndarray import ndarray as nd

__all__ = [
    "FlashAttention",
    "flash_attention",
    "flash_attention_forward",
    "flash_attention_forward_tiled",
]

# largest head dim the cuda kernel is instantiated for (see ndarray_backend_cuda.cu)
_CUDA_MAX_HEAD_DIM = 128


def _check_shapes(q_shape, k_shape, v_shape):
    assert len(q_shape) == len(k_shape) == len(v_shape) == 4, (
        "flash_attention expects (batch, num_head, seq_len, dim_head) inputs, got "
        f"{q_shape}, {k_shape}, {v_shape}")
    B, H, _, D = q_shape
    assert k_shape[:2] == (B, H) and v_shape[:2] == (B, H), (
        f"batch/head dims of q {q_shape}, k {k_shape}, v {v_shape} must match")
    assert k_shape == v_shape, f"k {k_shape} and v {v_shape} must have the same shape"
    assert k_shape[3] == D, f"q {q_shape} and k {k_shape} must have the same head dim"


def _causal_tile_mask(i0, i1, j0, j1, diag, device):
    """(i1-i0, j1-j0) additive mask: -inf where key j is after query i (j > i + diag)."""
    rows = numpy.arange(i0, i1)[:, None]
    cols = numpy.arange(j0, j1)[None, :]
    mask = numpy.where(cols > rows + diag, -numpy.inf, 0.0).astype(numpy.float32)
    return nd.array(mask, device=device)


def flash_attention_forward_tiled(q, k, v, *, scale, causal=False, block_q=64, block_k=64):
    """Tiled attention with online softmax, written with NDArray operations.

    Device-agnostic reference for the native ``flash_attention_fwd`` kernels.
    Only O(block_q * (block_k + D)) memory is live per tile; nothing of size
    Nq x Nk is ever allocated.

    Args:
        q: NDArray (B, H, Nq, D);  k, v: NDArray (B, H, Nk, D)
        scale: multiplier applied to q k^T
        causal: query i only sees keys j <= i + (Nk - Nq) (bottom-right aligned,
            matching nn.MultiHeadAttention's causal mask)
        block_q, block_k: query / key tile sizes

    Returns:
        (out, lse): NDArray (B, H, Nq, D) and NDArray (B, H, Nq)
    """
    _check_shapes(q.shape, k.shape, v.shape)
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    device = q.device
    BH = B * H
    diag = Nk - Nq
    # stands in for -inf as the "max" of a row that has seen no visible key yet,
    # so that exp(s - m) and exp(m_old - m) stay 0 instead of exp(-inf + inf) = nan
    finite_floor = -float(numpy.finfo(numpy.float32).max)

    q = q.compact().reshape((BH, Nq, D))
    k = k.compact().reshape((BH, Nk, D))
    v = v.compact().reshape((BH, Nk, D))
    out = nd.empty((BH, Nq, D), device=device)
    lse = nd.empty((BH, Nq), device=device)

    for bh in range(BH):
        Qb = q[bh, :, :].compact().reshape((Nq, D))
        Kb = k[bh, :, :].compact().reshape((Nk, D))
        Vb = v[bh, :, :].compact().reshape((Nk, D))

        for i0 in range(0, Nq, block_q):
            i1 = min(i0 + block_q, Nq)
            br = i1 - i0
            Qi = Qb[i0:i1, :].compact() * scale                         # (br, D)

            m = nd.full((br, 1), -math.inf, device=device)               # running row max
            l = nd.full((br, 1), 0.0, device=device)                     # running denominator
            O = nd.full((br, D), 0.0, device=device)                     # unnormalized output

            # keys past j_end are invisible to every row of this tile
            j_end = max(0, min(Nk, i1 + diag)) if causal else Nk
            for j0 in range(0, j_end, block_k):
                j1 = min(j0 + block_k, j_end)
                bc = j1 - j0
                Kj = Kb[j0:j1, :]
                Vj = Vb[j0:j1, :].compact()

                S = Qi @ Kj.permute((1, 0)).compact()                     # (br, bc)
                if causal:
                    S = S + _causal_tile_mask(i0, i1, j0, j1, diag, device)

                m_new = m.maximum(S.max(axis=1, keepdims=True))          # (br, 1)
                m_safe = m_new.maximum(finite_floor)
                P = (S - m_safe.broadcast_to((br, bc))).exp()            # (br, bc)
                alpha = (m - m_safe).exp()                               # (br, 1) rescale factor

                l = l * alpha + P.sum(axis=1, keepdims=True)
                O = O * alpha.broadcast_to((br, D)) + P @ Vj
                m = m_new

            # rows that saw no key keep l = 0 and m = -inf: they get out = 0 and lse = -inf.
            # Every other row has l >= 1 (its max term contributes exp(0)), so the clamp is a no-op.
            l = l.maximum(1e-30)
            out[bh, i0:i1, :] = O / l.broadcast_to((br, D))
            lse[bh, i0:i1] = m + l.log()

    return out.reshape((B, H, Nq, D)), lse.reshape((B, H, Nq))


def flash_attention_forward(q, k, v, *, scale, causal=False, block_q=64, block_k=64):
    """Forward pass on NDArrays: native kernel when the device has one, else the tiled fallback.

    Returns (out, lse) with shapes (B, H, Nq, D) and (B, H, Nq).
    """
    _check_shapes(q.shape, k.shape, v.shape)
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    device = q.device

    native = hasattr(device, "flash_attention_fwd") and not (
        device.name == "cuda" and D > _CUDA_MAX_HEAD_DIM)
    if not native:
        return flash_attention_forward_tiled(
            q, k, v, scale=scale, causal=causal, block_q=block_q, block_k=block_k)

    q, k, v = q.compact(), k.compact(), v.compact()
    out = NDArray.make((B, H, Nq, D), device=device)
    lse = NDArray.make((B, H, Nq), device=device)
    args = [q._handle, k._handle, v._handle, out._handle, lse._handle,
            B * H, Nq, Nk, D, float(scale), bool(causal)]
    if device.name != "cuda":  # the cuda kernel's tile sizes are compile-time constants
        args += [block_q, block_k]
    device.flash_attention_fwd(*args)
    return out, lse


class FlashAttention(TensorOp):
    """Fused scaled dot-product attention: softmax(scale * Q K^T + causal mask) V.

    Inputs q (B, H, Nq, D), k and v (B, H, Nk, D); output (B, H, Nq, D).
    After compute(), ``self.lse`` holds the (B, H, Nq) row log-sum-exp, saved for
    the backward pass.  No dropout on the attention probabilities.
    """

    def __init__(self, causal=False, scale=None, block_q=64, block_k=64):
        assert block_q > 0 and block_k > 0
        self.causal = causal
        self.scale = scale
        self.block_q = block_q
        self.block_k = block_k
        self.lse = None

    def compute(self, q, k, v):
        if isinstance(q, numpy.ndarray):  # NEEDLE_BACKEND=np: run the NDArray path on cpu_numpy
            device = nd.cpu_numpy()
            out = self.compute(*(nd.array(x, device=device) for x in (q, k, v)))
            self.lse = self.lse.numpy()
            return out.numpy()

        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(q.shape[-1])
        out, self.lse = flash_attention_forward(
            q, k, v, scale=scale, causal=self.causal,
            block_q=self.block_q, block_k=self.block_k)
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        raise NotImplementedError("FlashAttention backward pass is not implemented yet")


def flash_attention(q, k, v, causal=False, scale=None, block_q=64, block_k=64):
    """Memory-efficient attention over (batch, num_head, seq_len, dim_head) tensors.

    scale defaults to 1/sqrt(dim_head).  With causal=True, query i attends to keys
    j <= i + (Nk - Nq), matching nn.MultiHeadAttention(causal=True).
    """
    return FlashAttention(causal=causal, scale=scale, block_q=block_q, block_k=block_k)(q, k, v)
