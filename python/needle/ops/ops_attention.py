"""Memory-efficient (FlashAttention-style) scaled dot-product attention.

    out = softmax(scale * Q K^T + mask) V

computed over (query tile, key tile) blocks with an online softmax, so the
(Nq x Nk) attention matrix is never materialized.  Besides the output, the
forward pass produces the row log-sum-exp of the scores,

    lse[i] = log sum_j exp(scale * q_i . k_j)      (over visible keys j),

which is all the backward pass needs to recompute the softmax probabilities
tile by tile (the probabilities themselves are never stored).

Backward pass.  Write S = scale * Q K^T + mask, P = softmax(S), out = P V and
let dout be the adjoint of out.  Then

    dV = P^T dout
    dP = dout V^T
    dS = P * (dP - rowsum(P * dP))        (softmax Jacobian, applied row-wise)
    dQ = scale * dS K
    dK = scale * dS^T Q

All four (Nq x Nk) matrices are only ever needed one tile at a time:

    P_ij      = exp(S_ij - lse_i)                  recomputed from the saved lse
    delta_i   = rowsum(P * dP)_i = dout_i . out_i  since sum_j P_ij v_j = out_i

so delta costs O(N * D) and needs no pass over the keys, and every (query tile,
key tile) block contributes independently to dQ_i, dK_j and dV_j.

Dispatch: devices whose backend provides native ``flash_attention_fwd`` /
``flash_attention_bwd`` kernels (cpu, cuda) use them; anything else (e.g.
cpu_numpy, or a stale compiled backend) falls back to
``flash_attention_forward_tiled`` / ``flash_attention_backward_tiled``, the
same algorithms written with NDArray operations.
"""

import math

import numpy

from ..autograd import NDArray, Tensor, TensorOp, TensorTupleOp
from ..backend_ndarray import ndarray as nd
from .ops_tuple import tuple_get_item

__all__ = [
    "FlashAttention",
    "FlashAttentionBackward",
    "flash_attention",
    "flash_attention_forward",
    "flash_attention_forward_tiled",
    "flash_attention_backward",
    "flash_attention_backward_tiled",
]

# largest head dim the cuda kernel is instantiated for (see ndarray_backend_cuda.cu)
_CUDA_MAX_HEAD_DIM = 128

# Stands in for -inf as the running max / lse of a row that sees no key, so that
# exp(s - m) stays exp(-inf) = 0 instead of exp(-inf + inf) = nan.
_FINITE_FLOOR = -float(numpy.finfo(numpy.float32).max)


def _check_shapes(q_shape, k_shape, v_shape):
    assert len(q_shape) == len(k_shape) == len(v_shape) == 4, (
        "flash_attention expects (batch, num_head, seq_len, dim_head) inputs, got "
        f"{q_shape}, {k_shape}, {v_shape}")
    B, H, _, D = q_shape
    assert k_shape[:2] == (B, H) and v_shape[:2] == (B, H), (
        f"batch/head dims of q {q_shape}, k {k_shape}, v {v_shape} must match")
    assert k_shape == v_shape, f"k {k_shape} and v {v_shape} must have the same shape"
    assert k_shape[3] == D, f"q {q_shape} and k {k_shape} must have the same head dim"


def _has_native_kernel(device, name, head_dim):
    """Whether `device`'s compiled backend provides kernel `name` for this head dim."""
    return hasattr(device, name) and not (device.name == "cuda" and head_dim > _CUDA_MAX_HEAD_DIM)


def _default_scale(scale, head_dim):
    return scale if scale is not None else 1.0 / math.sqrt(head_dim)


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
                m_safe = m_new.maximum(_FINITE_FLOOR)
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

    if not _has_native_kernel(device, "flash_attention_fwd", D):
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


def flash_attention_backward_tiled(q, k, v, out, lse, dout, *, scale, causal=False,
                                   block_q=64, block_k=64):
    """Tiled backward pass, written with NDArray operations.

    Device-agnostic reference for the native ``flash_attention_bwd`` kernels.
    The probabilities of each (query tile, key tile) block are recomputed from
    the saved ``lse``, used for that block's contribution to the three
    gradients, and dropped; nothing of size Nq x Nk is ever allocated.

    Args:
        q: NDArray (B, H, Nq, D);  k, v: NDArray (B, H, Nk, D)
        out, lse: the forward outputs, (B, H, Nq, D) and (B, H, Nq)
        dout: adjoint of ``out``, (B, H, Nq, D)
        scale, causal, block_q, block_k: as in the forward pass

    Returns:
        (dq, dk, dv): the adjoints of q, k and v, with their shapes
    """
    _check_shapes(q.shape, k.shape, v.shape)
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    device = q.device
    BH = B * H
    diag = Nk - Nq

    def rows(x, bh):  # (BH, N, C) -> the (N, C) matrix of one (batch, head)
        return x[bh, :, :].compact().reshape(x.shape[1:])

    q = q.compact().reshape((BH, Nq, D))
    k = k.compact().reshape((BH, Nk, D))
    v = v.compact().reshape((BH, Nk, D))
    out = out.compact().reshape((BH, Nq, D))
    dout = dout.compact().reshape((BH, Nq, D))
    # rows that saw no key have lse = -inf; they are fully masked below, so any
    # finite stand-in gives them P = exp(-inf - lse) = 0 and hence zero gradients
    lse = lse.compact().reshape((BH, Nq, 1)).maximum(_FINITE_FLOOR)
    dq = nd.empty((BH, Nq, D), device=device)
    dk = nd.empty((BH, Nk, D), device=device)
    dv = nd.empty((BH, Nk, D), device=device)

    for bh in range(BH):
        Qb, Kb, Vb = rows(q, bh), rows(k, bh), rows(v, bh)
        Gb = rows(dout, bh)                                              # (Nq, D)
        lse_b = rows(lse, bh)                                            # (Nq, 1)
        delta_b = (Gb * rows(out, bh)).sum(axis=1, keepdims=True)        # (Nq, 1)
        dKb = nd.full((Nk, D), 0.0, device=device)
        dVb = nd.full((Nk, D), 0.0, device=device)

        for i0 in range(0, Nq, block_q):
            i1 = min(i0 + block_q, Nq)
            br = i1 - i0
            Qi = Qb[i0:i1, :].compact() * scale                         # (br, D)
            Gi = Gb[i0:i1, :].compact()                                 # (br, D)
            dQi = nd.full((br, D), 0.0, device=device)

            # keys past j_end are invisible to every row of this tile
            j_end = max(0, min(Nk, i1 + diag)) if causal else Nk
            for j0 in range(0, j_end, block_k):
                j1 = min(j0 + block_k, j_end)
                bc = j1 - j0
                Kj = Kb[j0:j1, :].compact()
                Vj = Vb[j0:j1, :].compact()

                S = Qi @ Kj.permute((1, 0)).compact()                     # (br, bc)
                if causal:
                    S = S + _causal_tile_mask(i0, i1, j0, j1, diag, device)
                P = (S - lse_b[i0:i1, :].broadcast_to((br, bc))).exp()
                dP = Gi @ Vj.permute((1, 0)).compact()
                dS = P * (dP - delta_b[i0:i1, :].broadcast_to((br, bc)))

                dQi = dQi + dS @ Kj
                dKb[j0:j1, :] = dKb[j0:j1, :] + dS.permute((1, 0)).compact() @ Qi
                dVb[j0:j1, :] = dVb[j0:j1, :] + P.permute((1, 0)).compact() @ Gi

            dq[bh, i0:i1, :] = dQi * scale

        dk[bh, :, :] = dKb    # scale is already folded into Qi
        dv[bh, :, :] = dVb

    return (dq.reshape((B, H, Nq, D)), dk.reshape((B, H, Nk, D)), dv.reshape((B, H, Nk, D)))


def flash_attention_backward(q, k, v, out, lse, dout, *, scale, causal=False,
                             block_q=64, block_k=64):
    """Backward pass on NDArrays: native kernel when the device has one, else the tiled fallback.

    Returns (dq, dk, dv), the adjoints of q, k and v given ``dout``, the adjoint of ``out``.
    """
    _check_shapes(q.shape, k.shape, v.shape)
    B, H, Nq, D = q.shape
    Nk = k.shape[2]
    device = q.device
    assert out.shape == dout.shape == q.shape, (
        f"out {out.shape} and its adjoint {dout.shape} must have the shape of q {q.shape}")
    assert lse.shape == (B, H, Nq), f"lse {lse.shape} must be {(B, H, Nq)}"

    if not _has_native_kernel(device, "flash_attention_bwd", D):
        return flash_attention_backward_tiled(
            q, k, v, out, lse, dout, scale=scale, causal=causal,
            block_q=block_q, block_k=block_k)

    inputs = [x.compact() for x in (q, k, v, out, lse, dout)]
    grads = [NDArray.make(x.shape, device=device) for x in (q, k, v)]
    args = [x._handle for x in inputs + grads] + [B * H, Nq, Nk, D, float(scale), bool(causal)]
    if device.name != "cuda":  # the cuda kernel's tile sizes are compile-time constants
        args += [block_q, block_k]
    device.flash_attention_bwd(*args)
    return tuple(grads)


def _on_ndarrays(fn, *arrays):
    """Call fn on NDArrays.  Under NEEDLE_BACKEND=np the arrays are numpy: run on cpu_numpy."""
    if not isinstance(arrays[0], numpy.ndarray):
        return fn(*arrays)
    device = nd.cpu_numpy()
    return tuple(x.numpy() for x in fn(*(nd.array(a, device=device) for a in arrays)))


class FlashAttention(TensorOp):
    """Fused scaled dot-product attention: softmax(scale * Q K^T + causal mask) V.

    Inputs q (B, H, Nq, D), k and v (B, H, Nk, D); output (B, H, Nq, D).
    After compute(), ``self.lse`` holds the (B, H, Nq) row log-sum-exp, which
    gradient() hands to FlashAttentionBackward.  No dropout on the attention
    probabilities.
    """

    def __init__(self, causal=False, scale=None, block_q=64, block_k=64):
        assert block_q > 0 and block_k > 0
        self.causal = causal
        self.scale = scale
        self.block_q = block_q
        self.block_k = block_k
        self.lse = None
        self._lse_out = None  # the output array self.lse belongs to

    def _forward(self, q, k, v):
        return _on_ndarrays(
            lambda q, k, v: flash_attention_forward(
                q, k, v, scale=_default_scale(self.scale, q.shape[-1]), causal=self.causal,
                block_q=self.block_q, block_k=self.block_k),
            q, k, v)

    def compute(self, q, k, v):
        out, self.lse = self._forward(q, k, v)
        self._lse_out = out
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        q, k, v = node.inputs
        lse = self.lse
        if node.realize_cached_data() is not self._lse_out:
            # this op instance has since been applied to other inputs: redo the forward for lse
            _, lse = self._forward(*(x.realize_cached_data() for x in node.inputs))
        grads = FlashAttentionBackward(
            lse, causal=self.causal, scale=self.scale,
            block_q=self.block_q, block_k=self.block_k)(q, k, v, node, out_grad)
        return tuple(tuple_get_item(grads, i) for i in range(3))


class FlashAttentionBackward(TensorTupleOp):
    """Adjoints of FlashAttention's inputs, computed tile by tile from the saved lse.

    Inputs q, k, v (the forward inputs), out (the forward output) and dout (its
    adjoint); output the tuple (dq, dk, dv).  ``lse`` is the forward op's row
    log-sum-exp; it is a constant of this op rather than a graph input.
    """

    def __init__(self, lse, causal=False, scale=None, block_q=64, block_k=64):
        self.lse = lse
        self.causal = causal
        self.scale = scale
        self.block_q = block_q
        self.block_k = block_k

    def compute(self, q, k, v, out, dout):
        return _on_ndarrays(
            lambda q, k, v, out, dout, lse: flash_attention_backward(
                q, k, v, out, lse, dout, scale=_default_scale(self.scale, q.shape[-1]),
                causal=self.causal, block_q=self.block_q, block_k=self.block_k),
            q, k, v, out, dout, self.lse)

    def gradient(self, out_grad, node):
        raise NotImplementedError("FlashAttention does not support second-order gradients")


def flash_attention(q, k, v, causal=False, scale=None, block_q=64, block_k=64):
    """Memory-efficient attention over (batch, num_head, seq_len, dim_head) tensors.

    scale defaults to 1/sqrt(dim_head).  With causal=True, query i attends to keys
    j <= i + (Nk - Nq), matching nn.MultiHeadAttention(causal=True).  Differentiable
    with respect to q, k and v.
    """
    return FlashAttention(causal=causal, scale=scale, block_q=block_q, block_k=block_k)(q, k, v)
