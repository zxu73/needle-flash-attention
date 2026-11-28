"""Backward-pass tests for the FlashAttention-style operator (needle.ops.flash_attention).

The gradients are checked against four independent sources: closed-form numpy gradients of
naive attention, central finite differences of the naive forward pass, needle's own autograd
through nn.MultiHeadAttention, and (when installed) PyTorch.

Run from the project root:  python -m pytest tests/flash_attention -q
"""
import sys

sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl
from needle.backend_ndarray import ndarray as nd
from needle.ops import (
    FlashAttention,
    FlashAttentionBackward,
    flash_attention,
    flash_attention_backward,
    flash_attention_backward_tiled,
    flash_attention_forward,
)

from test_flash_attention_forward import _DEVICE_IDS, _DEVICES, _SHAPES, _random_qkv, reference_attention


def reference_attention_grads(q, k, v, dout, causal, scale=None):
    """Closed-form (dq, dk, dv) of naive attention, materializing every (Nq, Nk) matrix."""
    q, k, v, dout = (x.astype(np.float64) for x in (q, k, v, dout))
    Nq, Nk, D = q.shape[-2], k.shape[-2], q.shape[-1]
    scale = 1.0 / np.sqrt(D) if scale is None else scale
    s = (q @ np.swapaxes(k, -1, -2)) * scale
    if causal:
        hidden = np.arange(Nk)[None, :] > np.arange(Nq)[:, None] + (Nk - Nq)
        s = np.where(hidden, -np.inf, s)
    visible_rows = np.isfinite(s).any(-1, keepdims=True)
    s = np.where(visible_rows, s, 0.0)  # rows that see no key: P = 0
    p = np.exp(s - s.max(-1, keepdims=True))
    p = np.where(visible_rows, p / p.sum(-1, keepdims=True), 0.0)

    dp = dout @ np.swapaxes(v, -1, -2)
    ds = p * (dp - (p * dp).sum(-1, keepdims=True))
    return scale * ds @ k, scale * np.swapaxes(ds, -1, -2) @ q, np.swapaxes(p, -1, -2) @ dout


def _random_dout(q, seed=100):
    return np.random.default_rng(seed).standard_normal(q.shape).astype(np.float32)


def _run_backward(q, k, v, dout, device, **kwargs):
    """Gradients of sum(flash_attention(q, k, v) * dout) w.r.t. q, k, v, through autograd."""
    tq, tk, tv = (ndl.Tensor(x, device=device) for x in (q, k, v))
    out = flash_attention(tq, tk, tv, **kwargs)
    (out * ndl.Tensor(dout, device=device, requires_grad=False)).sum().backward()
    return [t.grad.numpy() for t in (tq, tk, tv)]


def _assert_grads_match(grads, expected, rtol=1e-4, atol=1e-5):
    for name, grad, ref in zip("qkv", grads, expected):
        assert grad.shape == ref.shape
        np.testing.assert_allclose(grad, ref, rtol=rtol, atol=atol, err_msg=f"d{name}")


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_backward_matches_reference(shape, causal, device):
    q, k, v = _random_qkv(*shape)
    dout = _random_dout(q)
    grads = _run_backward(q, k, v, dout, device, causal=causal, block_q=16, block_k=24)
    _assert_grads_match(grads, reference_attention_grads(q, k, v, dout, causal))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_backward_matches_finite_differences(device, causal):
    """Check the derivation itself: central differences of the naive float64 forward pass."""
    q, k, v = (x.astype(np.float64) for x in _random_qkv(1, 2, 5, 7, 4, seed=11))
    dout = _random_dout(q)

    def loss(q, k, v):
        return (reference_attention(q, k, v, causal)[0] * dout).sum()

    eps, numerical = 1e-6, []
    for x in (q, k, v):
        grad = np.zeros_like(x)
        for idx in np.ndindex(*x.shape):
            saved = x[idx]
            x[idx] = saved + eps
            up = loss(q, k, v)
            x[idx] = saved - eps
            down = loss(q, k, v)
            x[idx] = saved
            grad[idx] = (up - down) / (2 * eps)
        numerical.append(grad)

    grads = _run_backward(*(x.astype(np.float32) for x in (q, k, v)), dout, device,
                          causal=causal, block_q=2, block_k=3)
    _assert_grads_match(grads, numerical)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_backward_matches_multihead_attention(device, causal):
    """Same gradients as needle's autograd through the materializing nn.MultiHeadAttention."""
    q, k, v = _random_qkv(2, 4, 31, 31, 32, seed=5)
    dout = ndl.Tensor(_random_dout(q), device=device, requires_grad=False)

    tq, tk, tv = (ndl.Tensor(x, device=device) for x in (q, k, v))
    mha = ndl.nn.MultiHeadAttention(dropout=0.0, causal=causal, device=device)
    (mha(tq, tk, tv)[0] * dout).sum().backward()
    expected = [t.grad.numpy() for t in (tq, tk, tv)]

    grads = _run_backward(q, k, v, dout.numpy(), device, causal=causal)
    _assert_grads_match(grads, expected)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_backward_matches_pytorch(device, causal):
    torch = pytest.importorskip("torch")
    q, k, v = _random_qkv(2, 3, 40, 40, 16, seed=12)
    dout = _random_dout(q)

    tq, tk, tv = (torch.tensor(x, requires_grad=True) for x in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(tq, tk, tv, is_causal=causal)
    (out * torch.tensor(dout)).sum().backward()
    expected = [t.grad.numpy() for t in (tq, tk, tv)]

    grads = _run_backward(q, k, v, dout, device, causal=causal)
    _assert_grads_match(grads, expected)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_rows_without_visible_keys_get_zero_gradient(device):
    # causal with Nq > Nk: the first Nq - Nk queries precede every key (out = 0, lse = -inf)
    q, k, v = _random_qkv(1, 2, 12, 5, 16)
    dout = _random_dout(q)
    grads = _run_backward(q, k, v, dout, device, causal=True, block_q=4, block_k=2)
    assert all(np.all(np.isfinite(g)) for g in grads)
    np.testing.assert_array_equal(grads[0][:, :, :7], 0.0)
    _assert_grads_match(grads, reference_attention_grads(q, k, v, dout, causal=True))


@pytest.mark.parametrize("block_q,block_k", [(1, 1), (7, 5), (16, 64), (64, 16), (256, 256)])
@pytest.mark.parametrize("device", [ndl.cpu(), ndl.cpu_numpy()], ids=["cpu", "cpu_numpy"])
def test_block_sizes(block_q, block_k, device):
    q, k, v = _random_qkv(1, 2, 45, 53, 16, seed=1)
    dout = _random_dout(q)
    for causal in (False, True):
        grads = _run_backward(q, k, v, dout, device, causal=causal,
                              block_q=block_q, block_k=block_k)
        _assert_grads_match(grads, reference_attention_grads(q, k, v, dout, causal))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_tiled_fallback_matches_native(causal):
    q, k, v = _random_qkv(2, 2, 50, 41, 24, seed=2)
    dev = ndl.cpu()
    assert hasattr(dev, "flash_attention_bwd"), "rebuild the cpu backend (make)"
    args = [nd.array(x, device=dev) for x in (q, k, v)]
    out, lse = flash_attention_forward(*args, scale=0.3, causal=causal)
    dout = nd.array(_random_dout(q), device=dev)

    tiled = flash_attention_backward_tiled(*args, out, lse, dout, scale=0.3, causal=causal,
                                           block_q=8, block_k=16)
    native = flash_attention_backward(*args, out, lse, dout, scale=0.3, causal=causal)
    _assert_grads_match([g.numpy() for g in tiled], [g.numpy() for g in native],
                        rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_custom_scale(device):
    q, k, v = _random_qkv(1, 2, 33, 33, 16, seed=3)
    dout = _random_dout(q)
    grads = _run_backward(q, k, v, dout, device, scale=0.05)
    _assert_grads_match(grads, reference_attention_grads(q, k, v, dout, False, scale=0.05))


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_large_logits_are_stable(device):
    # scores in the hundreds: exp(s - lse) must not overflow where exp(s) would
    q, k, v = _random_qkv(1, 1, 40, 40, 16, seed=4, scale=12.0)
    dout = _random_dout(q)
    for causal in (False, True):
        grads = _run_backward(q, k, v, dout, device, causal=causal, block_q=8, block_k=8)
        assert all(np.all(np.isfinite(g)) for g in grads)
        _assert_grads_match(grads, reference_attention_grads(q, k, v, dout, causal),
                            rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_self_attention_inside_a_graph(device, causal):
    """q, k and v derived from one tensor through other ops, as in AttentionLayer."""
    x = _random_qkv(2, 9, 3, 3, 16, seed=6)[0]  # read as (B, N, H, D)
    dout = _random_dout(np.swapaxes(x, 1, 2))

    tx = ndl.Tensor(x, device=device)
    heads = ndl.ops.transpose(tx * 2.0, axes=(1, 2))  # (B, H, N, D), not compact
    assert not heads.realize_cached_data().is_compact()
    out = flash_attention(heads, heads, heads, causal=causal)
    (out * ndl.Tensor(dout, device=device, requires_grad=False)).sum().backward()

    h = 2.0 * np.swapaxes(x, 1, 2)
    dq, dk, dv = reference_attention_grads(h, h, h, dout, causal)
    expected = 2.0 * np.swapaxes(dq + dk + dv, 1, 2)
    np.testing.assert_allclose(tx.grad.numpy(), expected, rtol=1e-4, atol=1e-5)


def test_gradient_is_a_fused_graph_node():
    q, k, v = _random_qkv(1, 1, 8, 8, 8)
    tq, tk, tv = (ndl.Tensor(x, device=ndl.cpu()) for x in (q, k, v))
    out = flash_attention(tq, tk, tv)
    dout = ndl.Tensor(_random_dout(q), device=ndl.cpu())
    grads = out.op.gradient(dout, out)

    assert len(grads) == 3 and all(isinstance(g, ndl.Tensor) for g in grads)
    assert [g.shape for g in grads] == [tq.shape, tk.shape, tv.shape]
    backward_node = grads[0].inputs[0]
    assert isinstance(backward_node.op, FlashAttentionBackward)
    assert all(g.inputs[0] is backward_node for g in grads)  # one kernel call for all three
    with pytest.raises(NotImplementedError):  # no second-order gradients
        grads[0].sum().backward()


@pytest.mark.parametrize("device", [ndl.cpu(), ndl.cpu_numpy()], ids=["cpu", "cpu_numpy"])
def test_reused_op_instance(device):
    """An op applied twice must not differentiate the first call with the second call's lse."""
    op = FlashAttention(causal=True)
    q1, k1, v1 = _random_qkv(1, 2, 10, 10, 8, seed=20)
    q2, k2, v2 = _random_qkv(1, 2, 10, 10, 8, seed=21, scale=3.0)
    dout = _random_dout(q1)

    t1 = [ndl.Tensor(x, device=device) for x in (q1, k1, v1)]
    out1 = op(*t1)
    op(*(ndl.Tensor(x, device=device) for x in (q2, k2, v2)))  # overwrites op.lse
    (out1 * ndl.Tensor(dout, device=device, requires_grad=False)).sum().backward()
    _assert_grads_match([t.grad.numpy() for t in t1],
                        reference_attention_grads(q1, k1, v1, dout, causal=True))


@pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
def test_cuda_large_head_dim_falls_back_to_tiled():
    # the cuda kernels are instantiated up to D = 128; larger heads use the NDArray path
    q, k, v = _random_qkv(1, 1, 20, 20, 160, seed=7)
    dout = _random_dout(q)
    grads = _run_backward(q, k, v, dout, ndl.cuda(), causal=True)
    _assert_grads_match(grads, reference_attention_grads(q, k, v, dout, causal=True))


@pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
def test_cuda_failed_allocation_does_not_poison_later_calls():
    """A failed allocation leaves the cuda runtime's "last error" set; the attention kernels
    check that state after launching and must not report the stale failure as their own."""
    with pytest.raises(RuntimeError):
        nd.NDArray.make((1 << 38,), device=ndl.cuda())  # one terabyte: fails immediately

    q, k, v = _random_qkv(1, 2, 20, 20, 16, seed=13)
    dout = _random_dout(q)
    grads = _run_backward(q, k, v, dout, ndl.cuda(), causal=True)  # forward, then backward
    _assert_grads_match(grads, reference_attention_grads(q, k, v, dout, causal=True))


def test_never_allocates_full_score_matrix(monkeypatch):
    """No array of size >= Nq * Nk is allocated by the tiled path or the native path."""
    N, D = 1024, 16
    q, k, v = _random_qkv(1, 1, N, N, D, seed=8)
    dev = ndl.cpu()
    args = [nd.array(x, device=dev) for x in (q, k, v)]
    out, lse = flash_attention_forward(*args, scale=0.25, causal=True)
    dout = nd.array(_random_dout(q), device=dev)

    sizes = []
    real_make = nd.NDArray.make

    def recording_make(shape, strides=None, device=None, handle=None, offset=0):
        if handle is None:
            sizes.append(nd.prod(shape))
        return real_make(shape, strides=strides, device=device, handle=handle, offset=offset)

    monkeypatch.setattr(nd.NDArray, "make", staticmethod(recording_make))

    flash_attention_backward_tiled(*args, out, lse, dout, scale=0.25, causal=True,
                                   block_q=128, block_k=128)
    assert max(sizes) <= max(N * D, 128 * 128) < N * N

    sizes.clear()
    flash_attention_backward(*args, out, lse, dout, scale=0.25, causal=True)
    assert max(sizes) <= N * D
