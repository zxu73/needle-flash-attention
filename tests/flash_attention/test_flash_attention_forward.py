"""Forward-pass tests for the FlashAttention-style operator (needle.ops.flash_attention).

Run from the project root:  python -m pytest tests/flash_attention -q
"""
import sys

sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl
from needle.backend_ndarray import ndarray as nd
from needle.ops import FlashAttention, flash_attention, flash_attention_forward_tiled

_DEVICES = [
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
    ndl.cpu_numpy(),
]
_DEVICE_IDS = ["cpu", "cuda", "cpu_numpy"]

# (B, H, Nq, Nk, D): tile-aligned and ragged lengths, Nq != Nk both ways, assorted head dims
_SHAPES = [
    (1, 1, 1, 1, 8),
    (2, 3, 37, 37, 16),
    (1, 2, 64, 64, 32),
    (1, 2, 100, 70, 64),
    (2, 1, 65, 130, 100),
    (1, 1, 5, 9, 128),
]


def reference_attention(q, k, v, causal, scale=None):
    """Naive numpy attention, materializing the full score matrix.

    Returns (out, lse, visible_rows): rows with no visible key (causal with Nq > Nk)
    are flagged False and their out/lse are left as nan.
    """
    Nq, Nk, D = q.shape[-2], k.shape[-2], q.shape[-1]
    scale = 1.0 / np.sqrt(D) if scale is None else scale
    s = (q.astype(np.float64) @ np.swapaxes(k, -1, -2).astype(np.float64)) * scale
    if causal:
        hidden = np.arange(Nk)[None, :] > np.arange(Nq)[:, None] + (Nk - Nq)
        s = np.where(hidden, -np.inf, s)
    visible_rows = np.isfinite(s).any(-1)
    with np.errstate(invalid="ignore"):
        m = s.max(-1, keepdims=True)
        p = np.exp(s - m)
        l = p.sum(-1, keepdims=True)
        out = (p / l) @ v.astype(np.float64)
        lse = (m + np.log(l))[..., 0]
    return out, lse, visible_rows


def _random_qkv(B, H, Nq, Nk, D, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, Nq, D)).astype(np.float32) * scale
    k = rng.standard_normal((B, H, Nk, D)).astype(np.float32) * scale
    v = rng.standard_normal((B, H, Nk, D)).astype(np.float32)
    return q, k, v


def _run_op(q, k, v, device, **kwargs):
    op = FlashAttention(**kwargs)
    out = op(*(ndl.Tensor(x, device=device) for x in (q, k, v)))
    return out.numpy(), op.lse.numpy()


def _assert_matches(out, lse, q, k, v, causal, scale=None, atol=1e-5):
    out_ref, lse_ref, visible = reference_attention(q, k, v, causal, scale)
    np.testing.assert_allclose(out[visible], out_ref[visible], rtol=1e-4, atol=atol)
    np.testing.assert_allclose(lse[visible], lse_ref[visible], rtol=1e-4, atol=atol)
    # rows that see no key at all: zero output, lse = -inf
    np.testing.assert_array_equal(out[~visible], 0.0)
    assert np.all(np.isneginf(lse[~visible]))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_forward_matches_reference(shape, causal, device):
    q, k, v = _random_qkv(*shape)
    out, lse = _run_op(q, k, v, device, causal=causal, block_q=16, block_k=24)
    assert out.shape == q.shape and lse.shape == q.shape[:3]
    _assert_matches(out, lse, q, k, v, causal)


def test_rows_without_visible_keys():
    # causal with Nq > Nk: the first Nq - Nk queries precede every key
    q, k, v = _random_qkv(1, 2, 12, 5, 16)
    for device in [ndl.cpu(), ndl.cpu_numpy()] + ([ndl.cuda()] if ndl.cuda().enabled() else []):
        out, lse = _run_op(q, k, v, device, causal=True)
        np.testing.assert_array_equal(out[:, :, :7], 0.0)
        assert np.all(np.isneginf(lse[:, :, :7]))
        _assert_matches(out, lse, q, k, v, causal=True)


@pytest.mark.parametrize("block_q,block_k", [(1, 1), (7, 5), (16, 64), (64, 16), (256, 256)])
@pytest.mark.parametrize("device", [ndl.cpu(), ndl.cpu_numpy()], ids=["cpu", "cpu_numpy"])
def test_block_sizes(block_q, block_k, device):
    q, k, v = _random_qkv(1, 2, 45, 53, 16, seed=1)
    for causal in (False, True):
        out, lse = _run_op(q, k, v, device, causal=causal, block_q=block_q, block_k=block_k)
        _assert_matches(out, lse, q, k, v, causal)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_tiled_fallback_matches_native(causal):
    q, k, v = _random_qkv(2, 2, 50, 41, 24, seed=2)
    dev = ndl.cpu()
    args = [nd.array(x, device=dev) for x in (q, k, v)]
    out_t, lse_t = flash_attention_forward_tiled(*args, scale=0.3, causal=causal,
                                                 block_q=8, block_k=16)
    out_n, lse_n = _run_op(q, k, v, dev, causal=causal, scale=0.3)
    np.testing.assert_allclose(out_t.numpy(), out_n, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lse_t.numpy(), lse_n, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_custom_scale(device):
    q, k, v = _random_qkv(1, 2, 33, 33, 16, seed=3)
    out, lse = _run_op(q, k, v, device, scale=0.05)
    _assert_matches(out, lse, q, k, v, causal=False, scale=0.05)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_large_logits_are_stable(device):
    # scores in the hundreds would overflow exp() without the running-max subtraction
    q, k, v = _random_qkv(1, 1, 40, 40, 16, seed=4, scale=12.0)
    for causal in (False, True):
        out, lse = _run_op(q, k, v, device, causal=causal, block_q=8, block_k=8)
        assert np.all(np.isfinite(out)) and np.all(np.isfinite(lse))
        _assert_matches(out, lse, q, k, v, causal, atol=1e-4)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_matches_multihead_attention(device, causal):
    """Drop-in agreement with the existing (materializing) nn.MultiHeadAttention."""
    q, k, v = _random_qkv(2, 4, 31, 31, 32, seed=5)
    tq, tk, tv = (ndl.Tensor(x, device=device) for x in (q, k, v))
    mha = ndl.nn.MultiHeadAttention(dropout=0.0, causal=causal, device=device)
    expected, _ = mha(tq, tk, tv)
    out = flash_attention(tq, tk, tv, causal=causal)
    np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_matches_pytorch(device, causal):
    torch = pytest.importorskip("torch")
    q, k, v = _random_qkv(2, 3, 40, 40, 16, seed=9)
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(torch.tensor(x) for x in (q, k, v)), is_causal=causal).numpy()
    out, _ = _run_op(q, k, v, device, causal=causal)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_non_compact_inputs(device):
    # (B, N, H, D) -> (B, H, N, D) via transpose, as AttentionLayer produces
    q, k, v = _random_qkv(2, 9, 3, 3, 16, seed=6)  # read as (B, N, H, D)
    tq, tk, tv = (ndl.ops.transpose(ndl.Tensor(x, device=device), axes=(1, 2)) for x in (q, k, v))
    assert not tq.realize_cached_data().is_compact()
    out = flash_attention(tq, tk, tv, causal=True).numpy()
    out_ref, _, _ = reference_attention(*(np.swapaxes(x, 1, 2) for x in (q, k, v)), causal=True)
    np.testing.assert_allclose(out, out_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
def test_cuda_large_head_dim_falls_back_to_tiled():
    # the cuda kernel is instantiated up to D = 128; larger heads use the NDArray path
    q, k, v = _random_qkv(1, 1, 20, 20, 160, seed=7)
    out, lse = _run_op(q, k, v, ndl.cuda(), causal=True)
    _assert_matches(out, lse, q, k, v, causal=True)


def test_never_allocates_full_score_matrix(monkeypatch):
    """No array of size >= Nq * Nk is allocated by the tiled path or the native path."""
    N, D = 1024, 16
    q, k, v = _random_qkv(1, 1, N, N, D, seed=8)
    sizes = []
    real_make = nd.NDArray.make

    def recording_make(shape, strides=None, device=None, handle=None, offset=0):
        if handle is None:
            sizes.append(nd.prod(shape))
        return real_make(shape, strides=strides, device=device, handle=handle, offset=offset)

    dev = ndl.cpu()
    args = [nd.array(x, device=dev) for x in (q, k, v)]
    monkeypatch.setattr(nd.NDArray, "make", staticmethod(recording_make))

    sizes.clear()
    flash_attention_forward_tiled(*args, scale=0.25, causal=True, block_q=128, block_k=128)
    assert max(sizes) <= max(N * D, 128 * 128) < N * N

    sizes.clear()
    ndl.ops.flash_attention_forward(*args, scale=0.25, causal=True)
    assert max(sizes) <= N * D


def test_is_a_graph_node():
    # the backward pass through this node is covered in test_flash_attention_backward.py
    q, k, v = _random_qkv(1, 1, 8, 8, 8)
    tq, tk, tv = (ndl.Tensor(x, device=ndl.cpu(), requires_grad=True) for x in (q, k, v))
    out = flash_attention(tq, tk, tv)
    assert isinstance(out, ndl.Tensor) and isinstance(out.op, FlashAttention)
    assert out.requires_grad
    assert len(out.inputs) == 3 and all(a is b for a, b in zip(out.inputs, (tq, tk, tv)))


def test_shape_validation():
    q, k, v = _random_qkv(1, 2, 8, 8, 8)
    with pytest.raises(AssertionError):
        flash_attention(*(ndl.Tensor(x[0]) for x in (q, k, v)))          # 3D inputs
    with pytest.raises(AssertionError):
        flash_attention(ndl.Tensor(q), ndl.Tensor(k[:, :1]), ndl.Tensor(v[:, :1]))  # head mismatch
    with pytest.raises(AssertionError):
        flash_attention(ndl.Tensor(q), ndl.Tensor(k), ndl.Tensor(v[:, :, :5]))      # k/v length
