"""Integration tests: nn.MultiHeadAttention / AttentionLayer / TransformerLayer / Transformer
with flash=True against the same modules with the default (materializing) attention.

Run from the project root:  python -m pytest tests/flash_attention -q
"""
import sys

sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl
import needle.nn as nn
from needle.ops import FlashAttention

from test_flash_attention_forward import _DEVICE_IDS, _DEVICES

_TOL = dict(rtol=1e-4, atol=1e-5)


def _build_pair(module_cls, *args, **kwargs):
    """The same module twice, identically initialized: (default attention, flash attention)."""
    modules = []
    for flash in (False, True):
        np.random.seed(0)
        modules.append(module_cls(*args, flash=flash, **kwargs))
    naive, flash = modules
    for p_naive, p_flash in zip(naive.parameters(), flash.parameters()):
        np.testing.assert_array_equal(p_naive.numpy(), p_flash.numpy())
    return naive, flash


def _randn(*shape, device, seed=0):
    return ndl.Tensor(np.random.default_rng(seed).standard_normal(shape).astype(np.float32),
                      device=device, requires_grad=False)


def _assert_same_output_and_gradients(naive, flash, forward):
    """forward(module) -> Tensor.  Compares outputs, then every parameter's gradient."""
    out_naive, out_flash = forward(naive), forward(flash)
    np.testing.assert_allclose(out_flash.numpy(), out_naive.numpy(), **_TOL)

    weights = _randn(*out_naive.shape, device=out_naive.device, seed=99)
    (out_naive * weights).sum().backward()
    (out_flash * weights).sum().backward()
    grads = [(p_naive.grad.numpy(), p_flash.grad.numpy())
             for p_naive, p_flash in zip(naive.parameters(), flash.parameters())]
    assert grads
    # float32 round-off is relative to the overall gradient scale, also in parameters whose
    # true gradient is zero (e.g. the key pre-norm bias: shifting every key by the same
    # vector moves all scores of a row equally, which the softmax ignores)
    scale = max(1.0, max(np.abs(g_naive).max() for g_naive, _ in grads))
    for g_naive, g_flash in grads:
        np.testing.assert_allclose(g_flash, g_naive, rtol=1e-4, atol=1e-5 * scale)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_multi_head_attention(device, causal):
    q, k, v = (_randn(2, 4, 19, 16, device=device, seed=s) for s in range(3))
    naive = nn.MultiHeadAttention(causal=causal, device=device)
    flash = nn.MultiHeadAttention(causal=causal, flash=True, device=device)

    expected, probs = naive(q, k, v)
    result, no_probs = flash(q, k, v)
    assert probs is not None and no_probs is None
    np.testing.assert_allclose(result.numpy(), expected.numpy(), **_TOL)


def test_flash_is_off_by_default():
    layer = nn.Transformer(16, 32, 1, num_head=2, dim_head=8, device=ndl.cpu())
    attn = layer.layers.modules[0].attn
    assert attn.attn.flash is False
    layer(_randn(5, 2, 16, device=ndl.cpu()))
    assert attn.probs is not None


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_attention_layer(device, causal):
    naive, flash = _build_pair(nn.AttentionLayer, 27, 4, 8, causal=causal, device=device)
    x = _randn(3, 11, 27, device=device)
    _assert_same_output_and_gradients(naive, flash, lambda layer: layer(x))
    assert naive.probs is not None and flash.probs is None


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_attention_layer_cross_attention(device, causal):
    # keys / values of another length and width than the queries
    naive, flash = _build_pair(nn.AttentionLayer, 27, 4, 8, k_features=13, v_features=21,
                               out_features=10, causal=causal, device=device)
    q, k, v = (_randn(3, n, f, device=device, seed=s)
               for s, (n, f) in enumerate([(9, 27), (14, 13), (14, 21)]))
    _assert_same_output_and_gradients(naive, flash, lambda layer: layer(q, k, v))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_transformer_layer(device, causal):
    naive, flash = _build_pair(nn.TransformerLayer, 27, 4, 8, 32, causal=causal, device=device)
    x = _randn(3, 11, 27, device=device)
    _assert_same_output_and_gradients(naive, flash, lambda layer: layer(x))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_transformer(device, causal):
    naive, flash = _build_pair(nn.Transformer, 24, 32, 2, num_head=4, dim_head=8,
                               causal=causal, device=device, batch_first=True)
    x = _randn(3, 70, 24, device=device)  # longer than one 64-row attention tile
    _assert_same_output_and_gradients(naive, flash, lambda model: model(x)[0])

    # every layer really went through the fused op
    out = flash(x)[0]
    seen, stack, flash_nodes = set(), [out], 0
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        flash_nodes += isinstance(node.op, FlashAttention)
        stack.extend(node.inputs)
    assert flash_nodes == 2


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_dropout(device):
    """flash=True skips only the dropout on the attention probabilities."""
    naive, flash = _build_pair(nn.Transformer, 16, 32, 2, num_head=2, dim_head=8,
                               dropout=0.5, device=device, batch_first=True)
    x = _randn(2, 9, 16, device=device)

    naive.eval(), flash.eval()  # no dropout anywhere: identical
    eval_out = flash(x)[0].numpy()
    np.testing.assert_allclose(eval_out, naive(x)[0].numpy(), **_TOL)

    flash.train()  # the residual / MLP dropouts still apply
    train_out = flash(x)[0]
    assert np.all(np.isfinite(train_out.numpy()))
    assert not np.allclose(train_out.numpy(), eval_out, **_TOL)
    train_out.sum().backward()
    assert all(np.all(np.isfinite(p.grad.numpy())) for p in flash.parameters())


class _TinyLanguageModel(nn.Module):
    """Embedding -> causal Transformer -> linear head, over (seq_len, batch) token ids."""

    def __init__(self, vocab, flash, device):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, 16, device=device)
        self.transformer = nn.Transformer(16, 32, 2, num_head=2, dim_head=8, causal=True,
                                          flash=flash, device=device, sequence_len=32)
        self.head = nn.Linear(16, vocab, device=device)

    def forward(self, tokens):
        seq_len, batch = tokens.shape
        hidden, _ = self.transformer(self.embedding(tokens))
        return self.head(hidden.reshape((seq_len * batch, 16)))


@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICE_IDS)
def test_training_a_language_model(device):
    """Next-token prediction on a periodic sequence: flash trains, and exactly like the default."""
    vocab, seq_len, batch, steps = 7, 12, 8, 30
    starts = np.arange(batch)[None, :]
    tokens = (np.arange(seq_len + 1)[:, None] + starts) % vocab  # (seq_len + 1, batch)
    x = ndl.Tensor(tokens[:-1].astype(np.float32), device=device, requires_grad=False)
    y = ndl.Tensor(tokens[1:].reshape(-1).astype(np.float32), device=device, requires_grad=False)

    losses = {}
    for flash in (False, True):
        np.random.seed(0)
        model = _TinyLanguageModel(vocab, flash, device)
        opt = ndl.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.SoftmaxLoss()
        history = []
        for _ in range(steps):
            opt.reset_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            history.append(float(loss.numpy()))
        losses[flash] = history

    assert losses[True][-1] < 0.5 * losses[True][0]
    np.testing.assert_allclose(losses[True], losses[False], rtol=1e-2, atol=1e-3)
