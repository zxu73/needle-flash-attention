from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        # self.matmul(a, b) computes a @ b^T over the last two axes, so the keys
        # go in as-is: (B, H, Tq, D) x (B, H, Tk, D) -> (B, H, Tq, Tk)
        logits = self.matmul(q, k) / np.sqrt(q_dim)

        if self.causal:
            # large negative entries above the diagonal; constant, so no grad
            mask = Tensor(
                self.create_causal_mask(queries_len, keys_values_len, q.device),
                device=q.device, dtype=logits.dtype, requires_grad=False)
            logits = logits + mask.broadcast_to(logits.shape)

        probs = self.dropout(self.softmax(logits))

        # probs @ V == self.matmul(probs, V^T): (B, H, Tq, Tk) x (B, H, D, Tk)
        result = self.matmul(probs, ops.transpose(v, axes=(2, 3)))
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        H, D = self.num_head, self.dim_head

        def to_heads(x, seq_len, prenorm, projection):
            # LayerNorm1d / Linear expect 2D input: (B, T, F) -> (B*T, F)
            x = x.reshape((batch_size * seq_len, x.shape[-1]))
            x = projection(prenorm(x))                      # (B*T, H*D)
            # unravel heads from the channel axis: (B, T, H, D) -> (B, H, T, D)
            return ops.transpose(x.reshape((batch_size, seq_len, H, D)), axes=(1, 2))

        q_heads = to_heads(q, queries_len, self.prenorm_q, self.q_projection)
        k_heads = to_heads(k, keys_values_len, self.prenorm_k, self.k_projection)
        v_heads = to_heads(v, keys_values_len, self.prenorm_v, self.v_projection)

        x, self.probs = self.attn(q_heads, k_heads, v_heads)   # (B, H, Tq, D)

        # (B, H, Tq, D) -> (B, Tq, H, D) -> (B*Tq, H*D)
        x = ops.transpose(x, axes=(1, 2)).reshape((batch_size * queries_len, H * D))
        result = self.out_projection(x).reshape(
            (batch_size, queries_len, self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.q_features = q_features
        self.hidden_size = hidden_size

        # x = x + Dropout(Attention(x))   (attention has its own prenorm)
        self.attn = AttentionLayer(
            q_features, num_head, dim_head,
            dropout=dropout, causal=causal, device=device, dtype=dtype)
        self.dropout_attn = Dropout(dropout)

        # x = x + Dropout(Linear2(Dropout(ReLU(Linear1(LayerNorm(x))))))
        self.layernorm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.relu = ReLU()
        self.dropout_mlp_hidden = Dropout(dropout)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.dropout_mlp_out = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        # attention block (self-attention: k = v = x)
        x = x + self.dropout_attn(self.attn(x))

        # MLP block, applied independently at each timestep: (B, T, F) -> (B*T, F)
        h = x.reshape((batch_size * seq_len, x_dim))
        h = self.linear1(self.layernorm(h))
        h = self.dropout_mlp_hidden(self.relu(h))
        h = self.dropout_mlp_out(self.linear2(h))
        x = x + h.reshape((batch_size, seq_len, x_dim))
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.sequence_len = sequence_len

        # learnable positional encoding: timestep id -> vector, used like a word id
        self.positional_embedding = Embedding(
            sequence_len, embedding_size, device=device, dtype=dtype)

        self.layers = Sequential(*[
            TransformerLayer(
                embedding_size, num_head, dim_head, hidden_size,
                dropout=dropout, causal=causal, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, _ = x.shape
        assert seq_len <= self.sequence_len, \
            f"sequence of length {seq_len} exceeds sequence_len={self.sequence_len}"

        # timestep ids 0..T-1 as a (T, 1) "word id" tensor -> (T, 1, D)
        positions = Tensor(
            np.arange(seq_len, dtype=np.float32).reshape(seq_len, 1),
            device=x.device, dtype=self.dtype, requires_grad=False)
        pos = self.positional_embedding(positions)
        # (T, 1, D) -> (1, T, D) -> (B, T, D), shared across the batch
        pos = ops.transpose(pos, axes=(0, 1)).broadcast_to(x.shape)

        x = self.layers(x + pos)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)