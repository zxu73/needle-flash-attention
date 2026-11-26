"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(ops.exp(-x) + 1, -1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"Unknown nonlinearity '{nonlinearity}', expected 'tanh' or 'relu'")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        bound = 1.0 / np.sqrt(hidden_size)

        def uniform(*shape):
            return Parameter(init.rand(*shape, low=-bound, high=bound,
                                       device=device, dtype=dtype, requires_grad=True))

        self.W_ih = uniform(input_size, hidden_size)
        self.W_hh = uniform(hidden_size, hidden_size)
        if bias:
            self.bias_ih = uniform(hidden_size)
            self.bias_hh = uniform(hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        out_shape = (bs, self.hidden_size)
        if h is None:
            # Use explicit zeros (rather than dropping the term) so W_hh is
            # always part of the graph and always receives a gradient.
            h = init.zeros(*out_shape, device=X.device, dtype=X.dtype)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih is not None:
            out = out + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(out_shape)
            out = out + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(out_shape)
        if self.nonlinearity == "tanh":
            return ops.tanh(out)
        return ops.relu(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.rnn_cells = [
            RNNCell(input_size if k == 0 else hidden_size, hidden_size, bias=bias,
                    nonlinearity=nonlinearity, device=device, dtype=dtype)
            for k in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # No tensor indexing: split the time axis / layer axis into lists instead.
        layer_input = list(ops.split(X, 0))
        hs = list(ops.split(h0, 0)) if h0 is not None else [None] * self.num_layers
        h_n = []
        for cell, h in zip(self.rnn_cells, hs):
            layer_output = []
            for x_t in layer_input:
                h = cell(x_t, h)
                layer_output.append(h)
            h_n.append(h)
            layer_input = layer_output
        return ops.stack(layer_input, 0), ops.stack(h_n, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.sigmoid = Sigmoid()

        bound = 1.0 / np.sqrt(hidden_size)

        def uniform(*shape):
            return Parameter(init.rand(*shape, low=-bound, high=bound,
                                       device=device, dtype=dtype, requires_grad=True))

        self.W_ih = uniform(input_size, 4 * hidden_size)
        self.W_hh = uniform(hidden_size, 4 * hidden_size)
        if bias:
            self.bias_ih = uniform(4 * hidden_size)
            self.bias_hh = uniform(4 * hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h0, c0 = h if h is not None else (None, None)
        bs = X.shape[0]
        H = self.hidden_size
        gates_shape = (bs, 4 * H)

        # Explicit zeros (rather than dropping terms) keep W_hh in the graph
        # so it always receives a gradient.
        if h0 is None:
            h0 = init.zeros(bs, H, device=X.device, dtype=X.dtype)
        if c0 is None:
            c0 = init.zeros(bs, H, device=X.device, dtype=X.dtype)

        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih is not None:
            gates = gates + self.bias_ih.reshape((1, 4 * H)).broadcast_to(gates_shape)
            gates = gates + self.bias_hh.reshape((1, 4 * H)).broadcast_to(gates_shape)

        # Gate order matches PyTorch: input, forget, cell (g), output.
        i, f, g, o = ops.split(gates.reshape((bs, 4, H)), 1).tuple()
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), ops.tanh(g), self.sigmoid(o)

        c_next = f * c0 + i * g
        h_next = o * ops.tanh(c_next)
        return h_next, c_next
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.lstm_cells = [
            LSTMCell(input_size if k == 0 else hidden_size, hidden_size, bias=bias,
                     device=device, dtype=dtype)
            for k in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h0, c0 = h if h is not None else (None, None)
        hs = list(ops.split(h0, 0)) if h0 is not None else [None] * self.num_layers
        cs = list(ops.split(c0, 0)) if c0 is not None else [None] * self.num_layers

        layer_input = list(ops.split(X, 0))
        h_n, c_n = [], []
        for cell, h_t, c_t in zip(self.lstm_cells, hs, cs):
            layer_output = []
            for x_t in layer_input:
                h_t, c_t = cell(x_t, (h_t, c_t))
                layer_output.append(h_t)
            h_n.append(h_t)
            c_n.append(c_t)
            layer_input = layer_output
        return ops.stack(layer_input, 0), (ops.stack(h_n, 0), ops.stack(c_n, 0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim,
                                           device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x.reshape((seq_len * bs,)),
                               device=x.device, dtype=self.dtype)
        out = one_hot @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION