"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad * b * power(a, b - 1)
        grad_b = out_grad * node * log(a)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b * b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axis1, axis2 = self.axes if self.axes is not None else (-2, -1)
        order = list(range(len(a.shape)))
        order[axis1], order[axis2] = order[axis2], order[axis1]
        return a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # NDArray.reshape only works on compact arrays (no copy, just a new view)
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        out_shape = self.shape
        # right-align the input shape against the output shape: leading axes were
        # added by the broadcast, and size-1 axes were stretched by it
        extra = len(out_shape) - len(in_shape)
        axes = list(range(extra))
        for i, dim in enumerate(in_shape):
            if dim == 1 and out_shape[i + extra] != 1:
                axes.append(i + extra)
        if not axes:
            return reshape(out_grad, in_shape)
        return reshape(summation(out_grad, axes=tuple(axes)), in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # a full reduction is a 0-d scalar, as in numpy (the backend gives (1,))
            return a.sum().reshape(())
        axes = (self.axes,) if isinstance(self.axes, int) else self.axes
        # the NDArray backend reduces one axis at a time; go from the last axis
        # down so the remaining axis numbers stay valid
        for ax in sorted((ax % a.ndim for ax in axes), reverse=True):
            a = a.sum(axis=ax)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        if self.axes is None:
            axes = range(len(in_shape))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes
        # put back the axes that were summed away, then stretch along them
        keep_shape = list(in_shape)
        for ax in axes:
            keep_shape[ax] = 1
        return broadcast_to(reshape(out_grad, tuple(keep_shape)), in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        if a.ndim == 2 and b.ndim == 2:
            return a @ b
        # batched matmul: the NDArray backend only multiplies 2D arrays, so
        # broadcast the batch dims and multiply one matrix at a time
        m, n, p = a.shape[-2], a.shape[-1], b.shape[-1]
        batch_a, batch_b = a.shape[:-2], b.shape[:-2]
        nb = max(len(batch_a), len(batch_b))
        batch_a = (1,) * (nb - len(batch_a)) + tuple(batch_a)
        batch_b = (1,) * (nb - len(batch_b)) + tuple(batch_b)
        batch = tuple(max(x, y) for x, y in zip(batch_a, batch_b))
        size = 1
        for d in batch:
            size *= d
        a = a.compact().reshape(batch_a + (m, n)) if a.ndim == 2 else a
        b = b.compact().reshape(batch_b + (n, p)) if b.ndim == 2 else b
        a = a.broadcast_to(batch + (m, n)).compact().reshape((size, m, n))
        b = b.broadcast_to(batch + (n, p)).compact().reshape((size, n, p))
        out = array_api.empty((size, m, p), device=a.device)
        for i in range(size):
            out[i, :, :] = (
                a[i, :, :].compact().reshape((m, n))
                @ b[i, :, :].compact().reshape((n, p))
            )
        return out.reshape(batch + (m, p))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        # matmul broadcasts leading batch dims; sum those back out
        if len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes=tuple(range(len(grad_a.shape) - len(a.shape))))
        if len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes=tuple(range(len(grad_b.shape) - len(b.shape))))
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # relu is not twice differentiable, so reaching into the raw data for the
        # 0/1 mask is safe here
        mask = node.realize_cached_data() > 0
        return out_grad * Tensor.make_const(mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # d tanh(x) / dx = 1 - tanh(x)^2, and node already holds tanh(x)
        return out_grad * (-(node**2) + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = len(args)
        assert n > 0, "Stack needs at least one array"
        shape = tuple(args[0].shape)
        for arr in args:
            assert tuple(arr.shape) == shape, "All arrays need to be of the same size"
        out_shape = shape[: self.axis] + (n,) + shape[self.axis :]
        out = array_api.empty(out_shape, device=args[0].device)
        idx = [slice(None)] * len(out_shape)
        for i, arr in enumerate(args):
            idx[self.axis] = i
            out[tuple(idx)] = arr
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        out_shape = tuple(d for i, d in enumerate(A.shape) if i != self.axis)
        idx = [slice(None)] * A.ndim
        parts = []
        for i in range(n):
            idx[self.axis] = i
            parts.append(A[tuple(idx)].compact().reshape(out_shape))
        return tuple(parts)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack([out_grad[i] for i in range(len(out_grad))], self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        step = self.dilation + 1
        axes = _valid_axes(self.axes, a.ndim)
        new_shape = list(a.shape)
        idx = [slice(None)] * a.ndim
        for ax in axes:
            new_shape[ax] *= step
            idx[ax] = slice(None, None, step)
        # every (dilation+1)-th entry holds the input; zeros in between and after
        out = array_api.full(tuple(new_shape), 0.0, device=a.device)
        out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        step = self.dilation + 1
        idx = [slice(None)] * a.ndim
        for ax in _valid_axes(self.axes, a.ndim):
            idx[ax] = slice(None, None, step)
        return a[tuple(idx)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, C_in) NHWC input; B: (K, K, C_in, C_out) kernel
        A = A.compact()
        if self.padding > 0:
            p = self.padding
            A = A.pad(((0, 0), (p, p), (p, p), (0, 0)))
        N, H, W, C_in = A.shape
        KH, KW, _, C_out = B.shape
        s = self.stride
        Ns, Hs, Ws, Cs = A.strides
        H_out = (H - KH) // s + 1
        W_out = (W - KW) // s + 1
        inner = KH * KW * C_in
        # im2col: a strided view whose last three axes are each output pixel's
        # receptive field, then one big matmul against the flattened kernel
        cols = A.as_strided(
            (N, H_out, W_out, KH, KW, C_in),
            (Ns, Hs * s, Ws * s, Hs, Ws, Cs),
        ).compact().reshape((N * H_out * W_out, inner))
        out = cols @ B.compact().reshape((inner, C_out))
        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        N, H, W_in, C_in = X.shape
        K = W.shape[0]
        s, p = self.stride, self.padding

        # Undo the stride: put s-1 zeros between output-gradient entries.  Dilate
        # also appends zeros after the last entry; drop any that run past the
        # last window position so the shapes below line up exactly.
        G = out_grad
        if s > 1:
            G = dilate(G, (1, 2), s - 1)
            G = _narrow(G, 1, 0, H + 2 * p - K + 1)
            G = _narrow(G, 2, 0, W_in + 2 * p - K + 1)

        # X.grad: "full" convolution of out_grad with the spatially flipped
        # kernel, whose in/out channels are swapped: (K, K, C_out, C_in)
        W_t = transpose(flip(W, (0, 1)), (2, 3))
        pad_x = K - 1 - p
        if pad_x >= 0:
            X_grad = conv(G, W_t, padding=pad_x)
        else:
            X_grad = conv(G, W_t, padding=0)
            X_grad = _narrow(_narrow(X_grad, 1, -pad_x, H), 2, -pad_x, W_in)

        # W.grad: treat C_in as the batch and N as the channels, so the conv
        # itself sums over the batch; out_grad acts as the (D, D, N, C_out) kernel
        X_t = transpose(X, (0, 3))                    # (C_in, H, W, N)
        G_t = transpose(transpose(G, (0, 1)), (1, 2))  # (D, D, N, C_out)
        W_grad = conv(X_t, G_t, padding=p)            # (C_in, K, K, C_out)
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))  # (K, K, C_in, C_out)
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


def _valid_axes(axes, ndim):
    """Normalize negative axes and drop any the array doesn't have
    (e.g. dilating axis 2 of a 2D array is a no-op)."""
    return [ax % ndim for ax in axes if -ndim <= ax < ndim]


def _narrow(a, axis, start, length):
    """Keep entries [start, start + length) of `a` along `axis`.

    Built from split/stack so it stays differentiable. The conv gradient only
    needs it for odd shapes (e.g. a stride that doesn't evenly tile the
    input, or padding > kernel_size - 1); the common cases skip it entirely.
    """
    if start == 0 and length == a.shape[axis]:
        return a
    parts = split(a, axis)
    return stack([parts[i] for i in range(start, start + length)], axis)


