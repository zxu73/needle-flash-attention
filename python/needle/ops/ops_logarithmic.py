from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # softmax over axis 1 of a 2D input; no implicit broadcasting in NDArray
        max_z = Z.max(axis=1, keepdims=True).broadcast_to(Z.shape)
        shifted = Z - max_z
        lse = array_api.log(array_api.exp(shifted).sum(axis=1, keepdims=True))
        return shifted - lse.broadcast_to(Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # y = z - logsumexp(z), so each row's adjoint is
        # out_grad - softmax(z) * sum(out_grad), and softmax(z) = exp(y)
        z = node.inputs[0]
        row_sum = summation(out_grad, axes=(1,))
        row_sum = broadcast_to(reshape(row_sum, (z.shape[0], 1)), z.shape)
        return out_grad - exp(node) * row_sum
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axes = tuple(range(Z.ndim))
        elif isinstance(self.axes, int):
            axes = (self.axes % Z.ndim,)
        else:
            axes = tuple(ax % Z.ndim for ax in self.axes)
        # the NDArray backend reduces one axis at a time and never broadcasts
        # implicitly, so keep the reduced axes as size 1 and broadcast by hand
        max_keep = Z
        for ax in axes:
            max_keep = max_keep.max(axis=ax, keepdims=True)
        sum_keep = array_api.exp(Z - max_keep.broadcast_to(Z.shape))
        for ax in axes:
            sum_keep = sum_keep.sum(axis=ax, keepdims=True)
        out = array_api.log(sum_keep) + max_keep
        out_shape = tuple(d for i, d in enumerate(Z.shape) if i not in axes)
        # a full reduction gives a 0-d scalar, matching Summation and numpy
        return out.compact().reshape(out_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        if self.axes is None:
            axes = range(len(z.shape))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes
        # put back the reduced axes so the output and its adjoint line up with z
        keep_shape = list(z.shape)
        for ax in axes:
            keep_shape[ax] = 1
        keep_shape = tuple(keep_shape)
        lse = broadcast_to(reshape(node, keep_shape), z.shape)
        grad = broadcast_to(reshape(out_grad, keep_shape), z.shape)
        # d logsumexp / dz = softmax(z) = exp(z - logsumexp(z))
        return grad * exp(z - lse)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)