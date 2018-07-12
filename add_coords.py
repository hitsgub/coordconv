from chainer.backends import cuda
from chainer import function_node
import numpy as np


class AddCoords(function_node.FunctionNode):

    """Add coordinates channels function."""

    def __init__(self, with_r=False):
        self.with_r = with_r

    def calc_x_chs(self, gy):
        added_chs = len(gy.shape) - 2
        if self.with_r:
            added_chs += 1
        x_chs = gy.shape[1] - added_chs
        return x_chs

    def forward(self, inputs):
        self.retain_inputs(())
        x, = inputs
        n = x.shape[0]
        xp = cuda.get_array_module(x)
        coords = (np.linspace(-1, 1, w, dtype=np.float32) for w in x.shape[2:])
        grids = np.meshgrid(*coords)
        grids = np.asarray(grids, dtype=np.float32)
        if self.with_r:
            r = np.square(grids)
            r = r.sum(0, keepdims=True)
            grids = np.concatenate([grids, r], axis=0)
        reps = (n,) + (1,) * len(grids.shape)
        grids = np.tile(grids[None], reps)
        grids = xp.asarray(grids, dtype=np.float32)
        y = xp.concatenate([x, grids], axis=1)
        return y,

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        x_chs = self.calc_x_chs(gy)
        gx = gy[:, :x_chs]
        return gx,


def add_coords(x, with_r=False):
    """add coodinates channels to input tensor.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable to operation.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        with_r (bool): Additional channels includes L2norm from center or not.

    Returns:
        ~chainer.Variable: Output variable.
            shape[1] is to be (x.shape[1] + len(x.shape) - 2 + int(with_r))

    .. admonition:: Example

        >>> x = np.arange(12).reshape((3, 2, 2)).astype('f')
        >>> x
        array([[[ 0.,  1.],
                [ 2.,  3.]],
        <BLANKLINE>
               [[ 4.,  5.],
                [ 6.,  7.]],
        <BLANKLINE>
               [[ 8.,  9.],
                [10., 11.]]], dtype=float32)
        >>> y = add_coords(x)
        >>> y.data
        array([[[ 0.,  1.],
                [ 2.,  3.],
                [-1.,  1.]],
        <BLANKLINE>
               [[ 4.,  5.],
                [ 6.,  7.],
                [-1.,  1.]],
        <BLANKLINE>
               [[ 8.,  9.],
                [10., 11.],
                [-1.,  1.]]], dtype=float32)
    """
    y, = AddCoords(with_r).apply((x,))
    return y
