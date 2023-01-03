"""
Modified from Pytorch `MaxUnpool2d` module to support ONNX conversion.

Source: https://github.com/pytorch/pytorch/issues/25088#issuecomment-1090956803
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.pooling import _MaxUnpoolNd
from torch.nn.modules.utils import _pair

class MaxUnpool2dop(Function):
    """We warp the `torch.nn.functional.max_unpool2d`
    with an extra `symbolic` method, which is needed while exporting to ONNX.
    Users should not call this function directly.
    """

    @staticmethod
    def forward(ctx, input, indices, kernel_size, stride, padding,
                output_size):
        """Forward function of MaxUnpool2dop.
        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            kernel_size (Tuple): Size of the max pooling window.
            stride (Tuple): Stride of the max pooling window.
            padding (Tuple): Padding that was added to the input.
            output_size (List or Tuple): The shape of output tensor.
        Returns:
            Tensor: Output tensor.
        """
        return F.max_unpool2d(input, indices, kernel_size, stride, padding,
                              output_size)

    @staticmethod
    def symbolic(g, input, indices, kernel_size, stride, padding, output_size):
        # get shape
        input_shape = g.op('Shape', input)
        const_0 = g.op('Constant', value_t=torch.tensor(0))
        const_1 = g.op('Constant', value_t=torch.tensor(1))
        batch_size = g.op('Gather', input_shape, const_0, axis_i=0)
        channel = g.op('Gather', input_shape, const_1, axis_i=0)

        # height = (height - 1) * stride + kernel_size
        height = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(2)),
            axis_i=0)
        height = g.op('Sub', height, const_1)
        height = g.op('Mul', height,
                      g.op('Constant', value_t=torch.tensor(stride[1])))
        height = g.op('Add', height,
                      g.op('Constant', value_t=torch.tensor(kernel_size[1])))

        # width = (width - 1) * stride + kernel_size
        width = g.op(
            'Gather',
            input_shape,
            g.op('Constant', value_t=torch.tensor(3)),
            axis_i=0)
        width = g.op('Sub', width, const_1)
        width = g.op('Mul', width,
                     g.op('Constant', value_t=torch.tensor(stride[0])))
        width = g.op('Add', width,
                     g.op('Constant', value_t=torch.tensor(kernel_size[0])))

        # step of channel
        channel_step = g.op('Mul', height, width)
        # step of batch
        batch_step = g.op('Mul', channel_step, channel)

        # channel offset
        range_channel = g.op('Range', const_0, channel, const_1)
        range_channel = g.op(
            'Reshape', range_channel,
            g.op('Constant', value_t=torch.tensor([1, -1, 1, 1])))
        range_channel = g.op('Mul', range_channel, channel_step)
        range_channel = g.op('Cast', range_channel, to_i=7)  # 7 is int64

        # batch offset
        range_batch = g.op('Range', const_0, batch_size, const_1)
        range_batch = g.op(
            'Reshape', range_batch,
            g.op('Constant', value_t=torch.tensor([-1, 1, 1, 1])))
        range_batch = g.op('Mul', range_batch, batch_step)
        range_batch = g.op('Cast', range_batch, to_i=7)  # 7 is int64

        # update indices
        indices = g.op('Add', indices, range_channel)
        indices = g.op('Add', indices, range_batch)

        return g.op(
            'MaxUnpool',
            input,
            indices,
            kernel_shape_i=kernel_size,
            strides_i=stride)


class MaxUnpool2d(_MaxUnpoolNd):
    """This module is modified from Pytorch `MaxUnpool2d` module.
    Args:
      kernel_size (int or tuple): Size of the max pooling window.
      stride (int or tuple): Stride of the max pooling window.
          Default: None (It is set to `kernel_size` by default).
      padding (int or tuple): Padding that is added to the input.
          Default: 0.
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, indices, output_size=None):
        """Forward function of MaxUnpool2d.
        Args:
            input (Tensor): Tensor needed to upsample.
            indices (Tensor): Indices output of the previous MaxPool.
            output_size (List or Tuple): The shape of output tensor.
                Default: None.
        Returns:
            Tensor: Output tensor.
        """
        return MaxUnpool2dop.apply(input, indices, self.kernel_size,
                                   self.stride, self.padding, output_size)
