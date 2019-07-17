"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from ..extensions._EXTRA import (deform_conv_forward_cuda, deform_conv_backward_input_cuda,
                                 deform_conv_backward_parameters_cuda)


def conv_offset2d(input,
                  offset,
                  weight,
                  stride=1,
                  padding=0,
                  dilation=1,
                  deform_groups=1,
                  kernel_size=1):

    if input is not None and input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))

    return ConvOffset2dFunction.apply(input, offset, weight, _pair(stride),
                                      _pair(padding), _pair(dilation), deform_groups, _pair(kernel_size))


class ConvOffset2dFunction(Function):

    @staticmethod
    def symbolic(g, input, offset, weight, stride=1, padding=0, dilation=1, deformable_groups=1, kernel_size=1):
        return g.op('DeformableConv2D', input, offset, weight,
                    strides_i=stride, pads_i=[p for pair in zip(padding, padding) for p in pair],
                    dilations_i=dilation, deformable_groups_i=deformable_groups, kernel_shape_i=kernel_size)

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, deformable_groups=1, kernel_size=1):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.deformable_groups = deformable_groups

        output = input.new(*ConvOffset2dFunction._output_size(input, weight, ctx.stride, ctx.padding, ctx.dilation))
        bufs_ = [input.new_empty(input.shape), input.new_empty(input.shape)]  # columns, ones
        ctx.save_for_backward(input, offset, weight, bufs_[0], bufs_[1])

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(input, torch.cuda.FloatTensor):
                    raise NotImplementedError
            deform_conv_forward_cuda(
                input, weight, offset, output, bufs_[0], bufs_[1],
                weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0],
                ctx.deformable_groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight, bufs0, bufs1 = ctx.saved_tensors
        bufs_ = [bufs0, bufs1]

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not isinstance(grad_output.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            else:
                if not isinstance(grad_output, torch.cuda.FloatTensor):
                    raise NotImplementedError
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = input.new_zeros(input.size())
                grad_offset = offset.new_zeros(offset.size())
                deform_conv_backward_input_cuda(
                    input, offset, grad_output, grad_input,
                    grad_offset, weight, bufs_[0], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.deformable_groups)

            if ctx.needs_input_grad[2]:
                grad_weight = weight.new_zeros(weight.size())
                deform_conv_backward_parameters_cuda(
                    input, offset, grad_output,
                    grad_weight, bufs_[0], bufs_[1], weight.size(3),
                    weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                    ctx.dilation[0], ctx.deformable_groups, 1)

        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, stride, padding, dilation):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            str = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // str + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be {})'.format(
                    'x'.join(map(str, output_size))))
        return output_size


class ConvOffset2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 num_deformable_groups=1):
        super(ConvOffset2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.num_deformable_groups = num_deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return conv_offset2d(input, offset, self.weight, self.stride,
                             self.padding, self.dilation,
                             self.num_deformable_groups, self.kernel_size)
