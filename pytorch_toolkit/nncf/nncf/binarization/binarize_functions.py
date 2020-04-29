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

import warnings

import torch
from nncf.definitions import get_install_type
if get_install_type() == 'GPU':
    from .extensions import BinarizedFunctionsCUDA


class XNORBinarizeFn(torch.autograd.Function):
    """ Binarizes x into `scale` * { +1; -1}, where +1 or -1 are chosen based
        on whether the x element value is >0 or <0. `scale` is determined as mean of absolute
        values, per input channel (0-th dimension of x). """
    @staticmethod
    def symbolic(g, x):
        zero = g.constant(0, [1], 'float')
        zero = g.op("Unsqueeze", zero, axes_i=[1, 2, 3])
        scale = g.op("Abs", x)
        scale = g.op("ReduceMean", scale, axes_i=[1, 2, 3])
        scale_neg = g.op("Neg", scale)
        return g.op("FakeQuantize", x, zero, zero, scale_neg, scale, levels_i=2)

    @staticmethod
    def forward(ctx, x):
        if x.is_cuda:
            output = BinarizedFunctionsCUDA.WeightBinarize_forward(x, True)
        else:
            # Current CPU kernel implementations do not improve performance
            # output = BinarizedFunctionsCPU.WeightBinarize_forward(x, True)
            norm = x.abs().mean([1, 2, 3], keepdim=True)
            sign = ((x > 0).type(x.dtype) * 2 - 1)
            output = sign * norm
            return output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DOREFABinarizeFn(torch.autograd.Function):
    """ Binarizes x into `scale` * { +1; -1}, where +1 or -1 are chosen based
        on whether the x element value is >0 or <0. `scale` is determined as mean of absolute
        values of the entire x tensor. """
    @staticmethod
    def symbolic(g, x):
        zero = g.constant(0, [1], 'float')
        zero = g.op("Unsqueeze", zero, axes_i=[1, 2, 3])
        scale = g.op("Abs", x)
        scale = g.op("ReduceMean", scale, axes_i=[0, 1, 2, 3])
        scale_neg = g.op("Neg", scale)
        return g.op("FakeQuantize", x, zero, zero, scale_neg, scale, levels_i=2)

    @staticmethod
    def forward(ctx, x):
        if x.is_cuda:
            output = BinarizedFunctionsCUDA.WeightBinarize_forward(x, False)
        else:
            # Current CPU kernel implementations do not improve performance
            # output = BinarizedFunctionsCPU.WeightBinarize_forward(x, False)
            norm = x.abs().mean()
            sign = ((x > 0).type(x.dtype) * 2 - 1)
            output_flat = sign * norm
            return output_flat.view_as(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Activation binarization function
class ActivationBinarizationScaleThresholdFn(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, scale, threshold):
        zero = g.constant(0, [1], 'float')
        zero = g.op("Unsqueeze", zero, axes_i=[0, 2, 3])
        threshold = g.op("Mul", threshold, scale)
        scale = g.op("Unsqueeze", scale, axes_i=[0, 2, 3])
        return g.op("FakeQuantize", x, threshold, threshold, zero, scale, levels_i=2)

    @staticmethod
    def forward(ctx, input_, scale, threshold):
        if input_.is_cuda:
            output = BinarizedFunctionsCUDA.ActivationBinarize_forward(input_, scale, threshold)
        else:
            # Current CPU kernel implementations do not improve performance
            # output = BinarizedFunctionsCPU.ActivationBinarize_forward(input_, scale, threshold)
            shape = [1 for s in input_.shape]
            shape[1] = input_.shape[1]
            t = (threshold * scale).view(shape)
            output = (input_ > t).type(input_.dtype) * scale
            ctx.save_for_backward(input_, scale, output)
        ctx.save_for_backward(input_, scale, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.is_cuda:
            if not grad_output.is_contiguous():
                warnings.warn("grad_output is not contiguous!", RuntimeWarning)
                grad_output = grad_output.contiguous()

        input_, scale, output = ctx.saved_variables

        if input_.is_cuda:
            grad_input, grad_scale, grad_threshold = BinarizedFunctionsCUDA.ActivationBinarize_backward(grad_output,
                                                                                                        input_,
                                                                                                        scale, output)
        else:
            # Current CPU kernel implementations do not improve performance
            # grad_input, grad_scale, grad_threshold = BinarizedFunctionsCPU.ActivationBinarize_backward(grad_output,
            #                                                                                           input_,
            #                                                                                           scale, output)
            # calc gradient for input
            mask_lower = (input_ <= scale).type(input_.dtype)
            grad_input = grad_output * (input_ >= 0).type(input_.dtype) * mask_lower

            # calc gradient for scale
            err = (output - input_) * scale.reciprocal()
            grad_scale = grad_output * (mask_lower * err + (1 - mask_lower))
            grad_scale = grad_scale.sum().view(1)

            # calc gradient for threshold
            grad_threshold = -grad_output * (input_ > 0).type(input_.dtype) * (input_ < scale).type(input_.dtype)

            for idx, _ in enumerate(input_.shape):
                if idx != 1:  # sum over all dims except activations channel
                    grad_threshold = grad_threshold.sum(idx, keepdim=True)

        return grad_input, grad_scale, grad_threshold
