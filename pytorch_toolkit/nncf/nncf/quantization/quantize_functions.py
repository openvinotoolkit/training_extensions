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
import torch
import warnings

from .extensions import QuantizedFunctionsCPU, QuantizedFunctionsCUDA
from ..dynamic_graph.patch_pytorch import register_operator
from ..functions import STRound, clamp
from ..utils import is_tracing_state, no_jit_trace


class QuantizeSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scale, level_low, level_high, levels):
        input_low = scale * (level_low / level_high)
        input_range = scale - input_low

        if input_.is_cuda:
            if not input_.is_contiguous():
                warnings.warn("input_ is not contiguous!", RuntimeWarning)
                input_ = input_.contiguous()
            output = QuantizedFunctionsCUDA.Quantize_forward(input_, input_low, input_range, levels)
        else:
            output = QuantizedFunctionsCPU.Quantize_forward(input_, input_low, input_range, levels)

        ctx.save_for_backward(input_, input_low, input_range)
        ctx.levels = levels
        ctx.level_low = level_low
        ctx.level_high = level_high

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high

        if grad_output.is_cuda:
            if not grad_output.is_contiguous():
                warnings.warn("grad_output is not contiguous!", RuntimeWarning)
                grad_output = grad_output.contiguous()

            grad_input, _, grad_scale = QuantizedFunctionsCUDA.Quantize_backward(
                grad_output, input_, input_low, input_range, levels, level_low, level_high
            )
        else:
            grad_input, _, grad_scale = QuantizedFunctionsCPU.Quantize_backward(
                grad_output, input_, input_low, input_range, levels, level_low, level_high, False
            )

        return grad_input, grad_scale, None, None, None


class QuantizeAsymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, input_low, input_range, level_low, level_high, levels):
        if input_.is_cuda:
            if not input_.is_contiguous():
                warnings.warn("input_ is not contiguous!", RuntimeWarning)
                input_ = input_.contiguous()
            output = QuantizedFunctionsCUDA.Quantize_forward(input_, input_low, input_range, levels)
        else:
            output = QuantizedFunctionsCPU.Quantize_forward(input_, input_low, input_range, levels)

        ctx.save_for_backward(input_, input_low, input_range)
        ctx.levels = levels
        ctx.level_low = level_low
        ctx.level_high = level_high

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high

        if grad_output.is_cuda:
            if not grad_output.is_contiguous():
                warnings.warn("grad_output is not contiguous!", RuntimeWarning)
                grad_output = grad_output.contiguous()

            grad_input, grad_input_low, grad_input_range = QuantizedFunctionsCUDA.Quantize_backward(
                grad_output, input_, input_low, input_range, levels, level_low, level_high
            )
        else:
            grad_input, grad_input_low, grad_input_range = QuantizedFunctionsCPU.Quantize_backward(
                grad_output, input_, input_low, input_range, levels, level_low, level_high, True
            )

        return grad_input, grad_input_low, grad_input_range, None, None, None


def _quantize_autograd_to_range(input_, input_low, input_high, levels):
    input_ = input_ - input_low
    input_range = (input_high - input_low)
    scale = (levels - 1) / input_range
    output = clamp(input_, low=input_.new_zeros(input_.shape), high=input_range)
    output = output * scale
    output = STRound.apply(output)
    output = output * input_range / (levels - 1) + input_low
    return output


class ExportQuantize(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_, levels, input_low, input_high, output_low, output_high):
        return g.op("FakeQuantize", input_, input_low, input_high, output_low, output_high, levels_i=levels)

    @staticmethod
    def forward(ctx, input_, levels, input_low, input_high, output_low, output_high):
        output = _quantize_autograd_to_range(input_, input_low, input_high, levels)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # backward is not used during export
        return grad_output


@register_operator()
def symmetric_quantize(input_, levels, level_low, level_high, scale, eps):
    if is_tracing_state():
        with no_jit_trace():
            input_range = abs(scale) + eps
            # todo: take bias into account during input_low/input_high calculation
            input_low = input_range * level_low / level_high
            input_high = input_range
        return ExportQuantize.apply(input_, levels, input_low, input_high, input_low, input_high)
    scale_safe = abs(scale) + eps
    return QuantizeSymmetric.apply(input_, scale_safe, level_low, level_high, levels)


@register_operator()
def asymmetric_quantize(input_, levels, level_low, level_high, input_low, input_range, eps):
    if is_tracing_state():
        with no_jit_trace():
            input_range_safe = abs(input_range) + eps
            input_low_tuned, input_range_tuned = TuneRange.apply(input_low, input_range_safe, levels)
            input_high_tuned = input_low_tuned + input_range_tuned
        return ExportQuantize.apply(input_, levels, input_low_tuned, input_high_tuned, input_low_tuned,
                                    input_high_tuned)
    input_range_safe = abs(input_range) + eps
    input_low_tuned, input_range_tuned = TuneRange.apply(input_low, input_range_safe, levels)
    return QuantizeAsymmetric.apply(input_, input_low_tuned, input_range_tuned, level_low, level_high, levels)


class TuneRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_low, input_range, levels):
        input_high = input_range + input_low
        input_low_copy = input_low.clone()
        input_low_copy[input_low_copy > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        scale = levels / (input_high - input_low_copy)
        zp = torch.round(-input_low_copy * scale)

        new_input_low = torch.where(zp < n, zp / (zp - n) * input_high, input_low_copy)
        new_input_high = torch.where(zp > 0., (zp - n) / zp * input_low_copy, input_high)

        range_1 = input_high - new_input_low
        range_2 = new_input_high - input_low_copy

        mask = (range_1 > range_2).to(input_high.dtype)
        inv_mask = (1 - mask).abs()

        new_input_low = mask * new_input_low + inv_mask * input_low_copy
        new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

        return new_input_low, new_input_range

    @staticmethod
    def backward(ctx, grad_input_low, grad_input_range):
        return grad_input_low, grad_input_range, None
