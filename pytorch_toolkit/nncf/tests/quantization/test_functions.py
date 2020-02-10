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

import numpy as np
import pytest
import torch
from torch.autograd import Variable

from nncf.quantization.quantize_functions import asymmetric_quantize, symmetric_quantize
from nncf.utils import sum_like
from tests.test_helpers import get_grads, check_equal

EPS = 1


class ReferenceQuantizeAsymmetric:
    @staticmethod
    def forward(input_, input_low, input_range, levels):
        scale = (levels - 1) / input_range
        output = input_.clip(min=input_low, max=input_low + input_range)
        output -= input_low
        output *= scale
        output = output.round()
        output = output / scale
        output += input_low

        return output

    @staticmethod
    def backward(grad_output, input_, input_low, input_range, output, level_low, level_high, range_sign):
        mask_hi = (input_ > input_low + input_range).astype(float)
        mask_lo = (input_ < input_low).astype(float)

        mask_in = 1 - mask_hi - mask_lo
        err = (output - input_) * np.reciprocal(input_range * range_sign)
        grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
        grad_range = sum_like(grad_range, input_range)

        grad_input = grad_output * mask_in

        grad_low = grad_output * (mask_hi + mask_lo)
        grad_low = sum_like(grad_low, input_low)
        return [grad_input, grad_low, grad_range]

    @staticmethod
    def tune_range(input_low, input_range, levels):
        input_high = input_range + input_low
        input_low[input_low > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        scale = levels / (input_high - input_low)
        zp = np.round(-input_low * scale)

        new_input_low = np.where(zp < n, zp / (zp - n) * input_high, input_low)
        new_input_high = np.where(zp > 0., (zp - n) / zp * input_low, input_high)

        range_1 = input_high - new_input_low
        range_2 = new_input_high - input_low

        mask = (range_1 > range_2)
        inv_mask = abs(1 - mask)

        new_input_low = mask * new_input_low + inv_mask * input_low
        new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

        return new_input_low, new_input_range


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def idfn(val):
    if isinstance(val, list):
        return '[{}]'.format('-'.join([str(v) for v in val]))

    return None


@pytest.fixture
def _seed():
    np.random.seed(0)


def generate_input(input_size):
    return 2 * np.random.random_sample(input_size) - 1


def get_test_data(data_list, is_cuda=False, is_backward=False):
    results = []
    for data in data_list:
        result = torch.from_numpy(data.copy())
        if is_cuda:
            result = result.cuda()
        if is_backward:
            result = Variable(result, requires_grad=True)
        results.append(result)
    return results


@pytest.mark.parametrize('input_size',
                         [[1, 96, 112, 112],
                          [1, 192, 28, 28],
                          [1, 576, 14, 14],
                          [32, 96, 112, 112],
                          [32, 192, 28, 28],
                          [32, 576, 14, 14]],
                         ids=idfn)
@pytest.mark.parametrize('bits', (8, 4), ids=('8bit', '4bit'))
@pytest.mark.parametrize("use_cuda", [False, True], ids=['cpu', 'cuda'])
@pytest.mark.parametrize('scale_mode', ["single_scale", "per_channel_scale"])
@pytest.mark.parametrize("is_weights", (True, False), ids=('weights', 'activation'))
class TestParametrized:
    @pytest.mark.parametrize("is_signed", (True, False), ids=('signed', 'unsigned'))
    class TestSymmetric:
        @staticmethod
        def generate_scale(input_, scale_mode, is_weights):
            assert scale_mode in ["single_scale", "per_channel_scale"]

            def calc_scale(input_):
                return min(abs(input_.min()), abs(input_.max())) - input_.mean() / 4

            if scale_mode == "single_scale":
                return np.array([calc_scale(input_)])

            if scale_mode == "per_channel_scale":
                if is_weights:
                    channel_count = input_.shape[0]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[0] = channel_count
                    scales = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[idx, ...]
                        scales[idx] = calc_scale(single_input_channel)
                else:
                    channel_count = input_.shape[1]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[1] = channel_count
                    scales = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[:, idx, ...]
                        scales[0, idx] = calc_scale(single_input_channel)
                return scales

        @staticmethod
        def get_range_level(is_signed, is_weights, bits):
            levels = 2 ** bits
            if is_signed:
                if is_weights:
                    levels -= 1
                level_high = 2 ** (bits - 1) - 1
                level_low = -(level_high + 1)
                if is_weights:
                    level_low += 1
            else:
                level_high = 2 ** bits - 1
                level_low = 0
            return level_low, level_high, levels

        def test_quantize_symmetric_forward(self, _seed, is_signed, is_weights, input_size, bits, use_cuda, scale_mode):
            ref_input = generate_input(input_size)

            ref_scale = self.generate_scale(ref_input, scale_mode, is_weights)

            test_input, test_scale = get_test_data([ref_input, ref_scale], use_cuda)
            level_low, level_high, levels = self.get_range_level(is_signed, is_weights, bits)

            ref_scale = abs(ref_scale) + EPS
            ref_input_low = ref_scale * (level_low / level_high)
            ref_input_range = ref_scale - ref_input_low

            ref_value = ReferenceQuantizeAsymmetric.forward(ref_input, ref_input_low, ref_input_range, levels)

            test_value = symmetric_quantize(test_input, levels, level_low, level_high, test_scale, EPS)

            check_equal(ref_value, test_value, rtol=1e-3)

        def test_quantize_symmetric_backward(self, _seed, is_signed, is_weights, input_size, bits, use_cuda,
                                             scale_mode):
            ref_input = generate_input(input_size)

            ref_scale = self.generate_scale(ref_input, scale_mode, is_weights)
            level_low, level_high, levels = self.get_range_level(is_signed, is_weights, bits)
            test_input, test_scale = get_test_data([ref_input, ref_scale], use_cuda, is_backward=True)

            ref_scale = abs(ref_scale) + EPS
            ref_input_low = ref_scale * (level_low / level_high)
            ref_input_range = ref_scale - ref_input_low

            ref_output = ReferenceQuantizeAsymmetric.forward(ref_input, ref_input_low, ref_input_range, levels)
            ref_grads = ReferenceQuantizeAsymmetric.backward(np.ones(input_size), ref_input, ref_input_low,
                                                             ref_input_range, ref_output, level_low, level_high,
                                                             True)
            del ref_grads[1]
            test_value = symmetric_quantize(test_input, levels, level_low, level_high, test_scale, EPS)
            test_value.sum().backward()
            test_grads = get_grads([test_input, test_scale])

            check_equal(ref_output, test_value)
            check_equal(ref_grads, test_grads)

    @pytest.mark.parametrize("is_negative_range", (True, False), ids=('range<0', 'range>0'))
    class TestAsymmetric:
        @staticmethod
        def generate_range(input_, is_negative_range, scale_mode, is_weights):
            assert scale_mode in ["single_scale", "per_channel_scale"]

            def calc_low_and_range(input_, is_negative_range):
                input_low = input_.min() - input_.mean() / 4
                input_range = input_.max() - input_low
                if is_negative_range:
                    input_range *= -1
                return input_low, input_range

            if scale_mode == "single_scale":
                input_low, input_range = calc_low_and_range(input_, is_negative_range)
                return np.array([input_low]), np.array([input_range])

            if scale_mode == "per_channel_scale":
                if is_weights:
                    channel_count = input_.shape[0]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[0] = channel_count
                    input_low = np.zeros(scales_shape)
                    input_range = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[idx, ...]
                        input_low[idx], input_range[idx] = calc_low_and_range(single_input_channel, is_negative_range)
                else:
                    channel_count = input_.shape[1]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[1] = channel_count
                    input_low = np.zeros(scales_shape)
                    input_range = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[:, idx, ...]
                        input_low[0, idx], input_range[0, idx] = calc_low_and_range(single_input_channel,
                                                                                    is_negative_range)

                return input_low, input_range

        @staticmethod
        def get_range_level(bits):
            levels = 2 ** bits
            level_low = 0
            level_high = levels - 1
            return level_low, level_high, levels

        def test_quantize_asymmetric_forward(self, _seed, input_size, bits, use_cuda, is_negative_range, is_weights,
                                             scale_mode):
            level_low, level_high, levels = self.get_range_level(bits)
            ref_input = generate_input(input_size)
            ref_input_low, ref_input_range = self.generate_range(ref_input, is_negative_range, scale_mode, is_weights)
            test_input, test_input_low, test_input_range = get_test_data(
                [ref_input, ref_input_low, ref_input_range], use_cuda)

            ref_input_range = abs(ref_input_range) + EPS
            ref_input_low, ref_input_range = ReferenceQuantizeAsymmetric.tune_range(
                ref_input_low, ref_input_range, levels)
            ref_value = ReferenceQuantizeAsymmetric.forward(
                ref_input, ref_input_low, ref_input_range, levels)
            test_value = asymmetric_quantize(test_input, levels, level_low, level_high, test_input_low,
                                             test_input_range, EPS)

            check_equal(ref_value, test_value)

        def test_quantize_asymmetric_backward(self, _seed, input_size, bits, use_cuda, is_negative_range, is_weights,
                                              scale_mode):
            level_low, level_high, levels = self.get_range_level(bits)
            ref_input = generate_input(input_size)
            ref_input_low, ref_input_range = self.generate_range(ref_input, is_negative_range, scale_mode, is_weights)
            test_input, test_input_low, test_input_range = get_test_data(
                [ref_input, ref_input_low, ref_input_range], use_cuda, is_backward=True)

            range_sign = np.sign(ref_input_range)
            ref_input_range = abs(ref_input_range) + EPS
            ref_input_low, ref_input_range = ReferenceQuantizeAsymmetric.tune_range(
                ref_input_low, ref_input_range, levels)
            ref_output = ReferenceQuantizeAsymmetric.forward(ref_input, ref_input_low, ref_input_range, levels)
            ref_grads = ReferenceQuantizeAsymmetric.backward(
                np.ones(input_size), ref_input, ref_input_low, ref_input_range, ref_output, level_low,
                level_high, range_sign)

            test_value = asymmetric_quantize(test_input, levels, level_low, level_high, test_input_low,
                                             test_input_range, eps=EPS)
            test_value.sum().backward()
            test_grads = get_grads([test_input, test_input_low, test_input_range])

            check_equal(ref_grads, test_grads)
