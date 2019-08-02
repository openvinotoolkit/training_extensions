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

from nncf.quantization.quantize_functions import QuantizeSymmetric


class ReferenceQuantizeSymmetric:
    @staticmethod
    def forward(input_, scale, level_low, level_high):
        s = level_high / scale

        output = input_ * s
        output = output.clip(min=level_low, max=level_high)
        output = output.round()
        output = output / s

        return output

    @staticmethod
    def backward(grad_output, input_, scale, level_low, level_high, output):
        alpha = level_low / level_high
        mask_hi = (input_ > scale).astype(float)
        mask_lo = (input_ < scale * alpha).astype(float)
        mask_in = 1 - mask_hi - mask_lo

        val_grad_out = mask_hi + alpha * mask_lo
        err = (output - input_) * np.reciprocal(scale)
        grad_scale = grad_output * (err * mask_in + val_grad_out)
        grad_scale = grad_scale.sum()

        grad_input = grad_output * mask_in

        return [grad_input, grad_scale]


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_equal(test, reference):
    for i, (x, y) in enumerate(zip(test, reference)):
        y = y.cpu().detach().numpy()
        np.testing.assert_allclose(x, y, rtol=1e-4, err_msg="Index: {}".format(i))


def get_range_level(is_signed, is_weights, bits):
    if is_signed:
        level_high = 2 ** (bits - 1) - 1
        level_low = -(level_high + 1)
        if is_weights:
            level_low += 1
    else:
        level_high = 2 ** bits - 1
        level_low = 0
    return level_low, level_high

def idfn(val):
    if isinstance(val, list):
        return '[{}]'.format('-'.join([str(v) for v in val]))

    return None

@pytest.mark.parametrize("is_signed", (True, False), ids=('signed', 'unsigned'))
@pytest.mark.parametrize("is_weights", (True, False), ids=('weights', 'activation'))
@pytest.mark.parametrize('input_size',
                         [[1, 96, 112, 112],
                          [1, 192, 28, 28],
                          [1, 576, 14, 14],
                          [32, 96, 112, 112],
                          [32, 192, 28, 28],
                          [32, 576, 14, 14]],
                         ids=idfn)
@pytest.mark.parametrize('bits', (8, 4), ids=('8bit', '4bit'))
@pytest.mark.parametrize("use_cuda", (True, False), ids=('cuda', 'cpu'))
class TestParametrized:
    def test_quantize_symmetric_forward(self, is_signed, is_weights, input_size, bits, use_cuda):
        np.random.seed(0)
        ref_input = 2 * np.random.random_sample(input_size) - 1
        ref_scale = np.array([min(abs(ref_input.min()), abs(ref_input.max()))])

        test_input = torch.from_numpy(ref_input.copy())
        test_scale = torch.from_numpy(ref_scale.copy())

        if use_cuda:
            test_input = test_input.cuda()
            test_scale = test_scale.cuda()

        level_low, level_high = get_range_level(is_signed, is_weights, bits)

        ref_value = ReferenceQuantizeSymmetric.forward(ref_input, ref_scale, level_low, level_high)
        test_value = QuantizeSymmetric.apply(test_input, test_scale, level_low, level_high)

        check_equal(ref_value, test_value)

    def test_quantize_symmetric_backward(self, is_signed, is_weights, input_size, bits, use_cuda):
        np.random.seed(0)
        ref_input = 2 * np.random.random_sample(input_size) - 1
        ref_scale = np.array([min(abs(ref_input.min()), abs(ref_input.max()))])

        level_low, level_high = get_range_level(is_signed, is_weights, bits)

        ref_output = ReferenceQuantizeSymmetric.forward(ref_input, ref_scale, level_low, level_high)
        ref_grads = ReferenceQuantizeSymmetric.backward(
            np.ones(input_size), ref_input, ref_scale, level_low, level_high, ref_output
        )

        test_input = torch.from_numpy(ref_input.copy())
        test_scale = torch.from_numpy(ref_scale.copy())

        if use_cuda:
            test_input = test_input.cuda()
            test_scale = test_scale.cuda()

        test_input = Variable(test_input, requires_grad=True)
        test_scale = Variable(test_scale, requires_grad=True)

        test_value = QuantizeSymmetric.apply(test_input, test_scale, level_low, level_high)
        test_value.sum().backward()
        test_grads = get_grads([test_input, test_scale])

        check_equal(ref_grads, test_grads)
