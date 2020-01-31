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

from nncf.binarization.layers import xnor_binarize_op, dorefa_binarize_op, activation_bin_scale_threshold_op
from tests.test_helpers import check_equal, get_grads


# reference impl
class ReferenceXNORBinarize:
    @staticmethod
    def forward(x):
        norm = np.abs(x).mean((1, 2, 3), keepdims=True)
        sign = ((x > 0).astype(x.dtype) * 2 - 1)
        output = sign * norm
        return output

    @staticmethod
    def backward(grad_output):
        return grad_output


class ReferenceDOREFABinarize:
    @staticmethod
    def forward(x):
        norm = np.abs(x).mean()
        sign = ((x > 0).astype(x.dtype) * 2 - 1)
        return sign * norm

    @staticmethod
    def backward(grad_output):
        return grad_output


class ReferenceActivationBinarize:
    @staticmethod
    def forward(x, scale, threshold):
        shape = [1 for s in x.shape]
        shape[1] = x.shape[1]
        t = threshold * scale
        output = (x > t).astype(x.dtype) * scale
        return output

    @staticmethod
    def backward(grad_output, x, scale, output):

        # calc gradient for input
        mask_lower = (x <= scale).astype(x.dtype)
        grad_input = grad_output * (x >= 0).astype(x.dtype) * mask_lower

        # calc gradient for scale
        err = (output - x) / scale
        grad_scale = grad_output * (mask_lower * err + (1 - mask_lower))
        grad_scale = grad_scale.sum()

        # calc gradient for threshold
        grad_threshold = -grad_output * (x > 0).astype(x.dtype) * (x < scale).astype(x.dtype)

        for idx, _ in enumerate(x.shape):
            if idx != 1:  # activation channel dimension
                grad_threshold = grad_threshold.sum(idx, keepdims=True)

        return [grad_input, grad_scale, grad_threshold]


def idfn(val):
    if isinstance(val, list):
        return '[{}]'.format('-'.join([str(v) for v in val]))

    return None


@pytest.fixture
def _seed():
    np.random.seed(0)


def generate_input(input_size):
    return (2 * np.random.random_sample(input_size) - 1) * np.random.rand() + np.random.rand()


def generate_scale_threshold(input_size):
    threshold_shape = [1 for s in input_size]
    threshold_shape[1] = input_size[1]
    return np.random.random_sample(1), (2 * np.random.random_sample(threshold_shape) - 1)


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
@pytest.mark.parametrize("use_cuda", [False, True], ids=['cpu', 'cuda'])
class TestParametrized:
    @pytest.mark.parametrize('weight_bin_type', ["xnor", "dorefa"])
    class TestWeightBinarization:
        def test_binarize_weights_forward(self, _seed, input_size, weight_bin_type, use_cuda):
            ref_input = generate_input(input_size)

            test_input = get_test_data([ref_input], use_cuda)[0]

            if weight_bin_type == "xnor":
                ref_value = ReferenceXNORBinarize.forward(ref_input)
                test_value = xnor_binarize_op(test_input)
            elif weight_bin_type == "dorefa":
                ref_value = ReferenceDOREFABinarize.forward(ref_input)
                test_value = dorefa_binarize_op(test_input)

            check_equal(ref_value, test_value, rtol=1e-3)

    def test_binarize_activations_forward(self, _seed, input_size, use_cuda):
        ref_input = generate_input(input_size)
        ref_scale, ref_threshold = generate_scale_threshold(input_size)

        test_input, test_scale, test_threshold = get_test_data([ref_input, ref_scale, ref_threshold], use_cuda)

        ref_value = ReferenceActivationBinarize.forward(ref_input, ref_scale, ref_threshold)
        test_value = activation_bin_scale_threshold_op(test_input, test_scale, test_threshold)

        check_equal(ref_value, test_value, rtol=1e-3)

    def test_binarize_activations_backward(self, _seed, input_size, use_cuda):
        ref_input = generate_input(input_size)
        ref_scale, ref_threshold = generate_scale_threshold(input_size)

        test_input, test_scale, test_threshold = get_test_data([ref_input, ref_scale, ref_threshold], use_cuda,
                                                               is_backward=True)

        ref_value = ReferenceActivationBinarize.forward(ref_input, ref_scale, ref_threshold)
        ref_grads = ReferenceActivationBinarize.backward(np.ones(input_size), ref_input, ref_scale, ref_value)

        test_value = activation_bin_scale_threshold_op(test_input, test_scale, test_threshold)
        test_value.sum().backward()
        test_grads = get_grads([test_input, test_scale, test_threshold])

        check_equal(ref_grads, test_grads, rtol=1e-3)
