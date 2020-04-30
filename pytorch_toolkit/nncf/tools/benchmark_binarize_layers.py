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
import torch.nn as nn

from nncf.binarization.layers import XNORBinarize, DOREFABinarize, ActivationBinarizationScaleThreshold
from tools.benchmark import run_profile
from nncf.utils import get_per_channel_scale_shape

NBITS = 8
GPU_RUNS_LOW_BATCH = 10000
GPU_RUNS_HIGH_BATCH = 100
CPU_RUNS = 100
LOW_BATCH_INPUT_SIZE = [1, 96, 112, 112]
HIGH_BATCH_INPUT_SIZE = [128, 96, 112, 112]
TEST_PARAMS_STRUCT = [("low batch", LOW_BATCH_INPUT_SIZE, GPU_RUNS_LOW_BATCH),
                      ("high batch", HIGH_BATCH_INPUT_SIZE, GPU_RUNS_HIGH_BATCH)]


# reference impl
class ReferenceXNORBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        norm = x.abs().mean([1, 2, 3], keepdim=True)
        sign = ((x > 0).type(x.dtype) * 2 - 1)
        output = sign * norm
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ReferenceDOREFABinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        norm = x.abs().mean()
        sign = ((x > 0).type(x.dtype) * 2 - 1)
        output_flat = sign * norm
        return output_flat.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ReferenceActivationBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scale, threshold):
        shape = [1 for s in input_.shape]
        shape[1] = input_.shape[1]
        t = (threshold*scale).view(shape)
        output = (input_ > t).type(input_.dtype) * scale
        ctx.save_for_backward(input_, scale, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale, output = ctx.saved_variables

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


class ReferenceWeightBinarizationModule(nn.Module):
    def __init__(self, mode='xnor'):
        super().__init__()
        self.mode = mode
        if self.mode == 'xnor':
            self.binarize = ReferenceXNORBinarize.apply
        elif self.mode == 'dorefa':
            self.binarize = ReferenceDOREFABinarize.apply

    def forward(self, input_):
        return self.binarize(input_)


def get_test_scale(num_channels):
    torch.manual_seed(0)
    retval = torch.Tensor(num_channels)
    retval.random_(0, 1)
    return retval


def get_test_threshold(input_shape):
    torch.manual_seed(0)
    threshold_shape = get_per_channel_scale_shape(input_shape, is_weights=False)
    retval = torch.Tensor(torch.zeros(threshold_shape))
    retval.random_(-10, 10)
    return retval


class ReferenceActivationBinarizationModule(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.scale = torch.nn.Parameter(get_test_scale(num_channels=1))
        self.threshold = torch.nn.Parameter(get_test_threshold(input_shape))

    def forward(self, input_):
        return ReferenceActivationBinarize.apply(input_, self.scale, self.threshold)


if __name__ == '__main__':
    for input_name, input_size, gpu_runs in TEST_PARAMS_STRUCT:
        print()
        print("CUDA " + input_name)
        print("------------------------------------------------")
        print("Pytorch XNOR weight binarization (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            ReferenceWeightBinarizationModule('xnor').cuda(),
            input_size,
            'cuda',
            gpu_runs,
            forward_only=True)

        print()
        print("Custom XNOR weight binarization (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            XNORBinarize(enabled=True).cuda(),
            input_size,
            'cuda',
            gpu_runs,
            forward_only=True)

        print()
        print("Pytorch DoReFa weight binarization (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            ReferenceWeightBinarizationModule('dorefa').cuda(),
            input_size,
            'cuda',
            gpu_runs,
            forward_only=True)

        print()
        print("Custom DoReFa weight binarization (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            DOREFABinarize(enabled=True).cuda(),
            input_size,
            'cuda',
            gpu_runs,
            forward_only=True)

        print()
        print("Pytorch scale/threshold activation binarization (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            ReferenceActivationBinarizationModule(input_shape=LOW_BATCH_INPUT_SIZE).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Custom scale/threshold activation binarization (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        # Init the scales now so that it does not affect the benchmark
        act_bin_module = ActivationBinarizationScaleThreshold(input_shape=LOW_BATCH_INPUT_SIZE, enabled=True)
        act_bin_module.scale = torch.nn.Parameter(get_test_scale(1))
        act_bin_module.threshold = torch.nn.Parameter(get_test_threshold(LOW_BATCH_INPUT_SIZE))
        act_bin_module.is_scale_initialized = True
        run_profile(
            act_bin_module.cuda(),
            input_size,
            'cuda',
            gpu_runs)


    # CPU low batch
    print()
    print("CPU low batch")
    print("------------------------------------------------")
    print("Pytorch XNOR weight binarization (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        ReferenceWeightBinarizationModule('xnor'),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS,
        forward_only=True)

    print()
    print("Custom XNOR weight binarization (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        XNORBinarize(enabled=True),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS,
        forward_only=True)

    print()
    print("Pytorch DoReFa weight binarization (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        ReferenceWeightBinarizationModule('dorefa'),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS,
        forward_only=True)

    print()
    print("Custom DoReFa weight binarization (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        DOREFABinarize(enabled=True),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS,
        forward_only=True)

    print()
    print("Pytorch scale/threshold activation binarization (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        ReferenceActivationBinarizationModule(input_shape=LOW_BATCH_INPUT_SIZE),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Custom scale/threshold activation binarization (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    # Init the scales now so that it does not affect the benchmark
    act_bin_module = ActivationBinarizationScaleThreshold(input_shape=LOW_BATCH_INPUT_SIZE, enabled=True)
    act_bin_module.scale = torch.nn.Parameter(get_test_scale(1))
    act_bin_module.threshold = torch.nn.Parameter(get_test_threshold(LOW_BATCH_INPUT_SIZE))
    act_bin_module.is_scale_initialized = True
    run_profile(
        act_bin_module,
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)
