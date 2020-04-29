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
import torch.multiprocessing as mp
import torch.nn as nn

from nncf.quantization.layers import QuantizerConfig, AsymmetricQuantizer, SymmetricQuantizer
from nncf.utils import sum_like, get_per_channel_scale_shape

from tools.benchmark import run_profile, run_wall, run_worker

TIME_SCALES = {'ms': 1000}
NBITS = 8
GPU_RUNS_LOW_BATCH = 10000
GPU_RUNS_HIGH_BATCH = 100
CPU_RUNS = 100
LOW_BATCH_INPUT_SIZE = [1, 96, 112, 112]
HIGH_BATCH_INPUT_SIZE = [128, 96, 112, 112]
TEST_PARAMS_STRUCT = [("low batch", LOW_BATCH_INPUT_SIZE, GPU_RUNS_LOW_BATCH),
                      ("high batch", HIGH_BATCH_INPUT_SIZE, GPU_RUNS_HIGH_BATCH)]


# reference impl
class ReferenceQuantizeSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scale, bits):
        level_high = scale.new_tensor([2 ** (bits - 1) - 1])
        level_low = scale.new_tensor([-(level_high + 1)])
        s = level_high / scale

        output = input_ * s
        output = output.clamp(min=level_low[0], max=level_high[0])
        output = output.round()
        output = output / s

        ctx.save_for_backward(input_, scale, output)
        ctx.level_high = level_high
        ctx.level_low = level_low

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale, output = ctx.saved_tensors
        level_high = ctx.level_high
        level_low = ctx.level_low

        alpha = float(level_low) / float(level_high)
        mask_hi = (input_ > scale).type(input_.dtype)
        mask_lo = (input_ < scale * alpha).type(input_.dtype)
        mask_in = 1 - mask_hi - mask_lo

        val_grad_out = mask_hi + alpha * mask_lo
        err = (output - input_) * scale.reciprocal()
        grad_scale = grad_output * (err * mask_in + val_grad_out)
        grad_scale = sum_like(grad_scale, scale)

        # calc gradient for input
        grad_input = grad_output * mask_in

        return grad_input, grad_scale, None


class ReferenceQuantize(nn.Module):
    def __init__(self, num_bits=8, input_shape=None, is_weights=True, per_channel=False):
        super().__init__()
        self.input_shape = input_shape
        self.is_weights = is_weights
        scale_shape = 1
        if per_channel:
            scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)

        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_bits = num_bits
        self.level_high = 2 ** (self.num_bits - 1) - 1
        self.level_low = -(self.level_high + 1)
        self.quantize = ReferenceQuantizeSymmetric.apply

    def get_scale(self):
        return self.scale

    def forward(self, input_):
        return self.quantize(input_, self.scale, self.num_bits)



if __name__ == '__main__':
    for input_name, input_size, gpu_runs in TEST_PARAMS_STRUCT:
        print("CUDA " + input_name)
        print("------------------------------------------------")
        print("Pytorch Symmetric (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            ReferenceQuantize(NBITS).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Custom Symmetric (cuda 0 ) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            SymmetricQuantizer(QuantizerConfig(bits=NBITS)).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Pytorch Symmetric Per Weight Channel (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            ReferenceQuantize(NBITS,
                              input_shape=input_size,
                              per_channel=True,
                              is_weights=True).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Custom Symmetric Per Weight Channel  (cuda 0 ) impl")
        print("input size: {0}".format(input_size))
        run_profile(
            SymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                               input_shape=input_size,
                                               per_channel=True,
                                               is_weights=True)).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Pytorch Symmetric Per Activation Channel (cuda 0) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            ReferenceQuantize(NBITS,
                              input_shape=input_size,
                              per_channel=True,
                              is_weights=False).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Custom Symmetric Per Activation Channel  (cuda 0 ) impl")
        print("input size: {0}".format(input_size))
        run_profile(
            SymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                               input_shape=input_size,
                                               per_channel=True,
                                               is_weights=False)).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Custom Asymmetric (cuda 0 ) impl:")
        print("input size: {0}".format(input_size))
        run_profile(
            AsymmetricQuantizer(QuantizerConfig(bits=NBITS)).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Custom Asymmetric Per Weight Channel  (cuda 0 ) impl")
        print("input size: {0}".format(input_size))
        run_profile(
            AsymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                                input_shape=input_size,
                                                per_channel=True,
                                                is_weights=True)).cuda(),
            input_size,
            'cuda',
            gpu_runs)

        print()
        print("Custom Asymmetric Per Activation Channel  (cuda 0 ) impl")
        print("input size: {0}".format(input_size))
        run_profile(
            AsymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                                input_shape=input_size,
                                                per_channel=True,
                                                is_weights=False)).cuda(),
            input_size,
            'cuda',
            gpu_runs)


    # CPU low batch
    print()
    print("CPU low batch")
    print("------------------------------------------------")
    print("Pytorch Symmetric(cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        ReferenceQuantize(NBITS),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Custom Symmetric (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        SymmetricQuantizer(QuantizerConfig(bits=NBITS)),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Pytorch Symmetric Per Weight Channel (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        ReferenceQuantize(NBITS,
                          input_shape=LOW_BATCH_INPUT_SIZE,
                          per_channel=True,
                          is_weights=True),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Custom Symmetric Per Weight Channel  (cpu) impl")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        SymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                           input_shape=LOW_BATCH_INPUT_SIZE,
                                           per_channel=True,
                                           is_weights=True)),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Pytorch Symmetric Per Activation Channel (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        ReferenceQuantize(NBITS,
                          input_shape=LOW_BATCH_INPUT_SIZE,
                          per_channel=True,
                          is_weights=False),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Custom Symmetric Per Activation Channel  (cpu) impl")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        SymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                           input_shape=LOW_BATCH_INPUT_SIZE,
                                           per_channel=True,
                                           is_weights=False)),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Custom Asymmetric (cpu) impl:")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        AsymmetricQuantizer(QuantizerConfig(bits=NBITS)),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Custom Asymmetric Per Weight Channel  (cpu) impl")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        AsymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                            input_shape=LOW_BATCH_INPUT_SIZE,
                                            per_channel=True,
                                            is_weights=True)),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)

    print()
    print("Custom Asymmetric Per Activation Channel  (cpu) impl")
    print("input size: {0}".format(LOW_BATCH_INPUT_SIZE))
    run_profile(
        AsymmetricQuantizer(QuantizerConfig(bits=NBITS,
                                            input_shape=LOW_BATCH_INPUT_SIZE,
                                            per_channel=True,
                                            is_weights=False)),
        LOW_BATCH_INPUT_SIZE,
        'cpu',
        CPU_RUNS)


    # CUDA DataParallel high batch
    device_ids = range(torch.cuda.device_count())
    print()
    print("CUDA DataParallel high batch")
    print("------------------------------------------------")
    print("Pytorch Symmetric(cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    run_profile(
        nn.parallel.DataParallel(ReferenceQuantize(NBITS).cuda(), device_ids=device_ids),
        HIGH_BATCH_INPUT_SIZE,
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    print()
    print("Custom Symmetric (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    run_profile(
        nn.parallel.DataParallel(SymmetricQuantizer(QuantizerConfig(bits=NBITS)).cuda(),
                                 device_ids=device_ids),
        HIGH_BATCH_INPUT_SIZE,
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    print()
    print("Custom Asymmetric (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    run_profile(
        nn.parallel.DataParallel(AsymmetricQuantizer(QuantizerConfig(bits=NBITS)).cuda(),
                                 device_ids=device_ids),
        HIGH_BATCH_INPUT_SIZE,
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    # CUDA DataParallel high batch
    # wall time
    print()
    print("CUDA DataParallel high batch")
    print("------------------------------------------------")
    print("Pytorch Symmetric(cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    run_wall(
        nn.parallel.DataParallel(ReferenceQuantize(NBITS).cuda(), device_ids=device_ids),
        HIGH_BATCH_INPUT_SIZE,
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    print()
    print("Custom Symmetric (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    run_wall(
        nn.parallel.DataParallel(SymmetricQuantizer(QuantizerConfig(bits=NBITS)).cuda(),
                                 device_ids=device_ids),
        HIGH_BATCH_INPUT_SIZE,
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    print()
    print("Custom Assymetric (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    run_wall(
        nn.parallel.DataParallel(AsymmetricQuantizer(QuantizerConfig(bits=NBITS)).cuda(),
                                 device_ids=device_ids),
        HIGH_BATCH_INPUT_SIZE,
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    # CUDA DistributedDataParallel high batch
    # wall time
    NGPUS_PER_NODE = len(device_ids)
    WORLD_SIZE = NGPUS_PER_NODE
    print()
    print("CUDA DistributedDataParallel high batch")
    print("------------------------------------------------")
    print("Pytorch Symmetric(cuda {0}) DistributedDataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    mp.spawn(
        run_worker,
        nprocs=NGPUS_PER_NODE,
        args=(WORLD_SIZE, ReferenceQuantize(NBITS), TEST_PARAMS_STRUCT[1], GPU_RUNS_HIGH_BATCH))

    print()
    print("Custom Symmetric (cuda {0}) DistributedDataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    mp.spawn(
        run_worker,
        nprocs=NGPUS_PER_NODE,
        args=(WORLD_SIZE, SymmetricQuantizer(QuantizerConfig(bits=NBITS)), TEST_PARAMS_STRUCT[1],
              GPU_RUNS_HIGH_BATCH))

    print()
    print("Custom Asymmetric (cuda {0}) DistributedDataParallel impl:".format(device_ids))
    print("input size: {0}".format(HIGH_BATCH_INPUT_SIZE))
    mp.spawn(
        run_worker,
        nprocs=NGPUS_PER_NODE,
        args=(WORLD_SIZE, SymmetricQuantizer(QuantizerConfig(bits=NBITS)), TEST_PARAMS_STRUCT[1],
              GPU_RUNS_HIGH_BATCH))
