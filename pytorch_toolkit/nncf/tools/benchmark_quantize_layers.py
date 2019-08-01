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
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from nncf.quantization import Quantize

TIME_SCALES = {'ms': 1000}
INPUT_SIZE = [[1, 96, 112, 112], [128, 96, 112, 112]]
NBITS = 8
GPU_RUNS_LOW_BATCH = 10000
GPU_RUNS_HIGH_BATCH = 100
CPU_RUNS = 100


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
        grad_scale = grad_scale.sum().view_as(scale)

        # calc gradient for input
        grad_input = grad_output * mask_in

        return grad_input, grad_scale, None


class ReferenceQuantize(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.num_bits = num_bits
        self.level_high = 2 ** (self.num_bits - 1) - 1
        self.level_low = -(self.level_high + 1)
        self.quantize = ReferenceQuantizeSymmetric.apply

    def get_scale(self):
        return self.scale

    def forward(self, weight):
        return self.quantize(weight, self.scale, self.num_bits)


def warmup(layer, input_, runs):
    for _ in range(runs):
        new_i = layer(input_)
        new_i[0].sum().backward()


def run_wall(layer, input_size, device, runs, is_print=True):
    input_ = torch.randn(input_size, device=torch.device(device))

    # Force CUDA initialization & warm up
    warmup(layer, input_, 100)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        layer.zero_grad()
        new_i = layer(input_)
        new_i[0].sum().backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    ctime, scale = list(TIME_SCALES.items())[0]
    fbtime = elapsed / runs * scale

    if is_print:
        print('Forward&Backward: {0:.3f} {1}'.format(
            fbtime, ctime))


def run_profile(layer, input_size, device, runs):
    input_ = torch.randn(input_size, device=torch.device(device))

    # Force CUDA initialization & warm up
    warmup(layer, input_, 100)

    start = time.time()
    forward_min = math.inf
    forward_time = 0
    backward_min = math.inf
    backward_time = 0
    for _ in range(runs):
        layer.zero_grad()

        torch.cuda.synchronize()
        start = time.time()
        new_i = layer(input_)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

        torch.cuda.synchronize()
        start = time.time()
        new_i[0].sum().backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        backward_min = min(backward_min, elapsed)
        backward_time += elapsed

    ctime, scale = list(TIME_SCALES.items())[0]
    forward_min *= scale
    backward_min *= scale
    forward_average = forward_time / runs * scale
    backward_average = backward_time / runs * scale

    print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
        forward_min, forward_average, backward_min, backward_average, ctime))


def run_worker(gpu, world_size, layer, input_size, runs):
    dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:8899",
                            world_size=world_size, rank=gpu)

    device = torch.device('cuda:%d' % gpu)
    torch.cuda.set_device(gpu)

    batch = (int)(input_size[0] / world_size)
    if gpu == 0:
        run_size = input_size.copy()
        run_size[0] = input_size[0] - batch * (world_size - 1)
    else:
        run_size = input_size.copy()
        run_size[0] = batch

    run_model = layer.to(device)
    run_model = torch.nn.parallel.DistributedDataParallel(run_model, device_ids=[gpu])

    run_wall(run_model, run_size, device, runs, (gpu == 0))


if __name__ == '__main__':
    # CUDA low batch
    print("CUDA low batch")
    print("------------------------------------------------")
    print("Pytorch (cuda 0) impl:")
    print("input size: {0}".format(INPUT_SIZE[0]))
    run_profile(
        ReferenceQuantize(NBITS).cuda(),
        INPUT_SIZE[0],
        'cuda',
        GPU_RUNS_LOW_BATCH)

    print("Custom (cuda 0 ) impl:")
    print("input size: {0}".format(INPUT_SIZE[0]))
    run_profile(
        Quantize(num_bits=NBITS).cuda(),
        INPUT_SIZE[0],
        'cuda',
        GPU_RUNS_LOW_BATCH)

    # CUDA high batch
    print()
    print("CUDA high batch")
    print("------------------------------------------------")
    print("Pytorch (cuda 0) impl:")
    print("input size: {0}".format(INPUT_SIZE[1]))
    run_profile(
        ReferenceQuantize(NBITS).cuda(),
        INPUT_SIZE[1],
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    print("Custom (cuda 0 ) impl:")
    print("input size: {0}".format(INPUT_SIZE[1]))
    run_profile(
        Quantize(num_bits=NBITS).cuda(),
        INPUT_SIZE[1],
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    # CUDA DataParallel high batch
    device_ids = [x for x in range(torch.cuda.device_count())]
    print()
    print("CUDA DataParallel high batch")
    print("------------------------------------------------")
    print("Pytorch (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(INPUT_SIZE[1]))
    run_profile(
        nn.parallel.DataParallel(ReferenceQuantize(NBITS).cuda(), device_ids=device_ids),
        INPUT_SIZE[1],
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    print("Custom (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(INPUT_SIZE[1]))
    run_profile(
        nn.parallel.DataParallel(Quantize(NBITS).cuda(), device_ids=device_ids),
        INPUT_SIZE[1],
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    # CUDA DataParallel high batch
    # wall time
    print()
    print("CUDA DataParallel high batch")
    print("------------------------------------------------")
    print("Pytorch (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(INPUT_SIZE[1]))
    run_wall(
        nn.parallel.DataParallel(ReferenceQuantize(NBITS).cuda(), device_ids=device_ids),
        INPUT_SIZE[1],
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    print("Custom (cuda {0}) DataParallel impl:".format(device_ids))
    print("input size: {0}".format(INPUT_SIZE[1]))
    run_wall(
        nn.parallel.DataParallel(Quantize(num_bits=NBITS).cuda(), device_ids=device_ids),
        INPUT_SIZE[1],
        'cuda',
        GPU_RUNS_HIGH_BATCH)

    # CUDA DistributedDataParallel high batch
    # wall time
    NGPUS_PER_NODE = len(device_ids)
    WORLD_SIZE = NGPUS_PER_NODE
    print()
    print("CUDA DistributedDataParallel high batch")
    print("------------------------------------------------")
    print("Pytorch (cuda {0}) DistributedDataParallel impl:".format(device_ids))
    print("input size: {0}".format(INPUT_SIZE[1]))
    mp.spawn(
        run_worker,
        nprocs=NGPUS_PER_NODE,
        args=(WORLD_SIZE, ReferenceQuantize(NBITS), INPUT_SIZE[1], GPU_RUNS_HIGH_BATCH))
    print("Custom (cuda {0}) DistributedDataParallel impl:".format(device_ids))
    print("input size: {0}".format(INPUT_SIZE[1]))
    mp.spawn(
        run_worker,
        nprocs=NGPUS_PER_NODE,
        args=(WORLD_SIZE, Quantize(num_bits=NBITS), INPUT_SIZE[1], GPU_RUNS_HIGH_BATCH))

    # CPU low batch
    print()
    print("CPU low batch")
    print("------------------------------------------------")
    print("Pytorch (cpu) impl:")
    print("input size: {0}".format(INPUT_SIZE[0]))
    run_profile(
        ReferenceQuantize(NBITS),
        INPUT_SIZE[0],
        'cpu',
        CPU_RUNS)

    print("Custom (cpu) impl:")
    print("input size: {0}".format(INPUT_SIZE[0]))
    run_profile(
        Quantize(num_bits=NBITS),
        INPUT_SIZE[0],
        'cpu',
        CPU_RUNS)
