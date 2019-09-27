import math
import time

import torch
import torch.distributed as dist
import torch.nn as nn

TIME_SCALES = {'ms': 1000}


def warmup(layer, input_, runs, forward_only=False):
    for _ in range(runs):
        new_i = layer(input_)
        if not forward_only:
            new_i[0].sum().backward()


def run_wall(layer, input_size_, device, runs, is_print=True):
    input_ = torch.randn(input_size_, device=torch.device(device))

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


def run_profile(layer, input_size_, device, runs, forward_only=False):
    input_ = torch.randn(input_size_, device=torch.device(device))

    # Force CUDA initialization & warm up
    warmup(layer, input_, 100, forward_only)

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

        if not forward_only:
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

    print('Forward: min {0:.3f}{4} / avg {1:.3f}{4} | Backward: min {2:.3f}{4} / avg {3:.3f}{4}'.format(
        forward_min, forward_average, backward_min, backward_average, ctime))


def run_worker(gpu, world_size, layer, input_size_, runs):
    dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:8899",
                            world_size=world_size, rank=gpu)

    device = torch.device('cuda:%d' % gpu)
    torch.cuda.set_device(gpu)

    batch = (int)(input_size_[0] / world_size)
    if gpu == 0:
        run_size = input_size_.copy()
        run_size[0] = input_size_[0] - batch * (world_size - 1)
    else:
        run_size = input_size_.copy()
        run_size[0] = batch

    run_model = layer.to(device)
    run_model = nn.parallel.DistributedDataParallel(run_model, device_ids=[gpu])

    run_wall(run_model, run_size, device, runs, (gpu == 0))
