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

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel._functions import Scatter


class ScatterShallow(object):

    @staticmethod
    def recursive_apply(target_gpus, dim, input):
        if isinstance(input, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, input)[0]
        elif isinstance(input, list):
            return [ScatterShallow.recursive_apply(target_gpus, dim, i) for i in input]
        elif isinstance(input, tuple):
            return (ScatterShallow.recursive_apply(target_gpus, dim, i) for i in input)
        elif isinstance(input, dict):
            return {k: ScatterShallow.recursive_apply(target_gpus, dim, v) for k, v in input.items()}
        return input

    @staticmethod
    def apply(target_gpus, dim, input):
        # Output is list with size = number of GPUs.
        output = [[] for _ in target_gpus]

        chunk_size = len(input) // len(target_gpus)
        chunk_indexes = np.full(len(target_gpus) + 1, 0, dtype=int)  # [0, 0, 0...]
        chunk_indexes[1:] = chunk_size  # [0, n, n...]
        chunk_indexes[1:len(input) % len(target_gpus) + 1] += 1  # [0, n + 1, n + 1, n, n...]
        chunk_indexes = np.cumsum(chunk_indexes)  # [0, n1, n2, n3...]

        chunks = []
        for i, j in zip(chunk_indexes[:-1], chunk_indexes[1:]):
            chunks.append(input[i:j])

        used_gpus = 0
        for gpu_id, chunk in enumerate(chunks):
            target_gpu = target_gpus[gpu_id]
            # Calculate number of GPUs which have got data
            if len(chunk) > 0:
                used_gpus += 1
            output[gpu_id] = ScatterShallow.recursive_apply([target_gpu], dim, chunk)

        return output[:used_gpus]


def scatter(inputs, target_gpus, dim=0):
    """Reimplemented case if object is list, case for dict removed.
    Other cases are the same as in the base class
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            result = Scatter.apply(target_gpus, None, dim, obj)
            return result
        if isinstance(obj, list) and len(obj) > 0:
            result = ScatterShallow.apply(target_gpus, dim, obj)
            return result
        # `inputs` is either a tuple for positional arguments or a dict for keyword arguments,
        # so just recursively go deeper.
        if isinstance(obj, tuple) and len(obj) > 0:
            result = list(zip(*map(scatter_map, obj)))
            return result
        if isinstance(obj, dict) and len(obj) > 0:
            keys_and_values = list(zip(*map(scatter_map, obj.items())))
            result = list(map(type(obj), keys_and_values))
            return result
        return [obj for targets in target_gpus]

    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """This function is the same as in the base class `nn.DataParallel`
    except using of reimplemented function `scatter`
    """
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class ShallowDataParallel(nn.DataParallel):
    """Define custom method scatter
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, self.dim)


def collate(samples):
    batch = defaultdict(list)
    for sample in samples:
        for k, v in sample.items():
            batch[k].append(v)
    batch['batch_idx'] = list(range(len(samples)))
    return batch
