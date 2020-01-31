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

from typing import Iterable

import numpy as np
import torch


class TensorMeta:
    @staticmethod
    def default_comparator(lhs: 'TensorMeta', rhs: 'TensorMeta'):
        return lhs.index == rhs.index and lhs.creator_id == rhs.creator_id and lhs.shape[1:] == rhs.shape[1:]

    def __init__(self, creator_id, index, shape):
        self.creator_id = creator_id
        self.index = index
        self.shape = tuple(int(dim) for dim in shape)  # Handle cases when shape is a tuple of Tensors

    def __eq__(self, other):
        if not isinstance(other, TensorMeta):
            return False
        return self.default_comparator(self, other)

    def __hash__(self):
        return hash((self.creator_id, self.index, self.shape))

    def __str__(self):
        return "C{}_I{}_".format(self.creator_id, self.index) + "S" + "x".join([str(s) for s in self.shape])


class TracedTensor(torch.Tensor):
    # pylint: disable=abstract-method
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_meta = None

    @staticmethod
    def from_torch_tensor(tensor, tensor_meta: TensorMeta):
        tensor.tensor_meta = tensor_meta
        tensor.__class__ = TracedTensor
        return tensor


def is_iterable(item):
    non_iterable_types = (str, bytes, bytearray, torch.Tensor, np.ndarray)
    return isinstance(item, Iterable) and not isinstance(item, non_iterable_types)


def flatten(items):
    it = items.items() if hasattr(items, 'items') else iter(items)
    for item in it:
        if is_iterable(item):
            for i in flatten(item):
                yield i
        else:
            yield item


def flatten_args(args, kwargs):
    return list(flatten(args)) + list(flatten(kwargs))


def trace_tensors(operator_output, node: 'NNCFNode'):
    if isinstance(operator_output, (list, tuple)):
        output_ = []
        for i, x in enumerate(operator_output):
            meta = TensorMeta(node.node_id, i, x.shape)
            output_.append(TracedTensor.from_torch_tensor(x, meta))
        return operator_output.__class__(output_)
    if isinstance(operator_output, torch.Tensor):
        meta = TensorMeta(node.node_id, 0, operator_output.shape)
        return TracedTensor.from_torch_tensor(operator_output, meta)
    raise ValueError("Unknown return type. Can not trace function call")


def make_input_infos(inputs):
    input_infos = []
    for i, node_input in enumerate(inputs):
        if isinstance(node_input, TracedTensor):
            input_infos.append(node_input.tensor_meta)
        elif isinstance(node_input, torch.Tensor) and not isinstance(node_input, TracedTensor):
            meta = TensorMeta(None, i, node_input.shape)
            input_infos.append(meta)
        else:
            input_infos.append(None)
    return input_infos
