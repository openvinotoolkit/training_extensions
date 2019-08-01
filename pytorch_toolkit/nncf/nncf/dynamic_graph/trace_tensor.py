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
    def __init__(self, creator_id, index, shape):
        self.creator_id = creator_id
        self.index = index
        self.shape = tuple(shape)

    def __eq__(self, other):
        if not isinstance(other, TensorMeta):
            return False
        return self.index == other.index and self.creator_id == other.creator_id and self.shape[1:] == other.shape[1:]


class TracedTensor(torch.Tensor):
    # pylint: disable=abstract-method
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_meta = None

    @staticmethod
    def from_torch_tensor(tensor, tensor_meta):
        tensor.tensor_meta = tensor_meta
        tensor.__class__ = TracedTensor
        return tensor


def get_caller_context(operator_type, ctx):
    """
    Designed to work in the following way - for each scope the context will
    track the number of the calls to the operators with the name operator_type
    and return this number as the caller context. The counter values are preserved
    until reset by a corresponding member function of the context, which must be
    called after each model iteration - this is usually handled inside NNCF.
    This mechanism allows to discern between multiple function calls inside the
    same module that would each require their own instance of compression layers
    - for instance, multiple `relu` function calls (either on their own or inside a
    `for` cycle), and at the same moment allow the checkpoints to be loaded if the
    model had changed in the meantime in a way that does not impact the major function
    call order (e.g. if comments were added to the .py file with the model)
    """
    return str(ctx.get_operator_call_count_in_scope(operator_type, ctx.scopes))


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


def trace_tensors(operator_output, node):
    if isinstance(operator_output, (list, tuple)):
        output_ = (
            TracedTensor.from_torch_tensor(x, make_tensor_meta(node, i, x.shape))
            for i, x in enumerate(operator_output)
        )
        return operator_output.__class__(output_)
    if isinstance(operator_output, torch.Tensor):
        return TracedTensor.from_torch_tensor(
            operator_output, make_tensor_meta(node, 0, operator_output.shape)
        )
    raise ValueError("Unknown return type. Can not trace function call")


def make_input_infos(inputs):
    input_infos = []
    for i, node_input in enumerate(inputs):
        if isinstance(node_input, TracedTensor):
            input_infos.append(node_input.tensor_meta)
        elif isinstance(node_input, torch.Tensor) and not isinstance(node_input, TracedTensor):
            input_infos.append(TensorMeta(None, i, node_input.shape))
        else:
            input_infos.append(node_input)
    return input_infos


def make_tensor_meta(node, output_idx, tensor_shpae):
    if output_idx not in node.setdefault('outputs', {}):
        node['outputs'][output_idx] = TensorMeta(node['id'], output_idx, tensor_shpae)
    return node['outputs'][output_idx]


def _has_same_type(obj1, obj2):
    return isinstance(obj1, obj2.__class__) and isinstance(obj2, obj1.__class__)


def _single_input_match(saved_input, actual_input):
    if isinstance(saved_input, TensorMeta):
        # both input and actual are traceable tensors
        if isinstance(actual_input, TracedTensor):
            return actual_input.tensor_meta == saved_input
        # actual input is not traceable tensor => assume it is an input node
        return saved_input.creator_id is None

    if not _has_same_type(saved_input, actual_input):
        return False

    if isinstance(actual_input, np.ndarray):
        return actual_input.dtype == saved_input.dtype and actual_input.shape == saved_input.shape

    # input and saved are plain types

    # Scalar arguments change when switching train / eval
    # return saved_input == actual_input
    return True


def inputs_match(node_inputs, real_inputs):
    if node_inputs is None and real_inputs:
        return False
    if len(node_inputs) != len(real_inputs):
        return False

    for saved_input, actual_input in zip(node_inputs, real_inputs):
        if not _single_input_match(saved_input, actual_input):
            return False
    return True
