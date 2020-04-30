"""
 Copyright (c) 2020 Intel Corporation
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
import torch
import torch.nn as nn

from nncf.layer_utils import COMPRESSION_MODULES
from nncf.utils import is_tracing_state, no_jit_trace


@COMPRESSION_MODULES.register()
class FilterPruningBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("_binary_filter_pruning_mask", torch.ones(size))

    @property
    def binary_filter_pruning_mask(self):
        return self._binary_filter_pruning_mask

    @binary_filter_pruning_mask.setter
    def binary_filter_pruning_mask(self, mask):
        with torch.no_grad():
            self._binary_filter_pruning_mask.set_(mask)

    def forward(self, conv_weight):
        if is_tracing_state():
            with no_jit_trace():
                return conv_weight
        return conv_weight


def broadcast_filter_mask(filter_mask, shape):
    broadcasted_shape = np.ones(len(shape), dtype=np.int64)
    broadcasted_shape[0] = filter_mask.size(0)
    broadcasted_filter_mask = torch.reshape(filter_mask, tuple(broadcasted_shape))
    return broadcasted_filter_mask


def inplace_apply_filter_binary_mask(filter_mask, conv_weight, module_name=""):
    """
    Inplace applying binary filter mask to weight (or bias) of the convolution
    (by first dim of the conv weight).
    :param filter_mask: binary mask (should have the same shape as first dim of conv weight)
    :param conv_weight: weight or bias of convolution
    :return: result with applied mask
    """
    if filter_mask.size(0) != conv_weight.size(0):
        raise RuntimeError("Shape of mask = {} for module {} isn't broadcastable to weight shape={}."
                           " ".format(filter_mask.shape, module_name, conv_weight.shape))
    broadcasted_filter_mask = broadcast_filter_mask(filter_mask, conv_weight.shape)
    return conv_weight.mul_(broadcasted_filter_mask)


def apply_filter_binary_mask(filter_mask, conv_weight, module_name=""):
    """
    Applying binary filter mask to weight (or bias) of the convolution (applying by first dim of the conv weight)
    without changing the weight.
    :param filter_mask: binary mask (should have the same shape as first dim of conv weight)
    :param conv_weight: weight or bias of convolution
    :return: result with applied mask
    """
    if filter_mask.size(0) != conv_weight.size(0):
        raise RuntimeError("Shape of mask = {} for module {} isn't broadcastable to weight shape={}."
                           " ".format(filter_mask.shape, module_name, conv_weight.shape))
    broadcasted_filter_mask = broadcast_filter_mask(filter_mask, conv_weight.shape)
    return broadcasted_filter_mask * conv_weight
