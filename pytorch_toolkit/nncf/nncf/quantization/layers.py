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

import logging
from collections import namedtuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import distributed

from .initializers import MIN_MAX_INITIALIZERS
from .quantize_functions import symmetric_quantize, asymmetric_quantize
from ..layer_utils import COMPRESSION_MODULES
from ..registry import Registry
from ..utils import get_per_channel_scale_shape

logger = logging.getLogger(__name__)

QUANTIZATION_MODULES = Registry('quantization_modules')
BINARIZATION_MODULES = Registry('binarization_modules')


class QuantizationMode:
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class BinarizationMode:
    XNOR = "xnor"
    DOREFA = "dorefa"


QuantizationParams = namedtuple(
    'QuantizationParams', ['bits', 'mode', 'signed', 'signed_scope', 'per_channel']
)
QuantizationParams.__new__.__defaults__ = (8, QuantizationMode.SYMMETRIC, False, [], False)


class QuantizerConfig:
    def __init__(self, params: QuantizationParams, input_shape=None, is_weights=False, per_channel=False,
                 within_signed_scope=False):
        self.params = params
        self.is_weights = is_weights
        self.within_signed_scope = within_signed_scope
        self.per_channel = per_channel
        self.input_shape = input_shape


class BaseQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.config = config
        self.init_stage = False
        self.initialized = False
        self.state_dict_name = None

        class LoadStateListener:
            """
               Check whether a quantization module are going to be updated by new values from state_dict or checkpoint.
            """

            def __init__(self, module):
                # pylint: disable=protected-access
                self.hook = module._register_load_state_dict_pre_hook(partial(self.hook_fn, module=module))

            def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                        module):
                if module.state_dict_name:
                    for module_key in module.state_dict().keys():
                        candidate = module.state_dict_name + '.' + module_key
                        if candidate in state_dict:
                            module.initialized = True

            def close(self):
                self.hook.remove()

        self.load_listener = LoadStateListener(self)

    def forward(self, x):
        if self.init_stage:
            return x
        return self.quantize(x)

    def quantize(self, x):
        raise NotImplementedError


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizer(BaseQuantizer):
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config.input_shape
        self.per_channel = config.per_channel
        self.is_weights = config.is_weights
        self.within_signed_scope = config.within_signed_scope
        params = config.params
        self.num_bits = params.bits
        self.signed_tensor = nn.Parameter(torch.IntTensor([params.signed]), requires_grad=False)
        self.collect_scale_statistics = False

        scale_shape = 1
        if self.per_channel:
            scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)
        self.scale = nn.Parameter(torch.ones(scale_shape), requires_grad=True)

        self.init_stage = False
        self.eps = 1e-16

        self.level_high = self.level_low = 0
        self.levels = 2 ** self.num_bits
        if self.is_weights:
            self.levels -= 1

    def set_level_ranges(self):
        if self.signed:
            self.level_high = 2 ** (self.num_bits - 1) - 1
            self.level_low = -(self.level_high + 1)
            if self.is_weights:
                self.level_low += 1
        else:
            self.level_high = 2 ** self.num_bits - 1
            self.level_low = 0

    @property
    def signed(self):
        return self.signed_tensor.item() == 1

    @signed.setter
    def signed(self, signed: bool):
        self.signed_tensor.fill_(signed)

    def quantize(self, x):
        self.set_level_ranges()
        return symmetric_quantize(x, self.levels, self.level_low, self.level_high, self.scale, self.eps)


@MIN_MAX_INITIALIZERS.register('SymmetricQuantizer')
def _initializer(module, name, min_value, max_value, distributed_):
    if min_value.item == np.inf or max_value.item() == -np.inf:
        raise AttributeError('Statistics is not collected for {}'.format(name))
    sign = min_value.item() < 0 or module.within_signed_scope
    if sign != module.signed:
        logger.warning("signed set incorrectly")
    module.signed = int(sign)
    if abs(max_value) > 0.1:
        module.scale.data.fill_(max_value.item())
    if distributed_:
        distributed.broadcast(module.scale, 0)
        distributed.broadcast(module.signed_tensor, 0)
    logger.debug("Statistics: min={:.2f} max={:.2f}".format(min_value.item(), max_value.item()))
    logger.info(
        "Set sign: {} and scale: {:04.2f} for {}".format(module.signed, module.scale.item(), name))


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizer(BaseQuantizer):
    def __init__(self, config):
        super().__init__(config)
        self.is_weights = config.is_weights
        self.input_shape = config.input_shape
        self.per_channel = config.per_channel

        params = config.params
        self.bits = params.bits

        scale_shape = 1
        if self.per_channel:
            scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)

        self.input_low = nn.Parameter(torch.zeros(scale_shape), requires_grad=True)
        self.input_range = nn.Parameter(torch.ones(scale_shape), requires_grad=True)
        self.eps = 1e-16

    @property
    def signed(self):
        return True

    @property
    def level_high(self):
        return 2 ** self.bits - 1

    @property
    def level_low(self):
        return 0

    @property
    def levels(self):
        return 2 ** self.bits

    def quantize(self, x):
        return asymmetric_quantize(x, self.levels, self.level_low, self.level_high, self.input_low, self.input_range,
                                   self.eps)


@MIN_MAX_INITIALIZERS.register('AsymmetricQuantizer')
def _initializer(module, name, min_value, max_value, distributed_):
    if min_value.item() == np.inf or max_value.item() == -np.inf:
        raise AttributeError('Statistics is not collected for {}'.format(name))
    module.input_low.data.fill_(min_value.item())
    range_ = (max_value - min_value).item()
    if range_ > 0.01:
        module.input_range.data.fill_(range_)
    if distributed_:
        distributed.broadcast(module.input_low, 0)
        distributed.broadcast(module.input_range, 0)
    logger.debug("Statistics: min={:.2f} max={:.2f}".format(min_value.item(), max_value.item()))
    logger.info("Set input_low: {:04.2f} and input_range: {:04.2f} for {}"
                .format(module.input_low.item(), module.input_range.item(), name))
