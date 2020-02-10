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
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import distributed

from nncf.debug import is_debug
from nncf.initialization import INITIALIZABLE_MODULES
from .quantize_functions import symmetric_quantize, asymmetric_quantize
from ..layer_utils import COMPRESSION_MODULES
from ..registry import Registry
from ..utils import get_per_channel_scale_shape, get_flat_tensor_contents_string

logger = logging.getLogger(__name__)

QUANTIZATION_MODULES = Registry('quantization_modules')
BINARIZATION_MODULES = Registry('binarization_modules')


class QuantizationMode:
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class BinarizationMode:
    XNOR = "xnor"
    DOREFA = "dorefa"


class QuantizerConfig:
    def __init__(self, bits=8,
                 mode=QuantizationMode.SYMMETRIC,
                 signedness_to_force=None,
                 per_channel=False,
                 input_shape=None,
                 is_weights=False):
        self.bits = bits
        self.mode = mode
        self.signedness_to_force = signedness_to_force
        self.per_channel = per_channel
        self.is_weights = is_weights
        self.input_shape = input_shape


class BaseQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.config = config
        self.init_stage = False
        self.initialized = False
        self.state_dict_name = None
        self.call_count = 0
        self.scale_shape = 1

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
        if is_debug():
            self.call_count += 1
        if self.init_stage:
            return x
        return self.quantize(x)

    def quantize(self, x):
        raise NotImplementedError

    def reset_call_counter(self):
        self.call_count = 0

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def apply_minmax_init(self, min_values, max_values, distributed_,
                          log_module_name: str = None):
        raise NotImplementedError

@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.SYMMETRIC)
@INITIALIZABLE_MODULES.register()
class SymmetricQuantizer(BaseQuantizer):
    SCALE_PARAM_NAME = 'scale'

    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config.input_shape
        self.per_channel = config.per_channel
        self.is_weights = config.is_weights
        self.num_bits = config.bits
        self.signedness_to_force = config.signedness_to_force
        self.signed_tensor = nn.Parameter(torch.IntTensor([0]), requires_grad=False)
        self.collect_scale_statistics = False

        if self.per_channel:
            self.scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)
        self.scale = nn.Parameter(torch.ones(self.scale_shape), requires_grad=True)
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

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.SCALE_PARAM_NAME: self.scale.detach()}

    def apply_minmax_init(self, min_values, max_values, distributed_, log_module_name: str = None):
        if self.initialized:
            logger.debug("Skipped initializing {} - loaded from checkpoint".format(log_module_name))
            return
        if torch.any(torch.eq(min_values, np.inf)) or torch.any(torch.eq(max_values, -np.inf)):
            raise AttributeError('Statistics is not collected for {}'.format(log_module_name))
        sign = torch.any(torch.lt(min_values, 0))
        if self.signedness_to_force is not None and sign != self.signedness_to_force:
            logger.warning("Forcing signed to {} for module {}".format(self.signedness_to_force, log_module_name))
            sign = self.signedness_to_force
        self.signed = int(sign)

        abs_max = torch.max(torch.abs(max_values), torch.abs(min_values))
        SCALE_LOWER_THRESHOLD = 0.1
        self.scale.fill_(SCALE_LOWER_THRESHOLD)
        self.scale.masked_scatter_(torch.gt(abs_max, SCALE_LOWER_THRESHOLD), abs_max)

        if distributed_:
            distributed.broadcast(self.scale, 0)
            distributed.broadcast(self.signed_tensor, 0)

        logger.info(
            "Set sign: {} and scale: {} for {}".format(self.signed,
                                                       get_flat_tensor_contents_string(self.scale),
                                                       log_module_name))

@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.ASYMMETRIC)
@INITIALIZABLE_MODULES.register()
class AsymmetricQuantizer(BaseQuantizer):
    INPUT_LOW_PARAM_NAME = 'input_low'
    INPUT_RANGE_PARAM_NAME = 'input_range'

    def __init__(self, config):
        super().__init__(config)
        self.is_weights = config.is_weights
        self.input_shape = config.input_shape
        self.per_channel = config.per_channel
        self.bits = config.bits

        self.scale_shape = 1
        if self.per_channel:
            self.scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)

        self.input_low = nn.Parameter(torch.zeros(self.scale_shape), requires_grad=True)
        self.input_range = nn.Parameter(torch.ones(self.scale_shape), requires_grad=True)
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

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.INPUT_LOW_PARAM_NAME: self.input_low.detach(),
                self.INPUT_RANGE_PARAM_NAME: self.input_range.detach()}

    def apply_minmax_init(self, min_values, max_values, distributed_, log_module_name: str = None):
        if self.initialized:
            logger.debug("Skipped initializing {} - loaded from checkpoint".format(log_module_name))
            return
        if torch.any(torch.eq(min_values, np.inf)) or torch.any(torch.eq(max_values, -np.inf)):
            raise AttributeError('Statistics is not collected for {}'.format(log_module_name))
        self.input_low.data = min_values.data
        range_ = max_values - min_values
        self.input_range.masked_scatter_(torch.gt(range_, 0.01), range_)

        if distributed_:
            distributed.broadcast(self.input_low, 0)
            distributed.broadcast(self.input_range, 0)
        logger.info("Set input_low: {} and input_range: {} for {}"
                    .format(get_flat_tensor_contents_string(self.input_low),
                            get_flat_tensor_contents_string(self.input_range), log_module_name))
