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

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch import distributed

from nncf.debug import is_debug
from nncf.functions import clamp
from nncf.nncf_logger import logger as nncf_logger
from .quantize_functions import symmetric_quantize, asymmetric_quantize
from ..layer_utils import COMPRESSION_MODULES
from ..registry import Registry
from ..utils import get_per_channel_scale_shape, get_flat_tensor_contents_string

QUANTIZATION_MODULES = Registry('quantization_modules')
INITIALIZABLE_MODULES = Registry('initializable_modules')


class QuantizationMode:
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


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
        # TODO: add optional level_low and level_high setting to be parsed from HW config

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return "B:{bits} M:{mode} SGN:{signedness} W:{is_weights} PC:{per_channel}".format(
            bits=self.bits,
            mode='S' if self.mode == QuantizationMode.SYMMETRIC else 'A',
            signedness='ANY' if self.signedness_to_force is None else ('S' if self.signedness_to_force else 'U'),
            is_weights='Y' if self.is_weights else 'N',
            per_channel='Y' if self.per_channel else 'N')

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return self.bits < other.bits or \
               (self.mode == QuantizationMode.SYMMETRIC and other.mode == QuantizationMode.ASYMMETRIC) or \
               (self.signedness_to_force is None and other.signedness_to_force is not None) or \
               (not self.per_channel and other.per_channel)


class BaseQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.input_shape = config.input_shape
        self.per_channel = config.per_channel
        self.is_weights = config.is_weights
        self.signedness_to_force = config.signedness_to_force
        self._num_bits = nn.Parameter(torch.IntTensor([config.bits]), requires_grad=False)

        self.level_high = 0
        self.level_low = 0
        self.levels = 0

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

    def enable_gradients(self):
        return NotImplementedError

    def disable_gradients(self):
        return NotImplementedError

    def forward(self, x):
        if is_debug():
            self.call_count += 1
        if self.init_stage:
            return x
        self.set_level_ranges()
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

    def set_level_ranges(self):
        raise NotImplementedError

    @property
    def signed(self):
        return NotImplementedError

    @property
    def num_bits(self):
        return self._num_bits.item()

    @num_bits.setter
    def num_bits(self, num_bits: int):
        self._num_bits.fill_(num_bits)

    def broadcast_num_bits(self, src: int = 0):
        distributed.broadcast(self._num_bits, src=src)


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizer(BaseQuantizer):
    SCALE_PARAM_NAME = 'scale'

    def __init__(self, config):
        super().__init__(config)
        self.signed_tensor = nn.Parameter(torch.IntTensor([0]), requires_grad=False)
        self.collect_scale_statistics = False
        if self.per_channel:
            self.scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)
        self.scale = nn.Parameter(torch.ones(self.scale_shape), requires_grad=True)
        self.eps = 1e-16

    def enable_gradients(self):
        self.scale.requires_grad = True

    def disable_gradients(self):
        self.scale.requires_grad = False

    def set_level_ranges(self):
        if self.signed:
            self.level_high = 2 ** (self.num_bits - 1) - 1
            self.level_low = -(self.level_high + 1)
            if self.is_weights:
                self.level_low += 1
        else:
            self.level_high = 2 ** self.num_bits - 1
            self.level_low = 0
        self.levels = 2 ** self.num_bits
        if self.is_weights:
            self.levels -= 1

    @property
    def signed(self):
        return self.signed_tensor.item() == 1

    @signed.setter
    def signed(self, signed: bool):
        self.signed_tensor.fill_(signed)

    def quantize(self, x):
        return symmetric_quantize(x, self.levels, self.level_low, self.level_high, self.scale, self.eps)

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.SCALE_PARAM_NAME: self.scale.detach()}

    def apply_minmax_init(self, min_values, max_values, distributed_, log_module_name: str = None):
        if self.initialized:
            nncf_logger.debug("Skipped initializing {} - loaded from checkpoint".format(log_module_name))
            return
        if torch.any(torch.eq(min_values, np.inf)) or torch.any(torch.eq(max_values, -np.inf)):
            raise AttributeError('Statistics is not collected for {}'.format(log_module_name))
        sign = torch.any(torch.lt(min_values, 0))
        if self.signedness_to_force is not None and sign != self.signedness_to_force:
            nncf_logger.warning("Forcing signed to {} for module {}".format(self.signedness_to_force, log_module_name))
            sign = self.signedness_to_force
        self.signed = int(sign)

        abs_max = torch.max(torch.abs(max_values), torch.abs(min_values))
        SCALE_LOWER_THRESHOLD = 0.1
        self.scale.fill_(SCALE_LOWER_THRESHOLD)
        self.scale.masked_scatter_(torch.gt(abs_max, SCALE_LOWER_THRESHOLD), abs_max)

        if distributed_:
            distributed.broadcast(self.scale, 0)
            distributed.broadcast(self.signed_tensor, 0)

        nncf_logger.info(
            "Set sign: {} and scale: {} for {}".format(self.signed,
                                                       get_flat_tensor_contents_string(self.scale),
                                                       log_module_name))


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizer(BaseQuantizer):
    INPUT_LOW_PARAM_NAME = 'input_low'
    INPUT_RANGE_PARAM_NAME = 'input_range'

    def __init__(self, config):
        super().__init__(config)
        self.scale_shape = 1
        if self.per_channel:
            self.scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)

        self.input_low = nn.Parameter(torch.zeros(self.scale_shape), requires_grad=True)
        self.input_range = nn.Parameter(torch.ones(self.scale_shape), requires_grad=True)
        self.eps = 1e-16

    def enable_gradients(self):
        self.input_low.requires_grad = True
        self.input_range.requires_grad = True

    def disable_gradients(self):
        self.input_low.requires_grad = False
        self.input_range.requires_grad = False

    @property
    def signed(self):
        return True

    def set_level_ranges(self):
        self.level_high = 2 ** self.num_bits - 1
        self.level_low = 0
        self.levels = 2 ** self.num_bits

    def quantize(self, x):
        return asymmetric_quantize(x, self.levels, self.level_low, self.level_high, self.input_low, self.input_range,
                                   self.eps)

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.INPUT_LOW_PARAM_NAME: self.input_low.detach(),
                self.INPUT_RANGE_PARAM_NAME: self.input_range.detach()}

    def apply_minmax_init(self, min_values, max_values, distributed_, log_module_name: str = None):
        if self.initialized:
            nncf_logger.debug("Skipped initializing {} - loaded from checkpoint".format(log_module_name))
            return
        if torch.any(torch.eq(min_values, np.inf)) or torch.any(torch.eq(max_values, -np.inf)):
            raise AttributeError('Statistics is not collected for {}'.format(log_module_name))
        ranges = max_values - min_values
        max_range = torch.max(max_values - min_values)
        eps = 1e-2
        correction = (clamp(ranges, low=eps * max_range, high=max_range) - ranges) * 0.5
        self.input_range.data = (ranges + 2 * correction).data
        self.input_low.data = (min_values - correction).data

        if distributed_:
            distributed.broadcast(self.input_low, 0)
            distributed.broadcast(self.input_range, 0)
        nncf_logger.info("Set input_low: {} and input_range: {} for {}"
                         .format(get_flat_tensor_contents_string(self.input_low),
                                 get_flat_tensor_contents_string(self.input_range), log_module_name))
