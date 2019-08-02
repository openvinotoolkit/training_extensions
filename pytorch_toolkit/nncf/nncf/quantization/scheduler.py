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

import torch.distributed as distributed

from ..compression_method_api import CompressionScheduler
from ..utils import get_all_modules_by_type

logger = logging.getLogger(__name__)


class QuantizeScheduler(CompressionScheduler):
    def __init__(self, model, init_scale_steps=0):
        super().__init__()
        self.model = model
        self.init_scale_steps = init_scale_steps
        self.distributed_broadcast = False

        self.scales_initialized = False
        if init_scale_steps == 0:
            self._init_scales(use_statistics=False)

    def step(self, last=None):
        super().step(last)

        if not self.scales_initialized and self.last_step >= self.init_scale_steps:
            self._init_scales(use_statistics=True)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.last_step >= self.init_scale_steps:
            self._init_scales(use_statistics=False)

    def _init_scales(self, use_statistics):
        self.scales_initialized = True
        # must iterate over OrderedDict to broadcast correctly
        for name, layer in get_all_modules_by_type(self.model, "Quantize").items():
            layer.init_scales(use_statistics)

            if self.distributed_broadcast and use_statistics:
                distributed.broadcast(layer.scale, 0)
                distributed.broadcast(layer.signed_tensor, 0)

            logger.info("Set sign: {} and scale: {:04.2f} for {}".format(layer.signed, layer.scale.item(), name))

    def distributed(self):
        self.distributed_broadcast = True
