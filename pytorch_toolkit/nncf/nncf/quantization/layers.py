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

import torch
import torch.nn as nn

from .quantize_functions import quantize as quantize_imp
from ..layers import COMPRESSION_MODULES

logger = logging.getLogger(__name__)


@COMPRESSION_MODULES.register()
class Quantize(nn.Module):
    def __init__(self, qbias=False, num_bits=8, signed=True, is_weights=False):
        super().__init__()
        self.signed_tensor = nn.Parameter(torch.IntTensor([signed]), requires_grad=False)
        self.collect_scale_statistics = False
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.qbias = nn.Parameter(torch.zeros(0), requires_grad=True) if qbias else None
        self.is_weights = is_weights
        self.num_bits = num_bits
        self.init_stage = False

    @property
    def signed(self):
        return self.signed_tensor.item() == 1

    @signed.setter
    def signed(self, signed: bool):
        self.signed_tensor.fill_(signed)

    def forward(self, x):
        if self.init_stage:
            return x
        return quantize_imp(x, self.scale, self.num_bits, self.qbias, self.signed, self.is_weights)
