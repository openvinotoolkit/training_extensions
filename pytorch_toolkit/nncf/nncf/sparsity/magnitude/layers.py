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

from .functions import calc_magnitude_binary_mask as calc_magnitude_binary_mask_impl, abs_magnitude
from ..layers import BinaryMask
from ...layers import COMPRESSION_MODULES


@COMPRESSION_MODULES.register()
class MagnitudeSparsifyingWeight(BinaryMask):
    def __init__(self, size, weight_importance=None, sparsify=True):
        super().__init__(size)
        self.sparsify = sparsify
        if weight_importance is None:
            weight_importance = abs_magnitude

        self.weight_importance = weight_importance
        self.threshold = 0

    def _calc_training_binary_mask(self, weight):
        if not self.sparsify:
            return self.binary_mask
        return calc_magnitude_binary_mask_impl(weight, self.weight_importance, self.threshold)

    def _calc_binary_mask(self, weight):
        return (self.weight_importance(weight) > self.threshold).float()
