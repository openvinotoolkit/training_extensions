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
from ..layers import BinaryMask
from ...layer_utils import COMPRESSION_MODULES


@COMPRESSION_MODULES.register()
class ConstSparsifyingWeight(BinaryMask):
    def _calc_training_binary_mask(self, weight):
        return self.binary_mask

    def _calc_binary_mask(self, weight):
        return self.binary_mask
