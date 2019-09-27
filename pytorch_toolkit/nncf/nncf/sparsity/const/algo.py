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

from ..layers import  BinaryMask
from ..base_algo import BaseSparsityAlgo
from ...algo_selector import COMPRESSION_ALGORITHMS

logger = logging.getLogger(__name__)


@COMPRESSION_ALGORITHMS.register('const_sparsity')
class ConstSparsity(BaseSparsityAlgo):
    def __init__(self, model, config, input_size, **kwargs):
        super().__init__(model, config, input_size)
        device = next(model.parameters()).device

        self.ignored_scopes = self.config.get('ignored_scopes')
        self.target_scopes = self.config.get('target_scopes')

        self._replace_sparsifying_modules_by_nncf_modules(device, self.ignored_scopes, self.target_scopes, logger)
        self._register_weight_sparsifying_operations(device, self.ignored_scopes, self.target_scopes, logger)

    def create_weight_sparsifying_operation(self, module):
        return BinaryMask(module.weight.size())

    def freeze(self):
        pass

    def set_sparsity_level(self, sparsity_level):
        pass
