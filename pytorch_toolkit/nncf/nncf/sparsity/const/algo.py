"""
 Copyright (c) 2019-2020 Intel Corporation
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

from nncf.compression_method_api import CompressionAlgorithmController
from nncf.nncf_network import NNCFNetwork
from nncf.sparsity.layers import BinaryMask
from nncf.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController
from nncf.algo_selector import COMPRESSION_ALGORITHMS


@COMPRESSION_ALGORITHMS.register('const_sparsity')
class ConstSparsityBuilder(BaseSparsityAlgoBuilder):
    def create_weight_sparsifying_operation(self, module):
        return BinaryMask(module.weight.size())

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return ConstSparsityController(target_model, self._sparsified_module_info)


class ConstSparsityController(BaseSparsityAlgoController):
    def freeze(self):
        pass

    def set_sparsity_level(self, sparsity_level):
        pass
