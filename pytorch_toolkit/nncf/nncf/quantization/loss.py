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

from ..compression_method_api import CompressionLoss
from ..utils import get_module_by_node_name, init_flops


class FLOPLoss(CompressionLoss):
    def __init__(self, model, node_names, flops_target=0.8, size=(1, 3, 224, 224), cuda=True):
        super().__init__()
        self.model = model
        self.node_names = node_names
        self.flops_target = flops_target
        self.flops_total = self._init_flops(size, cuda)

    def forward(self):
        loss = 0
        for node_name in self.node_names:
            m = get_module_by_node_name(self.model, node_name).mask
            loss += (m.loss() * m.flops).sum()
        return (loss / self.flops_total - self.flops_target).pow(2)

    def _init_flops(self, size, cuda):
        return init_flops(self.model, self.node_names, size, cuda)
