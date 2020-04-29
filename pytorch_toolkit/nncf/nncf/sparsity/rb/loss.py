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

import torch

from ...compression_method_api import CompressionLoss


# Actually in responsible to lean density to target value
class SparseLoss(CompressionLoss):
    def __init__(self, sparse_layers=None, target=1.0, p=0.05):
        super().__init__()
        self._sparse_layers = sparse_layers
        self.target = target
        self.p = p
        self.disabled = False
        self.current_sparsity = 0
        self.mean_sparse_prob = 0

    def set_layers(self, sparse_layers):
        self._sparse_layers = sparse_layers

    def disable(self):
        if not self.disabled:
            self.disabled = True

            for sparse_layer in self._sparse_layers:
                sparse_layer.sparsify = False

    def forward(self):
        if self.disabled:
            return 0

        params = 0
        loss = 0
        sparse_prob_sum = 0
        for sparse_layer in self._sparse_layers:
            if not self.disabled and not sparse_layer.sparsify:
                raise AssertionError(
                    "Invalid state of SparseLoss and SparsifiedWeight: mask is frozen for enabled loss")
            if sparse_layer.sparsify:
                sw_loss = sparse_layer.loss()
                params = params + sw_loss.view(-1).size(0)
                loss = loss + sw_loss.sum()
                sparse_prob_sum += torch.sigmoid(sparse_layer.mask).sum()

        self.mean_sparse_prob = (sparse_prob_sum / params).item()
        self.current_sparsity = 1 - loss / params
        return ((loss / params - self.target) / self.p).pow(2)

    @property
    def target_sparsity_rate(self):
        rate = 1 - self.target
        if rate < 0 or rate > 1:
            raise IndexError("Target is not within range(0,1)")
        return rate

    def statistics(self):
        return {'mean_sparse_prob': 1 - self.mean_sparse_prob}
