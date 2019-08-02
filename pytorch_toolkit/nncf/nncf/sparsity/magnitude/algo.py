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

from .functions import WEIGHT_IMPORTANCE_FUNCTIONS
from .layers import MagnitudeSparsifyingWeight
from ..base_algo import BaseSparsityAlgo
from ..schedulers import SPARSITY_SCHEDULERS
from ...algo_selector import COMPRESSION_ALGORITHMS

logger = logging.getLogger(__name__)


@COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
class MagnitudeSparsity(BaseSparsityAlgo):
    def __init__(self, model, config, input_size):
        super().__init__(model, config, input_size)
        self.sparsity_level = self.threshold = 0
        self.ignored_scopes = self.config.get('ignored_scopes', [])
        params = self.config.get("params", {})
        self.update_mask_on_forward = params.get("update_mask_on_forward", False)
        device = next(model.parameters()).device

        self.weight_importance = WEIGHT_IMPORTANCE_FUNCTIONS.get(
            self.config.get('weight_importance', 'normed_abs'))

        self._replace_sparsifying_modules_by_nncf_modules(device, self.ignored_scopes, logger)
        self._register_weight_sparsifying_operations(device, self.ignored_scopes, logger)

        scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "polynomial"))
        self._scheduler = scheduler_cls(self, self.config)

    def statistics(self):
        stas = super().statistics()
        stas['sparsity_threshold'] = self.threshold
        stas['sparsity_level'] = self.sparsity_level
        return stas

    def freeze(self):
        self.set_sparsify(False)

    def set_sparsify(self, sparsify):
        for minfo in self.sparsified_module_info:
            minfo.operand.sparsify = sparsify

    def do_dummy_forward(self):
        device = next(self.model.parameters()).device
        train_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(self.input_size).to(device))
        if train_mode:
            self.model.train()

    def set_sparsity_level(self, sparsity_level):
        if sparsity_level >= 1 or sparsity_level < 0:
            raise AttributeError(
                'Sparsity level should be within interval [0,1), actual value to set is: {}'.format(sparsity_level))
        self.sparsity_level = sparsity_level
        self.threshold = self._select_threshold()
        self._set_threshold(self.threshold)
        if not self.update_mask_on_forward:
            self.set_sparsify(True)
            self.do_dummy_forward()
            self.set_sparsify(False)

    def _select_threshold(self):
        all_weights = self._collect_all_weights()
        threshold = all_weights[int(all_weights.size(0) * self.sparsity_level)].item()
        return threshold

    def _set_threshold(self, threshold_val):
        for layer in self.sparsified_module_info:
            layer.operand.threshold = threshold_val

    def _collect_all_weights(self):
        all_weights = []

        def hook(module, inputs, _):
            all_weights.append(module.weight_importance(inputs[0]).view(-1))

        handles = []
        for minfo in self.sparsified_module_info:
            handles.append(minfo.operand.register_forward_hook(hook))
        self.do_dummy_forward()
        all_weights, _ = torch.cat(all_weights).sort()
        for h in handles:
            h.remove()
        return all_weights

    def create_weight_sparsifying_operation(self, module):
        return MagnitudeSparsifyingWeight(module.weight.size(), self.weight_importance)
