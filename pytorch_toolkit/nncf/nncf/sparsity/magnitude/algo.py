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

from .functions import WEIGHT_IMPORTANCE_FUNCTIONS, calc_magnitude_binary_mask
from ..layers import BinaryMask
from ..base_algo import BaseSparsityAlgo
from ..schedulers import SPARSITY_SCHEDULERS
from ...algo_selector import COMPRESSION_ALGORITHMS
from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.algo_selector import create_dummy_forward_fn

logger = logging.getLogger(__name__)


@COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
class MagnitudeSparsity(BaseSparsityAlgo):
    def __init__(self, model, config, input_infos: ModelInputInfo = None, dummy_forward_fn=None, **kwargs):
        super().__init__(model, config, input_infos, dummy_forward_fn)
        self.sparsity_level = self.threshold = 0

        self.ignored_scopes = self.config.get('ignored_scopes')
        self.target_scopes = self.config.get('target_scopes')
        self.dummy_forward_fn = dummy_forward_fn
        if self.dummy_forward_fn is None:
            self.dummy_forward_fn = create_dummy_forward_fn(input_infos)

        params = self.config.get("params", {})
        device = next(model.parameters()).device

        self.weight_importance = WEIGHT_IMPORTANCE_FUNCTIONS.get(
            self.config.get('weight_importance', 'normed_abs'))

        self._replace_sparsifying_modules_by_nncf_modules(device, self.ignored_scopes, self.target_scopes, logger)
        self._register_weight_sparsifying_operations(device, self.ignored_scopes, self.target_scopes, logger)

        scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "polynomial"))
        self._scheduler = scheduler_cls(self, self.config)

    def statistics(self):
        stats = super().statistics()
        stats['sparsity_threshold'] = self.threshold
        stats['sparsity_level'] = self.sparsity_level
        return stats

    def freeze(self):
        pass

    def do_dummy_forward(self):
        train_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            self.dummy_forward_fn(self.model)
        if train_mode:
            self.model.train()

    def set_sparsity_level(self, sparsity_level):
        if sparsity_level >= 1 or sparsity_level < 0:
            raise AttributeError(
                'Sparsity level should be within interval [0,1), actual value to set is: {}'.format(sparsity_level))
        self.sparsity_level = sparsity_level
        self.threshold = self._select_threshold()
        self._set_masks_for_threshold(self.threshold)

    def _select_threshold(self):
        all_weights = self._collect_all_weights()
        threshold = all_weights[int(all_weights.size(0) * self.sparsity_level)].item()
        return threshold

    def _set_masks_for_threshold(self, threshold_val):

        def hook(module, inputs):
            module.binary_mask = calc_magnitude_binary_mask(inputs[0],
                                                            self.weight_importance,
                                                            threshold_val)

        handles = []
        for layer in self.sparsified_module_info:
            handles.append(layer.operand.register_forward_pre_hook(hook))
        self.do_dummy_forward()
        for h in handles:
            h.remove()

    def _collect_all_weights(self):
        all_weights = []

        def hook(module, inputs):
            all_weights.append(self.weight_importance(inputs[0]).view(-1))

        handles = []
        for minfo in self.sparsified_module_info:
            handles.append(minfo.operand.register_forward_pre_hook(hook))
        self.do_dummy_forward()
        all_weights, _ = torch.cat(all_weights).sort()
        for h in handles:
            h.remove()
        return all_weights

    def create_weight_sparsifying_operation(self, module):
        return BinaryMask(module.weight.size())
