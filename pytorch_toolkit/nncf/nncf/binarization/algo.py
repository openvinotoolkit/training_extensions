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

from typing import Callable, Any

import torch

from .layers import BINARIZATION_MODULES, BinarizationMode, WeightBinarizer, ActivationBinarizer
from .binarized_network import BinarizedNetwork
from .schedulers import BINARIZATION_SCHEDULERS
from ..algo_selector import COMPRESSION_ALGORITHMS
from ..compression_method_api import CompressionAlgorithm
from nncf.dynamic_graph.graph_builder import ModelInputInfo


@COMPRESSION_ALGORITHMS.register('binarization')
class Binarization(CompressionAlgorithm):
    def __init__(self, model, config,
                 input_infos: ModelInputInfo = None,
                 dummy_forward_fn: Callable[[torch.nn.Module], Any] = None):
        super().__init__(model, config, input_infos, dummy_forward_fn)

        self.ignored_scopes = self.config.get('ignored_scopes')
        self.target_scopes = self.config.get('target_scopes')

        self.mode = self.config.get('mode', BinarizationMode.XNOR)

        self._model = BinarizedNetwork(model, self.__create_binarize_module,
                                       input_infos=self.input_infos,
                                       ignored_scopes=self.ignored_scopes,
                                       target_scopes=self.target_scopes)
        self.is_distributed = False
        scheduler_cls = BINARIZATION_SCHEDULERS.get("staged")
        self._scheduler = scheduler_cls(self, self.config)

    def distributed(self):
        self.is_distributed = True

    def __create_binarize_module(self):
        return BINARIZATION_MODULES.get(self.mode)()

    def enable_activation_binarization(self):
        for _, m in self._model.named_modules():
            if isinstance(m, ActivationBinarizer):
                m.enable()

    def enable_weight_binarization(self):
        for _, m in self._model.named_modules():
            if isinstance(m, WeightBinarizer):
                m.enable()
