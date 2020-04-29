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

from typing import List

import torch.nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nncf.compression_method_api import CompressionLoss, CompressionScheduler, \
    CompressionAlgorithmController
from nncf.nncf_network import NNCFNetwork


class CompositeCompressionLoss(CompressionLoss):
    def __init__(self):
        super().__init__()
        self._child_losses = torch.nn.ModuleList()

    @property
    def child_losses(self):
        return self._child_losses

    def add(self, child_loss):
        self._child_losses.append(child_loss)

    def forward(self):
        result_loss = 0
        for loss in self._child_losses:
            result_loss += loss()
        return result_loss

    def statistics(self):
        stats = {}
        for loss in self._child_losses:
            stats.update(loss.statistics())
        return stats


class CompositeCompressionScheduler(CompressionScheduler):
    def __init__(self):
        super().__init__()
        self._child_schedulers = []

    @property
    def child_schedulers(self):
        return self._child_schedulers

    def add(self, child_scheduler):
        self._child_schedulers.append(child_scheduler)

    def step(self, last=None):
        super().step(last)
        for scheduler in self._child_schedulers:
            scheduler.step(last)

    def epoch_step(self, last=None):
        super().epoch_step(last)
        for scheduler in self._child_schedulers:
            scheduler.epoch_step(last)

    def state_dict(self):
        result = {}
        for child_scheduler in self._child_schedulers:
            result.update(child_scheduler.state_dict())
        return result

    def load_state_dict(self, state_dict):
        for child_scheduler in self._child_schedulers:
            child_scheduler.load_state_dict(state_dict)

    def initialize(self):
        for child_scheduler in self._child_schedulers:
            child_scheduler.initialize()


class CompositeCompressionAlgorithmController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork):
        super().__init__(target_model)
        self._child_algos = []  # type: List[CompressionAlgorithmController]
        self._loss = CompositeCompressionLoss()
        self._scheduler = CompositeCompressionScheduler()

    @property
    def child_algos(self):
        return self._child_algos

    def add(self, child_algo: CompressionAlgorithmController):
        # pylint: disable=protected-access
        assert child_algo._model is self._model, "Cannot create a composite controller " \
                                                 "from controllers belonging to different models!"
        self.child_algos.append(child_algo)
        self._loss.add(child_algo.loss)
        self._scheduler.add(child_algo.scheduler)
        self._model = child_algo._model

    def distributed(self):
        for algo in self.child_algos:
            algo.distributed()

    def initialize(self, data_loader: DataLoader = None, criterion: _Loss = None):
        for algo in self.child_algos:
            algo.initialize(data_loader, criterion)

    def statistics(self):
        stats = {}
        for algo in self.child_algos:
            stats.update(algo.statistics())
        return stats

    def export_model(self, filename):
        self.child_algos[-1].export_model(filename)

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        for algo in self.child_algos:
            target_model = algo.apply_to(target_model)
        return target_model
