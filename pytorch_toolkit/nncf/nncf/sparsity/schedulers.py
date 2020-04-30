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

from bisect import bisect_right

import numpy as np

from ..algo_selector import Registry
from ..compression_method_api import CompressionScheduler
from ..config import Config

SPARSITY_SCHEDULERS = Registry("sparsity_schedulers")


class SparsityScheduler(CompressionScheduler):
    def __init__(self, sparsity_algo, params: Config = None):
        super().__init__()
        if params is None:
            self._params = Config()
        else:
            self._params = params

        self.algo = sparsity_algo
        self.sparsity_training_steps = self._params.get('sparsity_training_steps', 100)
        self.max_step = self._params.get('sparsity_steps', 90)
        self.max_sparsity = self._params.get('sparsity_target', 0.5)
        self.initial_sparsity = self._params.get('sparsity_init', 0)

    def initialize(self):
        self._set_sparsity_level()

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        self._set_sparsity_level()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_sparsity_level()

    def _set_sparsity_level(self):
        self.algo.set_sparsity_level(self.current_sparsity_level)
        if self.last_epoch >= self.sparsity_training_steps:
            self.algo.freeze()

    def _calc_density_level(self):
        raise NotImplementedError

    @property
    def current_sparsity_level(self):
        return 1 - self._calc_density_level()


@SPARSITY_SCHEDULERS.register("polynomial")
class PolynomialSparseScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params=None):
        super().__init__(sparsity_algo, params)
        self.power = self._params.get('power', 0.9)
        self._set_sparsity_level()

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        self._set_sparsity_level()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_sparsity_level()

    def _calc_density_level(self):
        step = (min(self.max_step, self.last_epoch) / self.max_step) ** self.power
        current_sparsity = self.initial_sparsity + (self.max_sparsity - self.initial_sparsity) * step
        return 1 - current_sparsity


@SPARSITY_SCHEDULERS.register("exponential")
class ExponentialSparsityScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params=None):
        super().__init__(sparsity_algo, params)
        self.a, self.k = self._init_exp(self.initial_sparsity, self.max_sparsity, sparsity_steps=self.max_step)
        self._set_sparsity_level()

    def _calc_density_level(self):
        curr_density = self.a * np.exp(-self.k * self.last_epoch)
        min_density = 1 - self.max_sparsity
        return min_density if curr_density < min_density else curr_density

    @staticmethod
    def _init_exp(initial_sparsity, max_sparsity, sparsity_steps=20):
        p1 = (0, 1 - initial_sparsity)
        p2 = (sparsity_steps, 1 - max_sparsity)
        k = np.log(p2[1] / p1[1]) / (p1[0] - p2[0])
        a = p1[1] / np.exp(-k * p1[0])
        return a, k


@SPARSITY_SCHEDULERS.register("adaptive")
class AdaptiveSparsityScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params=None):
        super().__init__(sparsity_algo, params)
        self.sparsity_loss = sparsity_algo.loss
        from .rb.loss import SparseLoss
        if not isinstance(self.sparsity_loss, SparseLoss):
            raise TypeError('AdaptiveSparseScheduler expects SparseLoss, but {} is given'.format(
                self.sparsity_loss.__class__.__name__))
        self.decay_step = params.get('step', 0.05)
        self.eps = params.get('eps', 0.03)
        self.patience = params.get('patience', 1)
        self.sparsity_target = self.initial_sparsity
        self.num_bad_epochs = 0
        self._set_sparsity_level()

    def epoch_step(self, epoch=None):
        super().step(epoch)
        if self.sparsity_loss.current_sparsity >= self.sparsity_target - self.eps:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self.sparsity_target = min(self.sparsity_target + self.decay_step, self.max_sparsity)
        self._set_sparsity_level()

    def state_dict(self):
        sd = super().state_dict()
        sd['num_bad_epochs'] = self.num_bad_epochs
        sd['current_sparsity_level'] = self.sparsity_target
        return sd

    def _calc_density_level(self):
        return 1 - self.sparsity_target


@SPARSITY_SCHEDULERS.register("multistep")
class MultiStepSparsityScheduler(SparsityScheduler):
    def _calc_density_level(self):
        return 1 - self.sparsity_level

    def __init__(self, sparsity_algo, params):
        super().__init__(sparsity_algo, params)
        self.sparsity_levels = self._params.get('sparsity_levels', [0.1, 0.5])
        self.steps = self._params.get('steps', [90])
        if len(self.steps) + 1 != len(self.sparsity_levels):
            raise AttributeError('number of sparsity levels must equal to number of steps + 1')

        self.initial_sparsity = self.sparsity_level = self.sparsity_levels[0]
        self.max_sparsity = max(self.sparsity_levels)
        self.sparsity_algo = sparsity_algo
        self.steps = sorted(self.steps)
        self.max_step = self.steps[-1]
        self.prev_ind = 0
        self._set_sparsity_level()

    def epoch_step(self, last=None):
        super().epoch_step(last)
        ind = bisect_right(self.steps, self.last_epoch)
        if ind != self.prev_ind:
            self.sparsity_level = self.sparsity_levels[ind]
            self.prev_ind = ind
        self._set_sparsity_level()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        ind = bisect_right(self.steps, self.last_epoch)
        if ind > 0:
            self.prev_ind = ind
            self.sparsity_level = self.sparsity_levels[ind]
            self._set_sparsity_level()
