"""
 Copyright (c) 2020 Intel Corporation
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
import numpy as np
import scipy.optimize
from nncf.compression_method_api import CompressionScheduler
from nncf.config import Config
from nncf.registry import Registry

PRUNING_SCHEDULERS = Registry("pruning_schedulers")


class PruningScheduler(CompressionScheduler):
    def __init__(self, pruning_algo, params: Config = None):
        super().__init__()
        if params is None:
            self._params = Config()
        else:
            self._params = params

        self.algo = pruning_algo

        # Number of initial steps of training before pruning
        self.num_init_steps = self._params.get('num_init_steps', 0)
        self.pruning_steps = self._params.get('pruning_steps', 100)

        # Pruning rates
        self.initial_pruning = self._params.get('pruning_init', 0)
        self.pruning_target = self._params.get('pruning_target', 0.5)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_pruning_level()

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        self._set_pruning_level()

    def _set_pruning_level(self):
        self.algo.set_pruning_rate(self.current_pruning_level)

        if self.last_epoch >= (self.pruning_steps + self.num_init_steps):
            self.algo.freeze()

    def _calc_density_level(self):
        raise NotImplementedError

    @property
    def current_pruning_level(self):
        if self.last_epoch >= self.num_init_steps:
            return 1 - self._calc_density_level()
        return 0


@PRUNING_SCHEDULERS.register("baseline")
class BaselinePruningScheduler(PruningScheduler):
    """
    Baseline scheduler that setting max pruning rate after num_init_steps epoch
    and freeze algorithm after it.
    """
    def __init__(self, pruning_algo, config=None):
        super().__init__(pruning_algo, config)
        self._set_pruning_level()

    def _calc_density_level(self):
        min_density = 1 - self.pruning_target
        return min_density

    def _set_pruning_level(self):
        self.algo.set_pruning_rate(self.current_pruning_level)
        if self.last_epoch >= self.num_init_steps:
            self.algo.freeze()


@PRUNING_SCHEDULERS.register("exponential")
class ExponentialPruningScheduler(PruningScheduler):
    """
    Calculates pruning rate progressively according to the formula
    P = 1 - a * exp(- k * epoch)
    Where:
    epoch - epoch number
    P - pruning rate for current epoch
    a, k - params
    """
    def __init__(self, pruning_algo, config=None):
        super().__init__(pruning_algo, config)
        self.a, self.k = self._init_exp(self.initial_pruning, self.pruning_target, pruning_steps=self.pruning_steps)
        self._set_pruning_level()

    def _calc_density_level(self):
        curr_density = self.a * np.exp(-self.k * (self.last_epoch - self.num_init_steps))
        min_density = 1 - self.pruning_target
        return min_density if curr_density < min_density else curr_density

    @staticmethod
    def _init_exp(initial_pruning, max_pruning, pruning_steps=20):
        p1 = (0, 1 - initial_pruning)
        p2 = (pruning_steps, 1 - max_pruning)
        k = np.log(p2[1] / p1[1]) / (p1[0] - p2[0])
        a = p1[1] / np.exp(-k * p1[0])
        return a, k


@PRUNING_SCHEDULERS.register("exponential_with_bias")
class ExponentialWithBiasPruningScheduler(PruningScheduler):
    """
    Calculates pruning rate progressively according to the formula
    P = a * exp(- k * epoch) + b
    Where:
    epoch - epoch number
    P - pruning rate for current epoch
    a, b, k - params
    """
    def __init__(self, pruning_algo, config=None):
        super().__init__(pruning_algo, config)
        self.a, self.b, self.k = self._init_exp(self.pruning_steps, self.initial_pruning, self.pruning_target)
        self._set_pruning_level()

    def _calc_density_level(self):
        curr_density = 1 - (self.a * np.exp(-self.k * (self.last_epoch - self.num_init_steps)) + self.b)
        min_density = 1 - self.pruning_target
        return min_density if curr_density < min_density else curr_density

    @staticmethod
    def _init_exp(E_max, P_min, P_max, D=1 / 8):
        """
        Find a, b, k for system (from SPFP paper):
        1. P_min = a + b
        2. P_max = a * exp(-k * E_max) + b
        3. 3/4 * P_max = a *  exp(-k * E_max * D) + b
        Where P_min, P_max - minimal and goal levels of pruning rate
        E_max - number of epochs for pruning
        """
        def get_b(a, k):
            return P_min - a

        def get_a(k):
            return (3 / 4 * P_max - P_min) / (np.exp(- D * k * E_max) - 1)

        def f_to_solve(x):
            y = np.exp(D * x * E_max)
            return 1 / 3 * y + 1 / (y ** 7) - 4 / 3

        k = scipy.optimize.fsolve(f_to_solve, [1])[0]
        a = get_a(k)
        b = get_b(a, k)
        return a, b, k
