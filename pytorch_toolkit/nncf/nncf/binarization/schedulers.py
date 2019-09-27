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

from nncf.algo_selector import Registry
from nncf.compression_method_api import CompressionScheduler
from nncf.config import Config

BINARIZATION_SCHEDULERS = Registry("binarization_schedulers")


@BINARIZATION_SCHEDULERS.register("staged")
class StagedBinarizationScheduler(CompressionScheduler):
    def __init__(self, binarization_algo, config=None):
        super().__init__()
        if config is None:
            config = Config()
        c = config['params']
        self.config = config
        self.algo = binarization_algo
        self.activations_bin_start_epoch = c.get('activations_bin_start_epoch', 1)
        self.weights_bin_start_epoch = c.get('weights_bin_start_epoch', 1)
        self._set_binarization_status()

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        self._set_binarization_status()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_binarization_status()

    def _set_binarization_status(self):
        if self.last_epoch >= self.activations_bin_start_epoch:
            self.algo.enable_activation_binarization()
        if self.last_epoch >= self.weights_bin_start_epoch:
            self.algo.enable_weight_binarization()

    def _calc_density_level(self):
        raise NotImplementedError
