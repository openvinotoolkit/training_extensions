# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from abc import ABCMeta, abstractmethod

import numpy as np

from action_detection.nn.data.core import ImageSize


class ModelParams(object):
    """Class to control the network parameters.
    """

    __metaclass__ = ABCMeta

    def __init__(self, config_values, batch_size, num_gpu):
        """Constructor.

        :param config_values: Loaded from config parameters
        :param batch_size: Target batch size
        :param num_gpu: Target number of GPUs to compute on
        """

        assert batch_size > 0
        assert num_gpu > 0

        self._config_values = config_values
        self._batch_size = batch_size
        self._num_gpu = num_gpu

        self._epoch_num_steps = int(float(config_values.TRAIN_DATA_SIZE) / float(batch_size * num_gpu))
        self._num_train_steps = int(config_values.MAX_NUM_TRAIN_EPOCHS * self._epoch_num_steps)
        self._num_val_steps = int(np.floor(float(config_values.VAL_DATA_SIZE) / float(batch_size * num_gpu)))

        model_specific_params = self._configure_params()
        self._params = self._add_shared_params(model_specific_params)

    def __getattr__(self, name):
        return self._params[name]

    @abstractmethod
    def _configure_params(self):
        """Returns the parameters for classification network.

        :return: Model parameters
        """

        pass

    def _add_shared_params(self, model_params):
        """Adds general parameters.

        :param model_params: Model-specific parameters
        :return: Updated model parameters
        """

        model_params.type = self._config_values.NETWORK_TYPE
        model_params.backbone = self._config_values.NETWORK_BACKBONE
        model_params.version = self._config_values.NETWORK_VERSION
        model_params.name = self._config_values.NETWORK_NAME
        model_params.image_size = ImageSize(*self._config_values.IMAGE_SIZE)
        model_params.epoch_steps = self._epoch_num_steps
        model_params.val_steps = self._num_val_steps
        model_params.max_train_steps = self._num_train_steps
        model_params.use_nesterov = self._config_values.USE_NESTEROV
        model_params.norm_kernels = self._config_values.NORMALIZE_KERNELS

        return model_params
