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

from os.path import exists

from action_detection.nn.monitors.classifier_monitor import ClassifierMonitor
from action_detection.nn.monitors.detector_monitor import DetectorMonitor
from action_detection.nn.monitors.action_monitor import ActionMonitor
from action_detection.nn.parameters.classifier_parameters import ClassifierParams
from action_detection.nn.parameters.detector_parameters import DetectorParams
from action_detection.nn.parameters.action_parameters import ActionParams
from action_detection.nn.parameters.common import load_config


def get_monitor(config_path, batch_size=1, num_gpu=1, log_dir='', src_scope='', snapshot_path='', init_model_path=''):
    """Configures task-specific monitor to control the network.

    :param config_path: Path to configuration file
    :param batch_size: Size of batch
    :param num_gpu: Number of GPUs
    :param log_dir: Directory for lor logging
    :param src_scope: Source network scope name to load from
    :param snapshot_path: Path to snapshot
    :param init_model_path: Path to model weights to initialize from
    :return: Configured network monitor
    """

    assert exists(config_path)
    assert batch_size > 0
    assert num_gpu > 0

    config_values = load_config(config_path)

    if config_values.NETWORK_TYPE == 'classification':
        model_params = ClassifierParams(config_values, batch_size, num_gpu)
        out_monitor = ClassifierMonitor(model_params, batch_size, num_gpu, log_dir,
                                        src_scope, snapshot_path, init_model_path)
    elif config_values.NETWORK_TYPE == 'detection':
        model_params = DetectorParams(config_values, batch_size, num_gpu)
        out_monitor = DetectorMonitor(model_params, batch_size, num_gpu, log_dir,
                                      src_scope, snapshot_path, init_model_path)
    elif config_values.NETWORK_TYPE == 'action':
        model_params = ActionParams(config_values, batch_size, num_gpu)
        out_monitor = ActionMonitor(model_params, batch_size, num_gpu, log_dir,
                                    src_scope, snapshot_path, init_model_path)
    else:
        raise Exception('Invalid network type: {}'.format(config_values.NETWORK_TYPE))

    return out_monitor
