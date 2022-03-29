# Copyright (C) 2021 Intel Corporation
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


import importlib
import yaml
import time

import numpy as np

from ote_sdk.entities.train_parameters import UpdateProgressCallback
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback


def load_template(path):
    with open(path) as f:
        template = yaml.safe_load(f)
    return template


def get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_activation_map(features):
    min_soft_score = np.min(features)
    max_soft_score = np.max(features)
    factor = 255.0 / (max_soft_score - min_soft_score + 1e-12)

    float_act_map = factor * (features - min_soft_score)
    int_act_map = np.uint8(np.floor(float_act_map))

    return int_act_map


class TrainingProgressCallback(TimeMonitorCallback):
    def __init__(self, update_progress_callback: UpdateProgressCallback):
        super().__init__(0, 0, 0, 0, update_progress_callback=update_progress_callback)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())

    def on_epoch_end(self, epoch, logs=None):
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        self._calculate_average_epoch()
        score = None
        if hasattr(self.update_progress_callback, 'metric') and isinstance(logs, dict):
            score = logs.get(self.update_progress_callback.metric, None)

        # Workaround for NNCF trainer, which uses callback of a different type.
        if score is not None:
            self.update_progress_callback(self.get_progress(), score=float(score))
        else:
            self.update_progress_callback(int(self.get_progress()))


class InferenceProgressCallback(TimeMonitorCallback):
    def __init__(self, num_test_steps, update_progress_callback: UpdateProgressCallback):
        super().__init__(
            num_epoch=0,
            num_train_steps=0,
            num_val_steps=0,
            num_test_steps=num_test_steps,
            update_progress_callback=update_progress_callback)

    def on_test_batch_end(self, batch=None, logs=None):
        super().on_test_batch_end(batch, logs)
        self.update_progress_callback(int(self.get_progress()))
