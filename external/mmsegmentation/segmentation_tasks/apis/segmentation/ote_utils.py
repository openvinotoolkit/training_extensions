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
            self.update_progress_callback(self.get_progress())


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


class OptimizationProgressCallback(TrainingProgressCallback):
    """ Progress callback used for optimization using NNCF
        There are four stages to the progress bar:
           - 5 % model is loaded
           - 10 % compressed model is initialized
           - 90 % compressed model is fine-tuned
           - 100 % model is serialized
    """
    def __init__(self, update_progress_callback: UpdateProgressCallback, load_progress: int = 5,
                 initialization_progress: int = 5, serialization_progress: int = 10):
        super().__init__(update_progress_callback=update_progress_callback)
        if load_progress + initialization_progress + serialization_progress >= 100:
            raise RuntimeError('Total optimization progress is more than 100%')

        self.load_progress = load_progress
        self.initialization_progress = initialization_progress
        self.serialization_progress = serialization_progress

        self.serialization_steps = None

        self.update_progress_callback(load_progress)

    def on_train_begin(self, logs=None):
        super(OptimizationProgressCallback, self).on_train_begin(logs)
        # Callback initialization takes place here after OTEProgressHook.before_run() is called
        train_progress = 100 - self.load_progress - self.initialization_progress - self.serialization_progress
        load_steps = self.total_steps * self.load_progress / train_progress
        initialization_steps = self.total_steps * self.initialization_progress / train_progress
        self.serialization_steps = self.total_steps * self.serialization_progress / train_progress
        self.total_steps += load_steps + initialization_steps + self.serialization_steps

        self.current_step = load_steps + initialization_steps
        self.update_progress_callback(self.get_progress())

    def on_train_end(self, logs=None):
        self.current_step = self.total_steps - self.test_steps - self.serialization_steps
        self.current_epoch = self.total_epochs
        self.is_training = False
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_initialization_end(self):
        self.update_progress_callback(self.load_progress + self.initialization_progress)

    def on_serialization_end(self):
        self.current_step += self.serialization_steps
        self.update_progress_callback(self.get_progress())
