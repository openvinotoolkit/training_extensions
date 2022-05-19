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
from typing import Iterable, Union
import yaml
import time

import numpy as np

from ote_sdk.entities.train_parameters import UpdateProgressCallback
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from ote_sdk.utils.argument_checks import (
    YamlFilePathCheck,
    check_input_parameters_type,
)


@check_input_parameters_type({"path": YamlFilePathCheck})
def load_template(path):
    with open(path) as f:
        template = yaml.safe_load(f)
    return template


@check_input_parameters_type()
def get_task_class(path: str):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@check_input_parameters_type()
def get_activation_map(features: Union[np.ndarray, Iterable, int, float]):
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
            score = float(score) if score is not None else None
        self.update_progress_callback(self.get_progress(), score=score)


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
        There are three stages to the progress bar:
           - 5 % model is loaded
           - 10 % compressed model is initialized
           - 10-100 % compressed model is being fine-tuned
    """
    def __init__(self, update_progress_callback: UpdateProgressCallback, loading_stage_progress_percentage: int = 5,
                 initialization_stage_progress_percentage: int = 5):
        super().__init__(update_progress_callback=update_progress_callback)
        if loading_stage_progress_percentage + initialization_stage_progress_percentage >= 100:
            raise RuntimeError('Total optimization progress percentage is more than 100%')

        self.loading_stage_progress_percentage = loading_stage_progress_percentage
        self.initialization_stage_progress_percentage = initialization_stage_progress_percentage

        # set loading_stage_progress_percentage from the start as the model is already loaded at this point
        self.update_progress_callback(loading_stage_progress_percentage)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        # Callback initialization takes place here after OTEProgressHook.before_run() is called
        train_percentage = 100 - self.loading_stage_progress_percentage - self.initialization_stage_progress_percentage
        loading_stage_steps = self.total_steps * self.loading_stage_progress_percentage / train_percentage
        initialization_stage_steps = self.total_steps * self.initialization_stage_progress_percentage / train_percentage
        self.total_steps += loading_stage_steps + initialization_stage_steps

        self.current_step = loading_stage_steps + initialization_stage_steps
        self.update_progress_callback(self.get_progress())

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_initialization_end(self):
        self.update_progress_callback(self.loading_stage_progress_percentage +
                                      self.initialization_stage_progress_percentage)
