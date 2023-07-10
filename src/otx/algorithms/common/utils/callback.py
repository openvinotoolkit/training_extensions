"""Collection of callback utils to run common OTX algorithms."""

# Copyright (C) 2022 Intel Corporation
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

import time

from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback


class TrainingProgressCallback(TimeMonitorCallback):
    """TrainingProgressCallback class for time monitoring."""

    def __init__(self, update_progress_callback, **kwargs):
        super().__init__(update_progress_callback=update_progress_callback, **kwargs)

    def on_train_batch_end(self, batch, logs=None):
        """Callback function on training batch ended."""
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())

    def on_epoch_end(self, epoch, logs=None):
        """Callback function on epoch ended."""
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        progress = ((epoch + 1) / self.total_epochs) * 100
        self._calculate_average_epoch()
        score = None
        if hasattr(self.update_progress_callback, "metric") and isinstance(logs, dict):
            score = logs.get(self.update_progress_callback.metric, None)
        self.update_progress_callback(progress, score=score)


class InferenceProgressCallback(TimeMonitorCallback):
    """InferenceProgressCallback class for time monitoring."""

    def __init__(self, num_test_steps, update_progress_callback, **kwargs):
        super().__init__(
            num_epoch=0,
            num_train_steps=0,
            num_val_steps=0,
            num_test_steps=num_test_steps,
            update_progress_callback=update_progress_callback,
            **kwargs,
        )

    def on_test_batch_end(self, batch=None, logs=None):
        """Callback function on testing batch ended."""
        super().on_test_batch_end(batch, logs)
        self.update_progress_callback(int(self.get_progress()))


class OptimizationProgressCallback(TrainingProgressCallback):
    """Progress callback used for optimization using NNCF.

    There are three stages to the progress bar:
       - 5 % model is loaded
       - 10 % compressed model is initialized
       - 10-100 % compressed model is being fine-tuned
    """

    def __init__(
        self,
        update_progress_callback,
        loading_stage_progress_percentage: int = 5,
        initialization_stage_progress_percentage: int = 5,
        **kwargs,
    ):
        super().__init__(update_progress_callback=update_progress_callback, **kwargs)
        if loading_stage_progress_percentage + initialization_stage_progress_percentage >= 100:
            raise RuntimeError("Total optimization progress percentage is more than 100%")

        self.loading_stage_progress_percentage = loading_stage_progress_percentage
        self.initialization_stage_progress_percentage = initialization_stage_progress_percentage

        # set loading_stage_progress_percentage from the start as the model is already loaded at this point
        if self.update_progress_callback:
            self.update_progress_callback(loading_stage_progress_percentage)

    def on_train_begin(self, logs=None):
        """Callback function when training beginning."""
        super().on_train_begin(logs)
        # Callback initialization takes place here after OTXProgressHook.before_run() is called
        train_percentage = 100 - self.loading_stage_progress_percentage - self.initialization_stage_progress_percentage
        loading_stage_steps = self.total_steps * self.loading_stage_progress_percentage / train_percentage
        initialization_stage_steps = self.total_steps * self.initialization_stage_progress_percentage / train_percentage
        self.total_steps += loading_stage_steps + initialization_stage_steps

        self.current_step = loading_stage_steps + initialization_stage_steps
        self.update_progress_callback(self.get_progress())

    def on_train_end(self, logs=None):
        """Callback function on training ended."""
        super().on_train_end(logs)
        self.update_progress_callback(self.get_progress(), score=logs)

    def on_initialization_end(self):
        """on_initialization_end callback for optimization using NNCF."""
        self.update_progress_callback(
            self.loading_stage_progress_percentage + self.initialization_stage_progress_percentage
        )
