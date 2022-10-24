"""Collections of Dataset utils for common OTX algorithms."""

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

import importlib
import time

from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback


def get_task_class(path: str):
    """Return Task classes."""
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class TrainingProgressCallback(TimeMonitorCallback):
    """TrainingProgressCallback class for time monitoring."""

    def __init__(self, update_progress_callback):
        super().__init__(0, 0, 0, 0, update_progress_callback=update_progress_callback)

    def on_train_batch_end(self, batch, logs=None):
        """Callback function on training batch ended."""
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())

    def on_epoch_end(self, epoch, logs=None):
        """Callback function on epoch ended."""
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        self._calculate_average_epoch()
        score = None
        if hasattr(self.update_progress_callback, "metric") and isinstance(logs, dict):
            score = logs.get(self.update_progress_callback.metric, None)
            score = float(score) if score is not None else None
            if score is not None:
                iter_num = logs.get("current_iters", None)
                if iter_num is not None:
                    print(f"score = {score} at epoch {epoch} / {int(iter_num)}")
                    # as a trick, score (at least if it's accuracy not the loss) and iteration number
                    # could be assembled just using summation and then disassembeled.
                    if score < 1.0:
                        score = score + int(iter_num)
                    else:
                        score = -(score + int(iter_num))
        self.update_progress_callback(self.get_progress(), score=score)
