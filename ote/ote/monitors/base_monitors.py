"""
 Copyright (c) 2021 Intel Corporation

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

import time

from torch.utils.tensorboard import SummaryWriter

from ote.interfaces.monitoring import (IMetricsMonitor,
                                    IPerformanceMonitor, IStopCallback)


class StopCallback(IStopCallback):
    def __init__(self):
        self.stop = False

    def stop(self):
        self.stop = True

    def check_stop(self):
        return self.stop

    def reset(self):
        self.stop = False


class MetricsMonitor(IMetricsMonitor):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.tb = None

    def add_scalar(self, capture: str, value: float, timestamp: int):
        if not self.tb:
            self.tb = SummaryWriter(self.log_dir)
        self.tb.add_scalar(capture, value, timestamp)

    def close(self):
        if self.tb:
            self.tb.close()
            self.tb = None


class PerformanceMonitor(IPerformanceMonitor):
    def init(self, total_epochs: int, num_train_steps: int, num_validation_steps: int):
        self.total_epochs = total_epochs
        self.num_train_steps = num_train_steps
        self.num_validation_steps = num_validation_steps

        self.train_steps_passed = 0
        self.val_steps_passed = 0
        self.train_epochs_passed = 0

        self.granularity = 1. / total_epochs
        self.start_epoch_time = 0
        self.start_batch_time = 0
        self.avg_epoch_time = 0
        self.avg_batch_time = 0
        self.avg_test_batch_time = 0

    def on_train_batch_begin(self):
        self.start_batch_time = time.time()

    def on_train_batch_end(self):
        self.train_steps_passed += 1
        batch_time = time.time() - self.start_batch_time
        self.avg_batch_time = (self.avg_batch_time * (self.train_steps_passed - 1) + batch_time) / self.train_steps_passed

    def on_test_batch_begin(self):
        self.start_batch_time = time.time()

    def on_test_batch_end(self):
        self.val_steps_passed += 1
        batch_time = time.time() - self.start_batch_time
        self.avg_test_batch_time = (self.avg_test_batch_time * (self.val_steps_passed - 1) + batch_time) / self.val_steps_passed

    def on_train_begin(self):
        self.train_steps_passed = 0
        self.val_steps_passed = 0

    def on_train_end(self):
        pass

    def on_train_epoch_begin(self):
        self.start_epoch_time = time.time()

    def on_train_epoch_end(self):
        self.val_steps_passed = 0
        self.train_epochs_passed += 1
        epoch_time = time.time() - self.start_epoch_time
        self.avg_epoch_time = (self.avg_epoch_time * (self.train_epochs_passed - 1) + epoch_time) / self.train_epochs_passed

    def get_training_progress(self) -> int:
        return int(self.train_epochs_passed / self.total_epochs * 100) + \
               int(self.granularity * (self.train_steps_passed % self.num_train_steps) / self.num_train_steps * 100)

    def get_test_progress(self) -> int:
        print(self.val_steps_passed, self.num_validation_steps)
        return int(self.val_steps_passed / self.num_validation_steps * 100)

    def get_test_eta(self) -> int:
        remaining_steps = self.num_validation_steps - self.val_steps_passed
        remaining_batch_time = remaining_steps * self.avg_batch_time
        return int(remaining_batch_time)

    def get_training_eta(self) -> int:
        remaining_steps = self.num_train_steps - self.train_steps_passed % self.num_train_steps
        remaining_epochs = self.total_epochs - self.train_epochs_passed

        remaining_batch_time = remaining_steps * self.avg_batch_time
        remaining_epoch_time = remaining_epochs * self.avg_epoch_time

        if remaining_epoch_time == 0:
            remaining_time = remaining_step_time * 1.5
        else:
            remaining_time = remaining_epoch_time
            epoch_completion = (self.train_steps_passed % self.num_train_steps) / self.num_train_steps
            remaining_time -= epoch_completion * self.avg_epoch_time

        return int(remaining_time)


class DefaultStopCallback(IStopCallback):
    def stop(self):
        pass

    def check_stop(self):
        pass
        return self.stop

    def reset(self):
        pass


class DefaultMetricsMonitor(IMetricsMonitor):
    def add_scalar(self, capture: str, value: float, timestamp: int):
        pass

    def close(self):
        pass


class DefaultPerformanceMonitor(IPerformanceMonitor):
    def init(self, total_epochs: int, num_train_steps: int, num_validation_steps: int):
        pass

    def on_train_batch_begin(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_test_batch_begin(self):
        pass

    def on_test_batch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_train_epoch_begin(self):
        pass

    def on_train_epoch_end(self):
        pass

    def get_training_progress(self) -> int:
        return 0

    def get_test_progress(self) -> int:
        return 0

    def get_test_eta(self) -> int:
        return 0

    def get_training_eta(self) -> int:
        return 0
