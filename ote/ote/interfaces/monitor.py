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


import abc


class IMetricsMonitor(metaclass=abc.ABCMeta):
    """
    Interface for providing Tensorboard-style logging
    """

    @abc.abstractclassmethod
    def add_scalar(self, capture: str, value: float, timestamp: int):
        """
        Similar to Tensorboard method that allows to log values of named scalar variables
        """

    @abc.abstractclassmethod
    def close(self):
        """
        Flushes the collected scalar values to log, closes corresponding file,
        then resets the monitor to default state
        """


class IPerformanceMonitor(metaclass=abc.ABCMeta):
    """
    Interface for collecting training performance numbers
    """
    @abc.abstractclassmethod
    def init(self, total_epochs: int, num_train_steps: int, num_validation_steps: int):
        """
        """

    @abc.abstractclassmethod
    def on_train_batch_begin(self):
        """
        Method starts timer that measures batch forward-backward time during training
        """

    @abc.abstractclassmethod
    def on_train_batch_end(self):
        """
        Method stops timer that measures batch forward-backward time during training
        """

    @abc.abstractclassmethod
    def on_test_batch_begin(self):
        """
        Method starts timer that measures batch forward-backward time during evaluation
        """

    @abc.abstractclassmethod
    def on_test_batch_end(self):
        """
        Method stops timer that measures batch forward-backward time during evaluation
        """

    @abc.abstractclassmethod
    def on_train_begin(self):
        """
        Method notifies the monitor that training has begun
        """

    @abc.abstractclassmethod
    def on_train_end(self):
        """
        Method notifies the monitor that training has finished
        """

    @abc.abstractclassmethod
    def on_train_epoch_begin(self):
        """
        Method notifies the monitor that the next training epoch has begun
        """

    @abc.abstractclassmethod
    def on_train_epoch_end(self):
        """
        Method notifies the monitor that training epoch has finished
        """

    @abc.abstractclassmethod
    def get_training_progress(self) -> int:
        """
        """

    @abc.abstractclassmethod
    def get_test_progress(self) -> int:
        """
        """

    @abc.abstractclassmethod
    def get_test_eta(self) -> int:
        """
        """

    @abc.abstractclassmethod
    def get_training_eta(self) -> int:
        """
        """


class IStopCallback(metaclass=abc.ABCMeta):
    """
    Interface for wrapping a stop signal.
    By default an object implementing this interface should be in permissive state
    """

    @abc.abstractclassmethod
    def check_stop(self) -> bool:
        """
        Method returns True if the object of this class is in stopping state,
        otherwise it returns False
        """

    @abc.abstractclassmethod
    def reset(self):
        """
        Method resets the internal state of the object to permissive
        """
