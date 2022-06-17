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

from torch.utils.tensorboard import SummaryWriter
from ote_sdk.utils.argument_checks import (
    DirectoryPathCheck,
    check_input_parameters_type,
)


class IMetricsMonitor(metaclass=abc.ABCMeta):
    """
    Interface for providing Tensorboard-style logging
    """

    @abc.abstractmethod
    def add_scalar(self, capture: str, value: float, timestamp: int):
        """
        Similar to Tensorboard method that allows to log values of named scalar variables
        """

    @abc.abstractmethod
    def close(self):
        """
        Flushes the collected scalar values to log, closes corresponding file,
        then resets the monitor to default state
        """


class IStopCallback(metaclass=abc.ABCMeta):
    """
    Interface for wrapping a stop signal.
    By default an object implementing this interface should be in permissive state
    """

    @abc.abstractmethod
    def check_stop(self) -> bool:
        """
        Method returns True if the object of this class is in stopping state,
        otherwise it returns False
        """

    @abc.abstractmethod
    def reset(self):
        """
        Method resets the internal state of the object to permissive
        """

class StopCallback(IStopCallback):
    def __init__(self):
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def check_stop(self):
        return self.stop_flag

    def reset(self):
        self.stop_flag = False


class MetricsMonitor(IMetricsMonitor):
    @check_input_parameters_type({"log_dir": DirectoryPathCheck})
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.tb = None

    @check_input_parameters_type()
    def add_scalar(self, capture: str, value: float, timestamp: int):
        if not self.tb:
            self.tb = SummaryWriter(self.log_dir)
        self.tb.add_scalar(capture, value, timestamp)

    def close(self):
        if self.tb:
            self.tb.close()
            self.tb = None


class DefaultStopCallback(IStopCallback):
    def stop(self):
        pass

    def check_stop(self):
        pass

    def reset(self):
        pass


class DefaultMetricsMonitor(IMetricsMonitor):
    def __init__(self):
        self.metrics_dict = {}

    @check_input_parameters_type()
    def add_scalar(self, capture: str, value: float, timestamp: int):
        if capture in self.metrics_dict:
            self.metrics_dict[capture].append((timestamp, value))
        else:
            self.metrics_dict[capture] = [(timestamp, value),]

    def get_metric_keys(self):
        return self.metrics_dict.keys()

    @check_input_parameters_type()
    def get_metric_values(self, capture: str):
        return [item[1] for item in self.metrics_dict[capture]]

    @check_input_parameters_type()
    def get_metric_timestamps(self, capture: str):
        return [item[0] for item in self.metrics_dict[capture]]

    def close(self):
        pass
