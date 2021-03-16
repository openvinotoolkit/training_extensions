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


class IMonitor:
    """
    Interface for providing Tensorboard-style logging and performance logging
    """

    @abc.abstractclassmethod
    def add_scalar(self, capture: str, value: float, timestamp: int):
        """
        Similar to Tensorboard method that allows to log information about named scalar variables
        """
        pass

    @abc.abstractclassmethod
    def on_train_batch_begin(self):
        """
        Method starts timer that measures batch forward-backward time during training
        """
        pass

    @abc.abstractclassmethod
    def on_train_batch_end(self):
        """
        Method stops timer that measures batch forward-backward time during training
        """
        pass

    @abc.abstractclassmethod
    def on_test_batch_begin(self):
        """
        Method starts timer that measures batch forward-backward time during evaluation
        """
        pass

    @abc.abstractclassmethod
    def on_test_batch_end(self):
        """
        Method stops timer that measures batch forward-backward time during evaluation
        """
        pass

    @abc.abstractclassmethod
    def on_train_begin(self):
        """
        Method notifies the monitor that training has begun
        """
        pass

    @abc.abstractclassmethod
    def on_train_end(self):
        """
        Method notifies the monitor that training has finished
        """
        pass

    @abc.abstractclassmethod
    def on_train_epoch_begin(self, epoch: int):
        """
        Method notifies the monitor that the next training epoch has begun
        """
        pass

    @abc.abstractclassmethod
    def on_train_epoch_end(self, epoch: int):
        """
        Method notifies the monitor that training epoch has finished
        """
        pass
