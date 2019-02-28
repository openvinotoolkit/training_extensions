"""
 Copyright (c) 2018 Intel Corporation
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
from abc import abstractmethod
import torch.nn as nn


class ModelInterface(nn.Module):
    """Abstract class for models"""

    @abstractmethod
    def set_dropout_ratio(self, ratio):
        """Sets dropout ratio of the model"""

    @abstractmethod
    def get_input_res(self):
        """Returns input resolution"""


from .rmnet_angular import RMNetAngular
from .mobilefacenet import MobileFaceNet
from .landnet import LandmarksNet
from .resnet_angular import ResNetAngular
from .se_resnet_angular import SEResNetAngular
from .shufflenet_v2_angular import ShuffleNetV2Angular


models_backbones = {'rmnet': RMNetAngular, 'mobilenet': MobileFaceNet, 'resnet': ResNetAngular,
                    'shufflenetv2': ShuffleNetV2Angular, 'se_resnet': SEResNetAngular}

models_landmarks = {'landnet': LandmarksNet}
