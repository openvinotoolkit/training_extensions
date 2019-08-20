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
from functools import partial
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
from .se_resnet_angular import SEResNetAngular
from .shufflenet_v2_angular import ShuffleNetV2Angular
from .backbones.se_resnet import se_resnet50, se_resnet101, se_resnet152
from .backbones.resnet import resnet50
from .backbones.se_resnext import se_resnext50, se_resnext101, se_resnext152


models_backbones = {'rmnet': RMNetAngular,
                    'mobilenetv2': MobileFaceNet,
                    'mobilenetv2_2x': partial(MobileFaceNet, width_multiplier=2.0),
                    'mobilenetv2_1_5x': partial(MobileFaceNet, width_multiplier=1.5),
                    'resnet50': partial(SEResNetAngular, base=resnet50),
                    'se_resnet50': partial(SEResNetAngular, base=se_resnet50),
                    'se_resnet101': partial(SEResNetAngular, base=se_resnet101),
                    'se_resnet152': partial(SEResNetAngular, base=se_resnet152),
                    'se_resnext50': partial(SEResNetAngular, base=se_resnext50),
                    'se_resnext101': partial(SEResNetAngular, base=se_resnext101),
                    'se_resnext152': partial(SEResNetAngular, base=se_resnext152),
                    'shufflenetv2': ShuffleNetV2Angular}

models_landmarks = {'landnet': LandmarksNet}
