# Copyright (C) 2019 Intel Corporation
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

from action_detection.nn.data.augmentation import ClassifierAugmentation
from action_detection.nn.data.core import ImageSize
from action_detection.nn.parameters.common import AttributedDict as Dict
from action_detection.nn.parameters.model_parameters import ModelParams


class ClassifierParams(ModelParams):
    """Class to control the classification model specific parameters.
    """

    def _configure_params(self):
        """Returns the parameters for classification network.

        :return: Model parameters
        """

        lr_params = Dict(boundaries=[int(e * self._epoch_num_steps) for e in self._config_values.LR_EPOCH_DROPS],
                         values=self._config_values.LR_EPOCH_VALUES)

        image_size = ImageSize(*self._config_values.IMAGE_SIZE)

        augmentation = ClassifierAugmentation(self._config_values.CROP_PROB,
                                              self._config_values.CROP_SCALE_LIMITS,
                                              self._config_values.CROP_VAR_LIMITS,
                                              self._config_values.BRIGHTNESS_DELTA,
                                              self._config_values.SATURATION_LIMITS,
                                              float(image_size.h) / float(image_size.w))

        return Dict(val_image_side=self._config_values.VAL_IMAGE_SIDE_SIZE,
                    val_central_fraction=self._config_values.VAL_CENTRAL_FRACTION,
                    num_classes=self._config_values.NUM_CLASSES,
                    image_augmentation=augmentation,
                    lr_params=lr_params,
                    keep_probe=self._config_values.DROPOUT_KEEP_PROBE)
