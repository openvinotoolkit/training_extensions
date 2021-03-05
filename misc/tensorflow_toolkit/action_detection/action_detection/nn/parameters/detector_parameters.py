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

import numpy as np

from action_detection.nn.data.augmentation import AugmentFactory, BrightnessAugmentor, SaturationAugmentor, \
    DetectionAugmentation
from action_detection.nn.data.core import ImageSize
from action_detection.nn.models import SSDHeadDesc
from action_detection.nn.parameters.common import AttributedDict as Dict
from action_detection.nn.parameters.model_parameters import ModelParams


class DetectorParams(ModelParams):
    """Class to control the detection model specific parameters.
    """

    @staticmethod
    def _scale_anchors(anchors, image_height, image_width):
        """Converts anchors from normalized format to the target.

        :param anchors: List of anchors
        :param image_height: Target image height
        :param image_width: Target image width
        :return: List of converted anchors
        """

        anchors = np.array(anchors, dtype=np.float32)
        return np.stack([anchors[:, 0] * image_height, anchors[:, 1] * image_width], axis=1)

    def _configure_params(self):
        """Returns the parameters for detection network.

        :return: Parameters
        """

        lr_params = Dict(boundaries=[int(e * self._epoch_num_steps) for e in self._config_values.LR_EPOCH_DROPS],
                         values=self._config_values.LR_EPOCH_VALUES)
        mbox_params = Dict(threshold=self._config_values.MBOX_THRESHOLD,
                           variance=self._config_values.VARIANCE,
                           bg_class=self._config_values.BG_CLASS_ID,
                           neg_factor=self._config_values.MBOX_NEG_FACTOR,
                           cl_weight=self._config_values.MBOX_CL_WEIGHTS,
                           entropy_weight=self._config_values.MBOX_ENTROPY_WEIGHT,
                           max_num_samples_per_gt=self._config_values.MBOX_MAX_NUM_MATCHES_PER_GT,
                           matches_drop_ratio=self._config_values.MBOX_MATCHES_DROP_RATIO,
                           instance_normalization=self._config_values.MBOX_DO_INSTANCE_NORMALIZATION,
                           comp_loss_max_num_samples=self._config_values.MBOX_COMPACTNESS_LOSS_MAX_NUM_SAMPLES,
                           repulsion_loss_weight=self._config_values.MBOX_REPULSION_LOSS_WEIGHT,
                           focal_alpha=self._config_values.MBOX_FOCAL_ALPHA,
                           focal_gamma=self._config_values.MBOX_FOCAL_GAMMA,
                           gh_num_bins=self._config_values.MBOX_GRADIENT_HARMONIZED_LOSS_NUM_BINS)

        image_size = ImageSize(*self._config_values.IMAGE_SIZE)

        tuple_augmentation = DetectionAugmentation(
            self._config_values.FREE_PROB, self._config_values.EXPAND_PROB,
            self._config_values.CROP_PROB, self._config_values.MAX_EXPAND_RATIO,
            self._config_values.CROP_SCALE_DELTA, self._config_values.CROP_SCALE_LIMITS,
            self._config_values.CROP_SHIFT_DELTA, float(image_size.h) / float(image_size.w))
        image_augmentation = AugmentFactory() \
            .add(BrightnessAugmentor(self._config_values.BRIGHTNESS_DELTA)) \
            .add(SaturationAugmentor(self._config_values.SATURATION_LIMITS))

        head_params = []
        for scale in self._config_values.NORMALIZED_ANCHORS:
            head_params.append(SSDHeadDesc(scale=scale,
                                           internal_size=self._config_values.INTERNAL_HEAD_SIZES[scale],
                                           num_classes=self._config_values.NUM_CLASSES,
                                           anchors=self._scale_anchors(self._config_values.NORMALIZED_ANCHORS[scale],
                                                                       image_size.h, image_size.w),
                                           clip=False,
                                           offset=0.5))

        return Dict(max_num_objects_per_image=self._config_values.MAX_NUM_DETECTIONS_PER_IMAGE,
                    num_classes=self._config_values.NUM_CLASSES,
                    bg_class=self._config_values.BG_CLASS_ID,
                    tuple_augmentation=tuple_augmentation,
                    image_augmentation=image_augmentation,
                    lr_params=lr_params,
                    mbox_params=mbox_params,
                    head_params=head_params,
                    labels_map=self._config_values.LABELS_MAP)
