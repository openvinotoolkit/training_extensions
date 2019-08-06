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

from action_detection.nn.data.augmentation import AugmentFactory, BrightnessAugmentor, SaturationAugmentor, \
    DetectionAugmentation
from action_detection.nn.data.core import ImageSize
from action_detection.nn.models import SSDHeadDesc
from action_detection.nn.parameters.common import AttributedDict as Dict
from action_detection.nn.parameters.detector_parameters import DetectorParams


class ActionParams(DetectorParams):
    """Class to control the action model specific parameters.
    """

    def _configure_params(self):
        """Returns the parameters for action network.

        :return: Parameters
        """

        lr_params = Dict(schedule=self._config_values.LR_SCHEDULE,
                         boundaries=[int(e * self._epoch_num_steps) for e in self._config_values.LR_EPOCH_DROPS],
                         values=self._config_values.LR_EPOCH_VALUES,
                         init_value=self._config_values.LR_INIT_VALUE,
                         first_decay_steps=int(self._config_values.LR_FIRST_DECAY_EPOCH * self._epoch_num_steps),
                         t_mul=self._config_values.LR_T_MUL,
                         m_mul=self._config_values.LR_M_MUL,
                         alpha=self._config_values.LR_ALPHA)
        mbox_params = Dict(threshold=self._config_values.MBOX_THRESHOLD,
                           variance=self._config_values.VARIANCE,
                           bg_class=self._config_values.DETECTION_BG_CLASS_ID,
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
                                           num_classes=self._config_values.DETECTION_NUM_CLASSES,
                                           anchors=self._scale_anchors(self._config_values.NORMALIZED_ANCHORS[scale],
                                                                       image_size.h, image_size.w),
                                           clip=False,
                                           offset=0.5))

        action_params = Dict(num_actions=len(self._config_values.VALID_ACTION_NAMES),
                             embedding_size=self._config_values.ACTION_EMBEDDING_SIZE,
                             undefined_action_id=self._config_values.UNDEFINED_ACTION_ID,
                             num_centers_per_action=self._config_values.NUM_CENTERS_PER_ACTION,
                             scale_start=self._config_values.SCALE_START_VALUE,
                             scale_end=self._config_values.SCALE_END_VALUE,
                             scale_num_steps=int(self._config_values.SCALE_NUM_EPOCHS * self._epoch_num_steps),
                             scale_power=self._config_values.SCALE_POWER,
                             max_entropy_weight=self._config_values.ACTION_ENTROPY_WEIGHT,
                             focal_alpha=self._config_values.ACTION_FOCAL_ALPHA,
                             focal_gamma=self._config_values.ACTION_FOCAL_GAMMA,
                             glob_pull_push_margin=self._config_values.GLOB_PULL_PUSH_MARGIN,
                             local_push_margin=self._config_values.LOCAL_PUSH_MARGIN,
                             num_samples=self._config_values.NUM_SAMPLES_PER_CLASS,
                             local_push_top_k=self._config_values.LOCAL_PUSH_LOSS_TOP_K,
                             weight_limits=self._config_values.ADAPTIVE_WEIGHT_LIMITS,
                             ce_loss_weight=self._config_values.CE_LOSS_WEIGHT,
                             auxiliary_loss_weight=self._config_values.AUXILIARY_LOSS_WEIGHT,
                             matches_threshold=self._config_values.MATCHES_THRESHOLD,
                             max_num_samples_per_gt=self._config_values.MAX_NUM_MATCHES_PER_GT,
                             sample_matches_drop_ratio=self._config_values.MATCHES_DROP_RATIO,
                             glob_pull_push_loss_top_k=self._config_values.GLOB_PULL_PUSH_LOSS_TOP_K,
                             center_loss_top_k=self._config_values.CENTER_LOSS_TOP_K,
                             center_loss_weight=self._config_values.CENTER_LOSS_WEIGHT,
                             num_bins=self._config_values.ACTION_GRADIENT_HARMONIZED_LOSS_NUM_BINS)

        action_names_map = {i: v for i, v in enumerate(self._config_values.VALID_ACTION_NAMES)}

        return Dict(max_num_objects_per_image=self._config_values.MAX_NUM_DETECTIONS_PER_IMAGE,
                    num_classes=self._config_values.DETECTION_NUM_CLASSES,
                    bg_class=self._config_values.DETECTION_BG_CLASS_ID,
                    num_actions=len(self._config_values.VALID_ACTION_NAMES),
                    tuple_augmentation=tuple_augmentation,
                    image_augmentation=image_augmentation,
                    lr_params=lr_params,
                    mbox_params=mbox_params,
                    head_params=head_params,
                    action_params=action_params,
                    labels_map=self._config_values.ACTIONS_MAP,
                    valid_actions=self._config_values.VALID_ACTION_NAMES,
                    ignore_classes=self._config_values.IGNORE_CLASSES,
                    use_class_balancing=self._config_values.USE_CLASS_BALANCING,
                    det_conf=self._config_values.DETECTION_CONFIDENCE,
                    action_conf=self._config_values.ACTION_CONFIDENCE,
                    action_colors_map=self._config_values.ACTION_COLORS_MAP,
                    action_names_map=action_names_map,
                    undefined_action_name=self._config_values.UNDEFINED_ACTION_NAME,
                    undefined_action_color=self._config_values.UNDEFINED_ACTION_COLOR)
