"""OTX Lite-DINO Class for object detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import DETECTORS

from otx.algorithms.detection.adapters.mmdet.models.detectors import CustomDINO
from otx.utils.logger import get_logger

logger = get_logger()


@DETECTORS.register_module()
class CustomLiteDINO(CustomDINO):
    """Custom Lite-DINO <https://arxiv.org/pdf/2303.07335.pdf> for object detection."""

    def load_state_dict_pre_hook(self, model_classes, ckpt_classes, ckpt_dict, *args, **kwargs):
        """Modify official lite dino version's weights before weight loading."""
        super(CustomDINO, self).load_state_dict_pre_hook(model_classes, ckpt_classes, ckpt_dict, *args, *kwargs)
