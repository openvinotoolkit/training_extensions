"""Initial file for mmdetection heads."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .cross_dataset_detector_head import CrossDatasetDetectorHead
from .custom_anchor_generator import SSDAnchorGeneratorClustered
from .custom_atss_head import CustomATSSHead, CustomATSSHeadTrackingLossDynamics
from .custom_dino_head import CustomDINOHead
from .custom_fcn_mask_head import CustomFCNMaskHead
from .custom_retina_head import CustomRetinaHead
from .custom_roi_head import CustomRoIHead
from .custom_rpn_head import CustomRPNHead
from .custom_ssd_head import CustomSSDHead
from .custom_vfnet_head import CustomVFNetHead
from .custom_yolox_head import CustomYOLOXHead
from .detr_head import DETRHeadExtension

__all__ = [
    "CrossDatasetDetectorHead",
    "SSDAnchorGeneratorClustered",
    "CustomATSSHead",
    "CustomDINOHead",
    "CustomFCNMaskHead",
    "CustomRetinaHead",
    "CustomSSDHead",
    "CustomRoIHead",
    "CustomVFNetHead",
    "CustomYOLOXHead",
    "DETRHeadExtension",
    "CustomRPNHead",
    # Loss dynamics tracking
    "CustomATSSHeadTrackingLossDynamics",
]
