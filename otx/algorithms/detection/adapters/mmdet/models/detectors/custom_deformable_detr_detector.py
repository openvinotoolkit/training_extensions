"""OTX Deformable DETR Class for mmdetection detectors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.deformable_detr import DeformableDETR


@DETECTORS.register_module()
class CustomDeformableDETR(DeformableDETR):
    """Custom Deformable DETR with task adapt.

    Deformable DETR does not support task adapt, so it just take task_adpat argument.
    """

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_adapt = task_adapt
