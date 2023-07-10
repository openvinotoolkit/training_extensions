"""Fast RCNN for Action Detection."""

# Copyright (C) 2022 Intel Corporation
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

import torch
from mmcv.runner import load_checkpoint
from mmcv.utils import ConfigDict
from mmdeploy.core import FUNCTION_REWRITER
from mmdet.models import DETECTORS
from mmdet.models.builder import build_detector
from mmdet.models.detectors import FastRCNN
from torch import nn

from otx.algorithms.action.configs.detection.base.faster_rcnn_config import (
    faster_rcnn,
    faster_rcnn_pretrained,
)


class ONNXPool3D(nn.Module):
    """3D pooling method for onnx export.

    ONNX dose not support dynamic pooling, therefore pooling operation should be changed into static way
    """

    def __init__(self, dim, pool_type):
        # TODO This is tempral solution to export two-stage action detection model.
        # This should be re-visited after fixing CVS-104657
        super().__init__()
        self.dim = dim
        if pool_type == "avg":
            self.pool = torch.mean
        else:
            self.pool = torch.max
        self.pool_type = pool_type

    def forward(self, x):
        """Forward method."""
        if self.dim == "temporal":
            if self.pool_type == "avg":
                return self.pool(x, 2, keepdim=True)
            return self.pool(x, 2, keepdim=True)[0]
        if self.pool_type == "avg":
            return self.pool(x, (3, 4), keepdim=True)
        return self.pool(self.pool(x, 3, keepdim=True)[0], 4, keepdim=True)[0]


@DETECTORS.register_module()
class AVAFastRCNN(FastRCNN):
    """Implementation of `Fast R-CNN` for Action Detection.

    Add forward_infer function for inference without pre-proposals
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self, backbone, roi_head, train_cfg, test_cfg, neck=None, pretrained=None):
        super(FastRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.detector = None

    def patch_for_export(self):
        """Patch mmdetection's FastRCNN for exporting to onnx."""
        self._add_detector()
        self._patch_pools()

    def _add_detector(self):
        """Add Person Detector for inference.

        Action classification backbone + Fast RCNN structure use pre-proposals instead outputs from person detector
        This saves training and evaluation time. However, this method doesn't support inference for the dataset
        without pre-proposals. Therefore, when the model infers to dataset without pre-proposal, person detector
        should be added to action detector.
        """
        detector = ConfigDict(faster_rcnn)
        self.detector = build_detector(detector)
        ckpt = load_checkpoint(self.detector, faster_rcnn_pretrained, map_location="cpu")
        self.detector.CLASSES = ckpt["meta"]["CLASSES"]
        if self.detector.CLASSES[0] != "person":
            raise Exception(
                f"Person detector should have person as the first category, but got {self.detector.CLASSES}"
            )

    def _patch_pools(self):
        """Patch pooling functions for ONNX export.

        AVAFastRCNN's bbox head has pooling funcitons, which contain dynamic shaping.
        This funciton changes those pooling functions from dynamic shaping to static shaping.
        """
        self.roi_head.bbox_head.temporal_pool = ONNXPool3D("temporal", self.roi_head.bbox_head.temporal_pool_type)
        self.roi_head.bbox_head.spatial_pool = ONNXPool3D("spatial", self.roi_head.bbox_head.spatial_pool_type)

    # pylint: disable=no-self-argument
    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.action.adapters.mmaction.models.detectors.fast_rcnn.AVAFastRCNN.forward"
    )
    def forward_infer(ctx, self, imgs, img_metas):
        """Forward function for inference without pre-proposal."""
        clip_len = imgs.shape[2]
        img = imgs[:, :, int(clip_len / 2), :, :]
        det_bboxes, det_labels = self.detector.simple_test(img, img_metas[0])
        prediction = [det_bboxes[0][det_labels[0] == 0]]
        prediction = self.simple_test(imgs, img_metas[0], proposals=prediction)
        return prediction
