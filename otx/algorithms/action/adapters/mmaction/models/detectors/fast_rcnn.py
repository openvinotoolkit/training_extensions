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

from functools import partial

from mmcv import ConfigDict
from mmcv.runner import load_checkpoint
from mmdet.models import DETECTORS
from mmdet.models.builder import build_detector
from mmdet.models.detectors import FastRCNN
from torch import nn
from torch.onnx import is_in_onnx_export

from .faster_rcnn_config import faster_rcnn, faster_rcnn_pretrained


class ONNXPool3D(nn.Module):
    """3D pooling method for onnx export.

    ONNX dose not support dynamic pooling, therefore pooling operation should be changed into static way
    """

    def __init__(self, dim, pool_type):
        super().__init__()
        self.dim = dim
        if pool_type == "avg":
            self.pool = nn.functional.avg_pool3d
        else:
            self.pool = nn.functional.max_pool3d

    def forward(self, x):
        """Forward method."""
        size_array = [int(s) for s in x.size()[2:]]
        if self.dim == "temporal":
            kernel_size = [size_array[0], 1, 1]
        else:
            kernel_size = [1, size_array[1], size_array[2]]
        return self.pool(x, kernel_size)


@DETECTORS.register_module()
class AVAFastRCNN(FastRCNN):
    """Implementation of `Fast R-CNN` for Action Detection.

    Add forwrad_infer function for inference without pre-proposals
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

    def add_detector(self):
        """Add Human Detector for inference.

        Action classification backbone + Fast RCNN structure use pre-proposals instead outputs from human detector
        This saves training and evaluation time. However this method don't support inference for the dataset
        without pre-proposals. Therefore when the model infer to dataset without pre-proposal, human detector
        should be added to action detector
        """
        detector = ConfigDict(faster_rcnn)
        self.detector = build_detector(detector)
        self.detector.forward = partial(self.detector.forward_test, postprocess=False)
        ckpt = load_checkpoint(self.detector, faster_rcnn_pretrained, map_location="cpu")
        self.detector.CLASSES = ckpt["meta"]["CLASSES"]
        if self.detector.CLASSES[0] != "person":
            raise Exception(f"Person detector should has person as the first category, but got {self.detector.CLASSES}")

    def patch_pools(self):
        """Patch pooling functions for ONNX export.

        AVAFastRCNN's bbox head has pooling funcitons, which contain dynamic shaping.
        This funciton change those pooling functions from dynamic shaping to static shaping.
        """
        self.roi_head.bbox_head.temporal_pool = ONNXPool3D(self.roi_head.bbox_head.temporal_pool_type, "temporal")
        self.roi_head.bbox_head.spatial_pool = ONNXPool3D(self.roi_head.bbox_head.spatial_pool_type, "spatial")

    def forward_infer(self, imgs, img_metas):
        """Forward function for inference without pre-proposal."""
        clip_len = imgs[0].shape[2]
        img = imgs[0][:, :, int(clip_len / 2), :, :]
        if is_in_onnx_export():
            det_bboxes, det_labels = self.detector([img], img_metas)[0]
        else:
            det_bboxes, det_labels = self.detector([img], img_metas)
        prediction = [[det_bboxes[0][det_labels[0] == 0]]]
        prediction = self.forward_test(imgs, img_metas, proposals=prediction)
        return prediction
