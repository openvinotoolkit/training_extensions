"""Unit Test for otx.algorithms.action.adapters.mmaction.data.cls_dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch
from mmcv.utils import Config, ConfigDict
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from torch import nn

from otx.algorithms.action.adapters.mmaction.models.detectors.fast_rcnn import (
    AVAFastRCNN,
    ONNXPool3D,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockDetector(nn.Module):
    """Mock class for person detector."""

    def simple_test(self, *args, **kwargs):
        """Return dummy person detection results."""

        sample_det_bboxes = torch.Tensor([[0.0, 0.0, 1.0, 1.0, 1.0]] * 100).unsqueeze(0)
        sample_det_labels = torch.ones(1, 100)
        sample_det_labels[0][0] = 0
        return sample_det_bboxes, sample_det_labels


class TestONNXPool3d:
    """Test ONNXPool3D class.

    1. Check every possible ONNXPool3D generation
    2. Check every possible ONNXPool3D actuall pooling input tensor as expected
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.temporal_avg_pool = ONNXPool3D("temporal", "avg")
        self.temporal_max_pool = ONNXPool3D("temporal", "max")
        self.spatial_avg_pool = ONNXPool3D("spatial", "avg")
        self.spatial_max_pool = ONNXPool3D("spatial", "max")

    @e2e_pytest_unit
    def test_init(self) -> None:
        """Test __init__ function."""

        self.temporal_avg_pool.pool == torch.mean
        self.temporal_max_pool.pool == torch.max
        self.spatial_avg_pool.pool == torch.mean
        self.spatial_max_pool.pool == torch.max

    @e2e_pytest_unit
    def test_forward(self) -> None:
        """Test forward function."""

        sample_input = torch.randn(1, 100, 8, 8, 8)
        output = self.temporal_avg_pool(sample_input)
        assert list(output.shape) == [1, 100, 1, 8, 8]
        output = self.temporal_max_pool(sample_input)
        assert list(output.shape) == [1, 100, 1, 8, 8]
        output = self.spatial_avg_pool(sample_input)
        assert list(output.shape) == [1, 100, 8, 1, 1]
        output = self.spatial_max_pool(sample_input)
        assert list(output.shape) == [1, 100, 8, 1, 1]


class TestAVAFastRCNN:
    """Test AVAFastRCNN class.

    1. Check _add_detector function
    2. Check _patch_pools function
    3. Check forward_infer function
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model = AVAFastRCNN(
            backbone=ConfigDict(type="X3D", gamma_w=1, gamma_b=2.25, gamma_d=2.2),
            roi_head=ConfigDict(
                type="AVARoIHead",
                bbox_roi_extractor=ConfigDict(
                    type="SingleRoIExtractor3D", roi_layer_type="RoIAlign", output_size=8, with_temporal_pool=True
                ),
                bbox_head=ConfigDict(
                    type="BBoxHeadAVA", in_channels=432, num_classes=81, multilabel=False, dropout_ratio=0.5
                ),
            ),
            train_cfg=ConfigDict(
                rcnn=ConfigDict(
                    assigner=ConfigDict(type="MaxIoUAssignerAVA", pos_iou_thr=0.9, neg_iou_thr=0.9, min_pos_iou=0.9),
                    sampler=ConfigDict(
                        type="RandomSampler", num=32, pos_fraction=1, neg_pos_ub=-1, add_gt_as_proposals=True
                    ),
                    pos_weight=1.0,
                    debug=False,
                )
            ),
            test_cfg=ConfigDict(rcnn=ConfigDict(action_thr=0.002)),
        )

    @e2e_pytest_unit
    def test_patch_for_export(self, mocker) -> None:
        """Test patch_for_export function."""

        mocker.patch.object(AVAFastRCNN, "_add_detector", return_value=True)
        mocker.patch.object(AVAFastRCNN, "_patch_pools", return_value=True)
        self.model.patch_for_export()

    @e2e_pytest_unit
    def test_add_detector(self, mocker) -> None:
        """Test add_deector function.

        <Steps>
            1. Check added detector is FasterRCNN
            2. Check added detector has COCO classes
            3. Check added detector's CLASSES is properly initialized
            4. Check added detector raise exception if detector's first class is not person
        """

        mock_deploy_cfg = Config(
            dict(codebase_config=dict(type="mmdet", task="ObjectDetection"), backend_config=dict(type="openvino"))
        )
        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.models.detectors.fast_rcnn.load_checkpoint",
            return_value={"meta": {"CLASSES": ["person", "motorcycle", "car"]}},
        )
        self.model.deploy_cfg = mock_deploy_cfg
        self.model._add_detector()
        assert isinstance(self.model.detector, FasterRCNN)
        assert self.model.detector.roi_head.bbox_head.num_classes == 80
        assert self.model.detector.CLASSES == ["person", "motorcycle", "car"]

        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.models.detectors.fast_rcnn.load_checkpoint",
            return_value={"meta": {"CLASSES": ["motorcycle", "car", "person"]}},
        )
        with pytest.raises(Exception):
            self.model._add_detector()

    @e2e_pytest_unit
    def test_patch_pools(self) -> None:
        """Test _patch_pools function.

        <Steps>
            1. Check bbox_head's temporal pool is avg_pool and it pools through temporal axis
            2. Check bbox_head's spatial pool is max_pool and it pools through spatial axis
        """

        self.model._patch_pools()
        assert isinstance(self.model.roi_head.bbox_head.temporal_pool, ONNXPool3D)
        assert self.model.roi_head.bbox_head.temporal_pool.pool == torch.mean
        assert self.model.roi_head.bbox_head.temporal_pool.dim == "temporal"
        assert isinstance(self.model.roi_head.bbox_head.spatial_pool, ONNXPool3D)
        assert self.model.roi_head.bbox_head.spatial_pool.pool == torch.max
        assert self.model.roi_head.bbox_head.spatial_pool.dim == "spatial"

    @e2e_pytest_unit
    def test_forward_infer(self, mocker) -> None:
        """Test forward_infer function.

        <Steps>
            1. Prepare sample imgs and img_metas
            2. Patch model's detector to MockDetector
            3. Check amount of output bboxes and output labels are same
            4. Check output bboxes have 4 cooridnates
            5. Check output labels have all num_classes of bbox_head
        """

        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.models.heads.roi_head.is_in_onnx_export", return_value=True
        )
        height = width = 256
        sample_imgs = torch.randn(1, 3, 32, height, width)
        sample_img_metas = {
            "img_metas": [
                [
                    {
                        "ori_shape": (height, width),
                        "img_shape": (height, width),
                        "pad_shape": (height, width),
                        "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    }
                ]
            ]
        }
        self.model.detector = MockDetector()
        with torch.no_grad():
            bboxes, labels = self.model.forward_infer(self=self.model, imgs=sample_imgs, **sample_img_metas)
        assert bboxes.shape[-1] == 4
        assert labels.shape[-1] == self.model.roi_head.bbox_head.num_classes
        assert bboxes.shape[0] == labels.shape[0]
