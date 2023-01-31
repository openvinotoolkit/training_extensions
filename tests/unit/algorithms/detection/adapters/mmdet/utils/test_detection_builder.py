"""MMdet model builder."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from mmcv.utils import Config

from otx.algorithms.detection.adapters.mmdet.utils.builder import build_detector
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestDetBuilder:
    """Test class for mmdet model builder."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.cfg = Config()
        self.cfg.model = dict(
            type="ATSS",
            neck=dict(
                type="FPN",
                in_channels=[24, 32, 96, 320],
                out_channels=64,
                start_level=1,
                add_extra_convs="on_output",
                num_outs=5,
                relu_before_extra_convs=True,
            ),
            bbox_head=dict(
                type="CustomATSSHead",
                num_classes=2,
                in_channels=64,
                stacked_convs=4,
                feat_channels=64,
                anchor_generator=dict(
                    type="AnchorGenerator",
                    ratios=[1.0],
                    octave_base_scale=8,
                    scales_per_octave=1,
                    strides=[8, 16, 32, 64, 128],
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
                use_qfl=False,
                qfl_cfg=dict(
                    type="QualityFocalLoss",
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0,
                ),
            ),
            train_cfg=dict(
                assigner=dict(type="ATSSAssigner", topk=9),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.6),
                max_per_img=100,
            ),
            backbone=dict(
                type="MobileNetV2",
                out_indices=(
                    2,
                    3,
                    4,
                    5,
                ),
                frozen_stages=-1,
                norm_eval=False,
                pretrained=None,
            ),
        )
        self.ckpt_url = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/ \
                         models/object_detection/v2/mobilenet_v2-atss.pth"

    @e2e_pytest_unit
    @pytest.mark.parametrize("cfg_options", [None, Config()])
    def test_build_detector(self, cfg_options):
        """Test for mmdet model builder."""
        model = build_detector(self.cfg, checkpoint=self.ckpt_url, cfg_options=cfg_options)
        assert model is not None
