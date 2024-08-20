# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMCCHead."""

import pytest
import torch
from otx.algo.keypoint_detection.heads.rtmcc_head import RTMCCHead
from otx.algo.keypoint_detection.losses.kl_discret_loss import KLDiscretLoss
from otx.core.data.entity.base import BboxInfo, ImageInfo
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity
from torchvision import tv_tensors


class TestRTMCCHead:
    @pytest.fixture()
    def fxt_features(self):
        batch_size = 2
        in_channels = 384  # Match the in_channels of the rtmdet_sep_bn_head fixture
        input_size = (192, 256)
        return [torch.rand(batch_size, in_channels, input_size[0] // 32, input_size[1] // 32)]

    @pytest.fixture()
    def fxt_gt_entity(self):
        batch_size = 2
        img_infos = [ImageInfo(img_idx=i, img_shape=(192, 256), ori_shape=(192, 256)) for i in range(batch_size)]
        keypoints = torch.randn((batch_size, 17, 2))
        keypoints_visible = torch.randn((batch_size, 17))
        return KeypointDetBatchDataEntity(
            batch_size=batch_size,
            images=tv_tensors.Image(data=torch.randn((batch_size, 3, 192, 256))),
            imgs_info=img_infos,
            bbox_info=BboxInfo(center=(96, 128), scale=(1, 1), rotation=0),
            bboxes=[],
            labels=[],
            keypoints=keypoints,
            keypoints_visible=keypoints_visible,
        )

    @pytest.fixture()
    def fxt_rtmcc_head(self) -> RTMCCHead:
        return RTMCCHead(
            out_channels=17,
            in_channels=384,
            input_size=(256, 192),
            in_featuremap_size=(6, 8),
            simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            loss=KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True),
            decoder_cfg={
                "input_size": (256, 192),
                "simcc_split_ratio": 2.0,
                "sigma": (4.9, 5.66),
                "normalize": False,
                "use_dark": False,
            },
            gau_cfg={
                "num_token": 17,
                "in_token_dims": 256,
                "out_token_dims": 256,
                "s": 128,
                "expansion_factor": 2,
                "act_fn": "SiLU",
                "use_rel_bias": False,
                "pos_enc": False,
            },
        )

    def test_forward(self, fxt_rtmcc_head, fxt_features) -> None:
        pred_x, pred_y = fxt_rtmcc_head(fxt_features)
        assert pred_x.shape[1] == fxt_rtmcc_head.out_channels
        assert pred_x.shape[2] == fxt_rtmcc_head.codec.input_size[1] * fxt_rtmcc_head.codec.simcc_split_ratio
        assert pred_y.shape[1] == fxt_rtmcc_head.out_channels
        assert pred_y.shape[2] == fxt_rtmcc_head.codec.input_size[0] * fxt_rtmcc_head.codec.simcc_split_ratio

    def test_loss(self, fxt_rtmcc_head, fxt_features, fxt_gt_entity) -> None:
        losses = fxt_rtmcc_head.loss(
            fxt_features,
            fxt_gt_entity,
        )
        assert "loss_kpt" in losses
        assert "loss_pose" in losses

    def test_predict(self, fxt_rtmcc_head, fxt_features) -> None:
        preds = fxt_rtmcc_head.predict(fxt_features)
        for pred in preds:
            assert isinstance(pred, tuple)
            assert len(pred) == 2
