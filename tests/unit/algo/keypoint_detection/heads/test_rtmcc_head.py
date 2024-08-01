# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMCCHead."""

import pytest
import torch
from otx.algo.keypoint_detection.heads.rtmcc_head import RTMCCHead
from otx.algo.keypoint_detection.losses.kl_discret_loss import KLDiscretLoss
from otx.algo.keypoint_detection.utils.simcc_label import SimCCLabel
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.base import ImageInfo
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
        keypoint_x_labels = [torch.randn((1, 17, 384)) for _ in range(batch_size)]
        keypoint_y_labels = [torch.randn((1, 17, 512)) for _ in range(batch_size)]
        keypoint_weights = [torch.randn((1, 17)) for _ in range(batch_size)]
        return KeypointDetBatchDataEntity(
            batch_size=batch_size,
            images=tv_tensors.Image(data=torch.randn((batch_size, 3, 192, 256))),
            imgs_info=img_infos,
            keypoint_x_labels=keypoint_x_labels,
            keypoint_y_labels=keypoint_y_labels,
            keypoint_weights=keypoint_weights,
            bboxes=[],
            labels=[],
            keypoints=[],
            keypoints_visible=[],
        )

    @pytest.fixture()
    def fxt_rtmcc_head(self) -> RTMCCHead:
        codec = SimCCLabel(
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False,
        )
        return RTMCCHead(
            out_channels=17,
            in_channels=384,
            input_size=codec.input_size,
            in_featuremap_size=tuple([s // 32 for s in codec.input_size]),
            simcc_split_ratio=codec.simcc_split_ratio,
            final_layer_kernel_size=7,
            loss=KLDiscretLoss(use_target_weight=True, beta=10.0, label_softmax=True),
            decoder=codec,
            gau_cfg={
                "hidden_dims": 256,
                "s": 128,
                "expansion_factor": 2,
                "act_fn": "SiLU",
                "use_rel_bias": False,
                "pos_enc": False,
            },
            train_cfg={},
            test_cfg={},
        )

    def test_forward(self, fxt_rtmcc_head, fxt_features) -> None:
        pred_x, pred_y = fxt_rtmcc_head(fxt_features)
        assert pred_x.shape[1] == fxt_rtmcc_head.out_channels
        assert pred_x.shape[2] == fxt_rtmcc_head.decoder.input_size[0] * fxt_rtmcc_head.decoder.simcc_split_ratio
        assert pred_y.shape[1] == fxt_rtmcc_head.out_channels
        assert pred_y.shape[2] == fxt_rtmcc_head.decoder.input_size[1] * fxt_rtmcc_head.decoder.simcc_split_ratio

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
            assert isinstance(pred, InstanceData)
            assert hasattr(pred, "keypoints")
            assert hasattr(pred, "scores")
            assert hasattr(pred, "keypoints_x_label")
            assert hasattr(pred, "keypoints_y_label")
