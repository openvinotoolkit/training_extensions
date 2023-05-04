# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Dict, Tuple

import pytest
import torch
from mmdet.models.builder import build_head

from otx.algorithms.detection.adapters.mmdet.models.heads import (
    CustomATSSHead,
    CustomATSSHeadTrackingLossDynamics,
)


class TestLossDynamicsTrackingHeads:
    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        torch.random.manual_seed(3003)

    @pytest.fixture
    def fxt_atss_head_with_tracking_loss(
        self, fxt_atss_head: CustomATSSHead, fxt_cfg_atss_head: Dict
    ) -> Tuple[CustomATSSHead, CustomATSSHeadTrackingLossDynamics]:
        fxt_cfg_atss_head["type"] = fxt_cfg_atss_head["type"] + "TrackingLossDynamics"

        atss_head_with_tracking_loss = build_head(fxt_cfg_atss_head)
        # Copy-paste atss_head's weights
        atss_head_with_tracking_loss.load_state_dict(fxt_atss_head.state_dict())
        return fxt_atss_head, atss_head_with_tracking_loss

    @torch.no_grad()
    def test_output_equivalance(self, fxt_atss_head_with_tracking_loss, fxt_head_input):
        atss_head, atss_head_with_tracking_loss = fxt_atss_head_with_tracking_loss
        feat, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore = fxt_head_input

        scores = atss_head_with_tracking_loss.forward(feat)
        expected_scores = atss_head.forward(feat)

        for actual, expected in zip(scores, expected_scores):
            # actual, expected are list (# of feature pyramid level)
            for a, e in zip(actual, expected):
                assert torch.allclose(a, e)

        losses = atss_head_with_tracking_loss.loss(*scores, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        expected_losses = atss_head.loss(*expected_scores, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

        for actual, expected in zip(losses.values(), expected_losses.values()):
            # actual, expected are list (# of feature pyramid level)
            for a, e in zip(actual, expected):
                assert torch.allclose(a, e)
