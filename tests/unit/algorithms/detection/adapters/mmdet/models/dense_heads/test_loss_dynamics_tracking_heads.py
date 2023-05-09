# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from copy import deepcopy
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
    def fxt_head_pair(
        self, request: pytest.FixtureRequest
    ) -> Tuple[CustomATSSHead, CustomATSSHeadTrackingLossDynamics]:
        fxt_cfg_name = request.param
        fxt_cfg_head = request.getfixturevalue(fxt_cfg_name)
        fxt_cfg_head_tracking_loss_dyns = deepcopy(fxt_cfg_head)
        fxt_cfg_head_tracking_loss_dyns["type"] = fxt_cfg_head["type"] + "TrackingLossDynamics"

        head = build_head(fxt_cfg_head)
        head_tracking_loss_dyns = build_head(fxt_cfg_head_tracking_loss_dyns)
        # Copy-paste the original head's weights
        head_tracking_loss_dyns.load_state_dict(head.state_dict())
        return head, head_tracking_loss_dyns

    @torch.no_grad()
    @pytest.mark.parametrize(
        "fxt_head_pair",
        [
            "fxt_cfg_atss_head",
            "fxt_cfg_ssd_head",
            "fxt_cfg_vfnet_head",
            "fxt_cfg_yolox_head",
        ],
        indirect=True,
    )
    def test_output_equivalance(self, fxt_head_pair, fxt_head_input):
        head, head_with_tracking_loss = fxt_head_pair
        feat, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore = fxt_head_input

        scores = head_with_tracking_loss.forward(feat)
        expected_scores = head.forward(feat)

        for actual, expected in zip(scores, expected_scores):
            # actual, expected are list (# of feature pyramid level)
            for a, e in zip(actual, expected):
                assert torch.allclose(a, e)

        losses = head_with_tracking_loss.loss(*scores, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        expected_losses = head.loss(*expected_scores, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

        for actual, expected in zip(losses.values(), expected_losses.values()):
            # actual, expected are list (# of feature pyramid level)
            if isinstance(actual, list) and isinstance(expected, list):
                for a, e in zip(actual, expected):
                    assert torch.allclose(a, e)
            else:
                assert torch.allclose(actual, expected)
