"""Test for CustomDeformableDETR Detector."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdet.models.builder import build_detector

from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_deformable_detr_detector import (
    CustomDeformableDETR,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomDeformableDETR:
    @e2e_pytest_unit
    def test_custom_deformable_detr_build(self, fxt_cfg_custom_deformable_detr):
        model = build_detector(fxt_cfg_custom_deformable_detr)
        assert isinstance(model, CustomDeformableDETR)
        assert model.cls_layers is not None

    @e2e_pytest_unit
    def test_custom_deformable_detr_load_state_dict_pre_hook(self, fxt_cfg_custom_deformable_detr, mocker):
        model = build_detector(fxt_cfg_custom_deformable_detr)
        chkpt_classes = ["person", "car"]
        model_classes = ["tree", "car", "person"]
        chkpt_dict = {
            "bbox_head.cls_branches.weight": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                ]
            ),
            "bbox_head.cls_branches.bias": torch.tensor(
                [
                    [1],
                    [2],
                ]
            ),
        }
        model_dict = {
            "bbox_head.cls_branches.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                ]
            ),
            "bbox_head.cls_branches.bias": torch.tensor(
                [
                    [3],
                    [4],
                    [5],
                ]
            ),
        }
        gt_dict = {
            "bbox_head.cls_branches.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                ]
            ),
            "bbox_head.cls_branches.bias": torch.tensor(
                [
                    [3],
                    [2],
                    [1],
                ]
            ),
        }

        mocker.patch.object(model, "state_dict", return_value=model_dict)
        model.load_state_dict_pre_hook(model_classes, chkpt_classes, chkpt_dict)
        for k, gt in gt_dict.items():
            assert (chkpt_dict[k] != gt).sum() == 0
