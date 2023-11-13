"""Unit tests of otx/v2/adapters/torch/mmengine/mmdet/modules/detectors/custom_atss_detector.py."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
import torch
from mmdet.registry import MODELS

from otx.v2.adapters.torch.mmengine.mmdet.modules.models.detectors.custom_atss_detector import (
    CustomATSS,
)


class TestCustomATSS:
    def test_custom_atss_build(self, fxt_cfg_custom_atss: Dict):
        model = MODELS.build(fxt_cfg_custom_atss)
        assert isinstance(model, CustomATSS)

    def test_custom_atss_load_state_dict_pre_hook(self):
        chkpt_classes = ["person", "car"]
        model_classes = ["tree", "car", "person"]
        chkpt_dict = {
            "bbox_head.atss_cls.weight": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                ]
            ),
            "bbox_head.atss_cls.bias": torch.tensor(
                [
                    [1],
                    [2],
                ]
            ),
        }
        model_dict = {
            "bbox_head.atss_cls.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                ]
            ),
            "bbox_head.atss_cls.bias": torch.tensor(
                [
                    [3],
                    [4],
                    [5],
                ]
            ),
        }
        gt_dict = {
            "bbox_head.atss_cls.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                ]
            ),
            "bbox_head.atss_cls.bias": torch.tensor(
                [
                    [3],
                    [2],
                    [1],
                ]
            ),
        }

        class Model:
            def state_dict(self):
                return model_dict

        CustomATSS.load_state_dict_pre_hook(Model(), model_classes, chkpt_classes, chkpt_dict, "")
        for k, gt in gt_dict.items():
            assert (chkpt_dict[k] != gt).sum() == 0
