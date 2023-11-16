"""Unit tests of otx/v2/adapters/torch/mmengine/mmdet/modules/detectors/custom_maskrcnn_detector.py."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
import torch
from mmdet.registry import MODELS
from mmengine.config import Config

from otx.v2.adapters.torch.mmengine.mmdet.modules.models.detectors.custom_maskrcnn_detector import (
    CustomMaskRCNN,
)


class TestCustomMaskRCNN:
    def test_custom_maskrcnn_build(self, fxt_cfg_custom_maskrcnn: Dict):
        model = MODELS.build(Config(fxt_cfg_custom_maskrcnn))
        assert isinstance(model, CustomMaskRCNN)

    def test_custom_maskrcnn_load_state_dict_pre_hook(self):
        chkpt_classes = ["person", "car"]
        model_classes = ["tree", "car", "person"]
        chkpt_dict = {
            "roi_head.bbox_head.fc_cls.weight": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                ]
            ),
            "roi_head.bbox_head.fc_cls.bias": torch.tensor(
                [
                    [1],
                    [2],
                ]
            ),
        }
        model_dict = {
            "roi_head.bbox_head.fc_cls.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                ]
            ),
            "roi_head.bbox_head.fc_cls.bias": torch.tensor(
                [
                    [3],
                    [4],
                    [5],
                ]
            ),
        }
        gt_dict = {
            "roi_head.bbox_head.fc_cls.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                ]
            ),
            "roi_head.bbox_head.fc_cls.bias": torch.tensor(
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

        CustomMaskRCNN.load_state_dict_pre_hook(Model(), model_classes, chkpt_classes, chkpt_dict, "")
        for k, gt in gt_dict.items():
            assert (chkpt_dict[k] != gt).sum() == 0
