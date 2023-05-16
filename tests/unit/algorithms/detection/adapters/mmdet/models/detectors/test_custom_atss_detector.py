from typing import Dict
import torch
from mmdet.models.builder import build_detector

from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_atss_detector import (
    CustomATSS,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomATSS:
    @e2e_pytest_unit
    def test_custom_atss_build(self, fxt_cfg_custom_atss: Dict):
        model = build_detector(fxt_cfg_custom_atss)
        assert isinstance(model, CustomATSS)

    @e2e_pytest_unit
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
