import torch

from otx.algorithms.detection.adapters.mmdet.models.backbones import imgclsmob
from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_single_stage_detector import (
    CustomSingleStageDetector,
)
from otx.algorithms.detection.adapters.mmdet.models.heads import custom_anchor_generator

__all__ = ["imgclsmob", "custom_anchor_generator"]

from mmdet.models.builder import build_detector

from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomSingleStageDetector:
    @e2e_pytest_unit
    def test_custom_single_stage_detector_build(self, fxt_cfg_custom_ssd):
        model = build_detector(fxt_cfg_custom_ssd)
        assert isinstance(model, CustomSingleStageDetector)

    @e2e_pytest_unit
    def test_custom_single_stage_detector_load_state_dict_pre_hook(self):
        chkpt_classes = ["person", "car"]
        model_classes = ["tree", "car", "person"]
        chkpt_dict = {
            "bbox_head.cls_convs.0.weight": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [0, 0, 0, 0],  # BG
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [0, 0, 0, 0],  # BG
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [0, 0, 0, 0],  # BG
                ]
            ),
            "bbox_head.cls_convs.0.bias": torch.tensor(
                [
                    [1],
                    [2],
                    [0],  # BG
                    [1],
                    [2],
                    [0],  # BG
                    [1],
                    [2],
                    [0],  # BG
                ]
            ),
        }
        model_dict = {
            "bbox_head.cls_convs.0.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                    [0, 0, 0, 0],  # BG
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                    [0, 0, 0, 0],  # BG
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                    [0, 0, 0, 0],  # BG
                ]
            ),
            "bbox_head.cls_convs.0.bias": torch.tensor(
                [
                    [3],
                    [4],
                    [5],
                    [0],  # BG
                    [3],
                    [4],
                    [5],
                    [0],  # BG
                    [3],
                    [4],
                    [5],
                    [0],  # BG
                ]
            ),
        }
        gt_dict = {
            "bbox_head.cls_convs.0.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],  # BG
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],  # BG
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],  # BG
                ]
            ),
            "bbox_head.cls_convs.0.bias": torch.tensor(
                [
                    [3],
                    [2],
                    [1],
                    [0],  # BG
                    [3],
                    [2],
                    [1],
                    [0],  # BG
                    [3],
                    [2],
                    [1],
                    [0],  # BG
                ]
            ),
        }

        class Model:
            def state_dict(self):
                return model_dict

        CustomSingleStageDetector.load_state_dict_pre_hook(Model(), model_classes, chkpt_classes, chkpt_dict, "")
        for k, gt in gt_dict.items():
            assert (chkpt_dict[k] != gt).sum() == 0
