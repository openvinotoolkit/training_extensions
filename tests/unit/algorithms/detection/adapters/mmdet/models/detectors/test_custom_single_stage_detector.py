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
    def test_custom_single_stage_detector_build(self):
        model_cfg = dict(
            type="CustomSingleStageDetector",
            backbone=dict(
                type="mmdet.mobilenetv2_w1", out_indices=(4, 5), frozen_stages=-1, norm_eval=False, pretrained=True
            ),
            neck=None,
            bbox_head=dict(
                type="CustomSSDHead",
                num_classes=80,
                in_channels=(int(96.0), int(320.0)),
                use_depthwise=True,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="ReLU"),
                init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
                loss_balancing=False,
                anchor_generator=dict(
                    type="SSDAnchorGeneratorClustered",
                    strides=(16, 32),
                    reclustering_anchors=True,
                    widths=[
                        [
                            38.641007923271076,
                            92.49516032784699,
                            271.4234764938237,
                            141.53469410876247,
                        ],
                        [
                            206.04136086566515,
                            386.6542727907841,
                            716.9892752215089,
                            453.75609561761405,
                            788.4629155558277,
                        ],
                    ],
                    heights=[
                        [
                            48.9243877087132,
                            147.73088476194903,
                            158.23569788707474,
                            324.14510379107367,
                        ],
                        [
                            587.6216059488938,
                            381.60024152086544,
                            323.5988913027747,
                            702.7486097568518,
                            741.4865860938451,
                        ],
                    ],
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=(0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2),
                ),
            ),
        )

        model = build_detector(model_cfg)
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
