import torch
from mmdet.models.builder import build_detector

from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_atss_detector import (
    CustomATSS,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomATSS:
    @e2e_pytest_unit
    def test_custom_atss_build(self):
        model_cfg = dict(
            type="CustomATSS",
            backbone=dict(
                avg_down=False,
                base_channels=64,
                conv_cfg=None,
                dcn=None,
                deep_stem=False,
                depth=18,
                dilations=(1, 1, 1, 1),
                frozen_stages=-1,
                in_channels=3,
                init_cfg=None,
                norm_cfg=dict(requires_grad=True, type="BN"),
                norm_eval=True,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                plugins=None,
                pretrained=None,
                stage_with_dcn=(False, False, False, False),
                stem_channels=None,
                strides=(1, 2, 2, 2),
                style="pytorch",
                type="mmdet.ResNet",
                with_cp=False,
                zero_init_residual=True,
            ),
            neck=dict(
                type="FPN",
                in_channels=[64, 128, 256, 512],
                out_channels=64,
                start_level=1,
                add_extra_convs="on_output",
                num_outs=5,
                relu_before_extra_convs=True,
            ),
            bbox_head=dict(
                type="CustomATSSHead",
                num_classes=2,
                in_channels=64,
                stacked_convs=4,
                feat_channels=64,
                anchor_generator=dict(
                    type="AnchorGenerator",
                    ratios=[1.0],
                    octave_base_scale=8,
                    scales_per_octave=1,
                    strides=[8, 16, 32, 64, 128],
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
                use_qfl=False,
                qfl_cfg=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0),
            ),
        )

        model = build_detector(model_cfg)
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
