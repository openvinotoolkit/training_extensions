import torch
from mmcv.utils import ConfigDict
from mmdet.models.builder import build_detector

from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_vfnet_detector import (
    CustomVFNet,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomVFNet:
    @e2e_pytest_unit
    def test_custom_vfnet_build(self):
        model_cfg = ConfigDict(
            type="CustomVFNet",
            backbone=dict(
                type="ResNet",
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="SyncBN", requires_grad=True),
                norm_eval=True,
                style="pytorch",
            ),
            neck=dict(
                type="FPN",
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs="on_output",
                num_outs=5,
                relu_before_extra_convs=True,
            ),
            bbox_head=dict(
                type="VFNetHead",
                num_classes=2,
                in_channels=256,
                stacked_convs=3,
                feat_channels=256,
                strides=[8, 16, 32, 64, 128],
                center_sampling=False,
                dcn_on_last_conv=False,
                use_atss=True,
                use_vfl=True,
                loss_cls=dict(
                    type="VarifocalLoss", use_sigmoid=True, alpha=0.75, gamma=2.0, iou_weighted=True, loss_weight=1.0
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=1.5),
                loss_bbox_refine=dict(type="GIoULoss", loss_weight=2.0),
            ),
            train_cfg=dict(assigner=dict(type="ATSSAssigner", topk=9), allowed_border=-1, pos_weight=-1, debug=False),
            test_cfg=dict(
                nms_pre=1000, min_bbox_size=0, score_thr=0.01, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100
            ),
            task_adapt=dict(
                src_classes=["person", "car"],
                dst_classes=["tree", "car", "person"],
            ),
        )

        model = build_detector(model_cfg)
        assert isinstance(model, CustomVFNet)

    @e2e_pytest_unit
    def test_custom_vfnet_load_state_dict_pre_hook(self):
        chkpt_classes = ["person", "car"]
        model_classes = ["tree", "car", "person"]
        chkpt_dict = {
            "bbox_head.vfnet_cls.weight": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                ]
            ),
            "bbox_head.vfnet_cls.bias": torch.tensor(
                [
                    [1],
                    [2],
                ]
            ),
        }
        model_dict = {
            "bbox_head.vfnet_cls.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                ]
            ),
            "bbox_head.vfnet_cls.bias": torch.tensor(
                [
                    [3],
                    [4],
                    [5],
                ]
            ),
        }
        gt_dict = {
            "bbox_head.vfnet_cls.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                ]
            ),
            "bbox_head.vfnet_cls.bias": torch.tensor(
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

        CustomVFNet.load_state_dict_pre_hook(Model(), model_classes, chkpt_classes, chkpt_dict, "")
        for k, gt in gt_dict.items():
            assert (chkpt_dict[k] != gt).sum() == 0
