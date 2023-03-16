import torch
from mmcv.utils import ConfigDict
from mmdet.models.builder import build_detector

from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_two_stage_detector import (
    CustomTwoStageDetector,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomTwoStageDetector:
    @e2e_pytest_unit
    def test_custom_two_stage_detector_build(self):
        model_cfg = ConfigDict(
            type="CustomTwoStageDetector",
            backbone=dict(
                type="ResNet",
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN", requires_grad=True),
                norm_eval=True,
                style="pytorch",
            ),
            neck=dict(type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
            rpn_head=dict(
                type="RPNHead",
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
                ),
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]
                ),
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            roi_head=dict(
                type="StandardRoIHead",
                bbox_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0.0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32],
                ),
                bbox_head=dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
                    ),
                    reg_class_agnostic=False,
                    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                ),
            ),
            # model training and testing settings
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False
                    ),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False,
                ),
                rpn_proposal=dict(
                    nms_across_levels=False, nms_pre=2000, nms_post=1000, max_num=1000, nms_thr=0.7, min_bbox_size=0
                ),
                rcnn=dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(
                        type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True
                    ),
                    pos_weight=-1,
                    debug=False,
                ),
            ),
            test_cfg=dict(
                rpn=dict(
                    nms_across_levels=False, nms_pre=1000, nms_post=1000, max_num=1000, nms_thr=0.7, min_bbox_size=0
                ),
                rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100)
                # soft-nms is also supported for rcnn testing
                # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
            ),
            task_adapt=dict(
                src_classes=["person", "car"],
                dst_classes=["tree", "car", "person"],
            ),
        )

        model = build_detector(model_cfg)
        assert isinstance(model, CustomTwoStageDetector)

    @e2e_pytest_unit
    def test_custom_two_stage_detector_load_state_dict_pre_hook(self):
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
            "roi_head.bbox_head.fc_reg.weight": torch.tensor(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                ]
            ),
            "roi_head.bbox_head.fc_reg.bias": torch.tensor(
                [
                    [1],
                    [1],
                    [1],
                    [1],
                    [2],
                    [2],
                    [2],
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
            "roi_head.bbox_head.fc_reg.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5],
                    [5, 5, 5, 5],
                    [5, 5, 5, 5],
                    [5, 5, 5, 5],
                ]
            ),
            "roi_head.bbox_head.fc_reg.bias": torch.tensor(
                [
                    [3],
                    [3],
                    [3],
                    [3],
                    [4],
                    [4],
                    [4],
                    [4],
                    [5],
                    [5],
                    [5],
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
            "roi_head.bbox_head.fc_reg.weight": torch.tensor(
                [
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [3, 3, 3, 3],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
            "roi_head.bbox_head.fc_reg.bias": torch.tensor(
                [
                    [3],
                    [3],
                    [3],
                    [3],
                    [2],
                    [2],
                    [2],
                    [2],
                    [1],
                    [1],
                    [1],
                    [1],
                ]
            ),
        }

        class Model:
            def state_dict(self):
                return model_dict

        CustomTwoStageDetector.load_state_dict_pre_hook(Model(), model_classes, chkpt_classes, chkpt_dict, "")
        for k, gt in gt_dict.items():
            assert (chkpt_dict[k] != gt).sum() == 0
