# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict

import mmcv
import pytest
from mmcv.utils import ConfigDict


@pytest.fixture
def fxt_cfg_custom_atss(num_classes: int = 3) -> Dict:
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(type="ATSSAssigner", topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        )
    )
    cfg = dict(
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
            num_classes=num_classes,
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
        train_cfg=train_cfg,
    )
    return cfg


@pytest.fixture
def fxt_cfg_custom_ssd(num_classes: int = 3) -> Dict:
    train_cfg = mmcv.Config(
        {
            "assigner": {
                "type": "MaxIoUAssigner",
                "min_pos_iou": 0.0,
                "ignore_iof_thr": -1,
                "gt_max_assign_all": False,
                "pos_iou_thr": 0.4,
                "neg_iou_thr": 0.4,
            },
            "smoothl1_beta": 1.0,
            "allowed_border": -1,
            "pos_weight": -1,
            "neg_pos_ratio": 3,
            "debug": False,
            "use_giou": False,
            "use_focal": False,
        }
    )

    cfg = dict(
        type="CustomSingleStageDetector",
        backbone=dict(type="mobilenetv2_w1", out_indices=(4, 5), frozen_stages=-1, norm_eval=False, pretrained=True),
        neck=None,
        bbox_head=dict(
            type="CustomSSDHead",
            num_classes=num_classes,
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
        train_cfg=train_cfg,
    )

    return cfg


@pytest.fixture
def fxt_cfg_custom_vfnet(num_classes: int = 3):
    return ConfigDict(
        type="CustomVFNet",
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
            type="CustomVFNetHead",
            num_classes=num_classes,
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


@pytest.fixture
def fxt_cfg_custom_yolox(num_classes: int = 3):
    cfg = {
        "train_cfg": mmcv.Config({"assigner": {"type": "SimOTAAssigner", "center_radius": 2.5}}),
        "type": "CustomYOLOX",
        "backbone": {"type": "CSPDarknet", "deepen_factor": 0.33, "widen_factor": 0.375, "out_indices": (2, 3, 4)},
        "neck": {"type": "YOLOXPAFPN", "in_channels": [96, 192, 384], "out_channels": 96, "num_csp_blocks": 1},
        "bbox_head": {"type": "CustomYOLOXHead", "num_classes": num_classes, "in_channels": 96, "feat_channels": 96},
        "task_adapt": {
            "src_classes": (
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ),
            "dst_classes": ["car", "tree", "bug"],
        },
    }
    return cfg


@pytest.fixture
def fxt_cfg_custom_deformable_detr(num_classes: int = 3):
    return ConfigDict(
        type="CustomDeformableDETR",
        backbone=dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=False),
            norm_eval=True,
            style="pytorch",
            init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        ),
        neck=dict(
            type="ChannelMapper",
            in_channels=[512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type="GN", num_groups=32),
            num_outs=4,
        ),
        bbox_head=dict(
            type="DeformableDETRHead",
            num_query=300,
            num_classes=num_classes,
            in_channels=2048,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=True,
            transformer=dict(
                type="DeformableDetrTransformer",
                encoder=dict(
                    type="DetrTransformerEncoder",
                    num_layers=6,
                    transformerlayers=dict(
                        type="BaseTransformerLayer",
                        attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "ffn", "norm"),
                    ),
                ),
                decoder=dict(
                    type="DeformableDetrTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1),
                            dict(type="MultiScaleDeformableAttention", embed_dims=256),
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ),
                ),
            ),
            positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        ),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
        task_adapt=dict(
            src_classes=["person", "car"],
            dst_classes=["tree", "car", "person"],
        ),
    )


@pytest.fixture
def fxt_cfg_custom_dino(num_classes: int = 3):
    return ConfigDict(
        type="CustomDINO",
        backbone=dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=False),
            norm_eval=True,
            style="pytorch",
            init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        ),
        neck=dict(
            type="ChannelMapper",
            in_channels=[512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type="GN", num_groups=32),
            num_outs=4,
        ),
        bbox_head=dict(
            type="CustomDINOHead",
            num_query=900,
            num_classes=num_classes,
            in_channels=2048,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=True,
            transformer=dict(
                type="CustomDINOTransformer",
                encoder=dict(
                    type="DetrTransformerEncoder",
                    num_layers=6,
                    transformerlayers=dict(
                        type="BaseTransformerLayer",
                        attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256, dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=("self_attn", "norm", "ffn", "norm"),
                    ),
                ),
                decoder=dict(
                    type="DINOTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.0),
                            dict(type="MultiScaleDeformableAttention", embed_dims=256, dropout=0.0),
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                    ),
                ),
            ),
            positional_encoding=dict(
                type="SinePositionalEncoding", num_feats=128, normalize=True, offset=0.0, temperature=20
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
            dn_cfg=dict(
                label_noise_scale=0.5,
                box_noise_scale=1.0,  # 0.4 for DN-DETR
                group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
            ),
        ),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="FocalLossCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=300),
        task_adapt=dict(
            src_classes=["person", "car"],
            dst_classes=["tree", "car", "person"],
        ),
    )
