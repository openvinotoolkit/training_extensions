"""Model configuration of Mask2Former model for Instance-Seg Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

_base_ = [
    "../../../../../recipes/stages/instance-segmentation/incremental.py",
    "../../base/models/detector.py",
]

task = "instance-segmentation"

num_things_classes = 1
num_classes = num_things_classes

model = dict(
    type="CustomMask2Former",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
    ),
    panoptic_head=dict(
        type="Mask2FormerHead",
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=0,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type="MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention",
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
        transformer_decoder=dict(
            type="DetrTransformerDecoder",
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=dict(
                    type="MultiheadAttention",
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False,
                ),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
                feedforward_channels=2048,
                operation_order=("cross_attn", "norm", "self_attn", "norm", "ffn", "norm"),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0),
        loss_dice=dict(
            type="DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
    ),
    panoptic_fusion_head=dict(
        type="CustomMaskFormerFusionHead",
        num_things_classes=num_things_classes,
        num_stuff_classes=0,
        loss_panoptic=None,
        init_cfg=None,
    ),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type="MaskHungarianAssigner",
            cls_cost=dict(type="ClassificationCost", weight=2.0),
            mask_cost=dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
            dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
        ),
        sampler=dict(type="MaskPseudoSampler"),
    ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        filter_low_score=True,
        score_threshold=0.1,
    ),
    init_cfg=None,
)

evaluation = dict(interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5])
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0),
        ),
        norm_decay_mult=0.0,
    ),
)

lr_config = dict(
    policy="ReduceLROnPlateau",
    metric="mAP",
    patience=5,
    iteration_patience=0,
    interval=1,
    min_lr=1e-08,
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=0.3333333333333333,
)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.01, norm_type=2))

load_from = ("https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
             "mask2former_r50_lsj_8x2_50e_coco/"
             "mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth")

# NOTE: Disable incremental learning for the time being
ignore = False
