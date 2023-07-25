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
depths = [2, 2, 18, 2]
num_classes = num_things_classes

model = dict(
    type="CustomMask2Former",
    backbone=dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
    ),
    panoptic_head=dict(
        type="Mask2FormerHead",
        in_channels=[96, 192, 384, 768],
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
        panoptic_on=False, semantic_on=False, instance_on=True, max_per_image=100, iou_thr=0.6, filter_low_score=True
    ),
    init_cfg=None,
)

custom_keys = dict(
    {
        "backbone": dict(lr_mult=0.1, decay_mult=1.0),
        "backbone.patch_embed.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "absolute_pos_embed": dict(lr_mult=0.1, decay_mult=0.0),
        "relative_position_bias_table": dict(lr_mult=0.1, decay_mult=0.0),
        "query_embed": dict(lr_mult=1.0, decay_mult=0.0),
        "query_feat": dict(lr_mult=1.0, decay_mult=0.0),
        "level_embed": dict(lr_mult=1.0, decay_mult=0.0),
        "backbone.stages.0.blocks.0.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.0.blocks.1.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.1.blocks.0.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.1.blocks.1.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.0.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.1.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.2.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.3.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.4.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.5.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.3.blocks.0.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.3.blocks.1.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.0.downsample.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.1.downsample.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.downsample.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.6.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.7.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.8.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.9.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.10.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.11.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.12.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.13.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.14.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.15.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.16.norm": dict(lr_mult=0.1, decay_mult=0.0),
        "backbone.stages.2.blocks.17.norm": dict(lr_mult=0.1, decay_mult=0.0),
    }
)

evaluation = dict(interval=1, metric="mAP", save_best="mAP", iou_thr=[0.5])
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.01, norm_type=2))

load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"

# NOTE: Disable incremental learning for the time being
ignore = False
