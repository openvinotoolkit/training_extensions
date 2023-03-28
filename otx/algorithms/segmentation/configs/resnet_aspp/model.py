"""Model configuration of OCR-Lite-HRnet-s-mod2 model for Segmentation Task."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

# pylint: disable=invalid-name

norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type='ClassIncrEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode='slide', crop_size=(512,512), stride=(341, 341)))

load_from = 'open-mmlab://resnet50_v1c'

fp16 = dict(loss_scale=512.0)

optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4, paramwise_cfg={'bias_decay_mult ': 0.0, 'norm_decay_mult ': 0.0})
optimizer_config = dict()
# optimizer_config = dict(
#     _delete_=True,
#     grad_clip=dict(
#         # method='adaptive',
#         # clip=0.2,
#         # method='default',
#         max_norm=40,
#         norm_type=2,
#     ),
# )

lr_config = dict(policy='poly',  warmup='linear',  warmup_iters=400, warmup_ratio=1e-6, power=0.9,  min_lr=1e-6, by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True, ignore_last=False),
        # dict(type='TensorboardLoggerHook')
    ],
)

dist_params = dict(backend="nccl", linear_scale_lr=False)

runner = dict(type="EpochBasedRunner", max_epochs=100)

checkpoint_config = dict(by_epoch=True, interval=1)

evaluation = dict(interval=1, metric=["mDice", "mIoU"], show_log=True)

find_unused_parameters = False

task_adapt = dict(
    type="mpa",
    op="REPLACE",
)

ignore = True

cudnn_benchmark = False

deterministic = False

hparams = dict(dummy=0)

# yapf:disable
log_config = dict(
    interval=100, hooks=[dict(type="TextLoggerHook", ignore_last=False), dict(type="TensorboardLoggerHook")]
)
# yapf:enable

log_level = "INFO"

resume_from = None
workflow = [("train", 1)]
