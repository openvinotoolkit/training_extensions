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

_base_ = [
    "../../../common/adapters/mmcv/configs/backbones/lite_hrnet_s.py",
]

model = dict(
    type="ClassIncrEncoderDecoder",
    pretrained=None,
    decode_head=dict(
        type="FCNHead",
        in_channels=[60, 120, 240],
        in_index=(0, 1, 2),
        input_transform='resize_concat',
        channels=60,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
        ],
    ),
    train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)),
    test_cfg=dict(mode='slide', crop_size=(512,512), stride=(341, 341))
)

load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions\
/models/custom_semantic_segmentation/litehrnetsv2_imagenet1k_rsc.pth"

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
