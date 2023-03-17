"""Model Configuration of VFNet model for Detection Task."""

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
    "../../../../../recipes/stages/detection/incremental.py",
    "../../../../common/adapters/mmcv/configs/backbones/resnet50.yaml",
    "../../base/models/detector.py",
]

model = dict(
    type="CustomVFNet",
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
            type="VarifocalLoss",
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=1.5),
        loss_bbox_refine=dict(type="GIoULoss", loss_weight=2.0),
    ),
    train_cfg=dict(
        assigner=dict(type="ATSSAssigner", topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100,
    ),
)

load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions\
/models/object_detection/v2/resnet50-vfnet.pth"

__img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

data = dict(
    pipeline_options=dict(
        MinIouRandomCrop=dict(min_crop_size=0.1),
        Resize=dict(
            img_scale=[(1344, 480), (1344, 960)],
            multiscale_mode="range",
        ),
        Normalize=dict(**__img_norm_cfg),
        MultiScaleFlipAug=dict(
            img_scale=(1344, 800),
            flip=False,
            transforms=[
                dict(type="Resize", keep_ratio=False),
                dict(type="Normalize", **__img_norm_cfg),
                dict(type="Pad", size_divisor=32),
                dict(type="ImageToTensor", keys=["img"]),
                dict(type="Collect", keys=["img"]),
            ],
        ),
    ),
)
