"""Model configuration of OCR-Lite-HRnet-18-mod2 model for SupCon Segmentation Task."""

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
    "../../../../../recipes/stages/segmentation/supcon.py",
    "../../../../common/adapters/mmcv/configs/backbones/lite_hrnet_18.py",
]

model = dict(
    type="SupConDetConB",
    pretrained=(
        "https://storage.openvinotoolkit.org/repositories/"
        "openvino_training_extensions/models/custom_semantic_segmentation/"
        "litehrnet18_imagenet1k_rsc.pth"
    ),
    num_classes=256,
    num_samples=16,
    downsample=4,
    input_transform="resize_concat",
    in_index=[0, 1, 2, 3],
    neck=dict(
        type="SelfSLMLP",
        in_channels=600,
        hid_channels=256,
        out_channels=128,
        norm_cfg=dict(type="BN1d", requires_grad=True),
        with_avg_pool=False,
    ),
    head=dict(
        type="DetConHead",
        predictor=dict(
            type="SelfSLMLP",
            in_channels=128,
            hid_channels=256,
            out_channels=128,
            norm_cfg=dict(type="BN1d", requires_grad=True),
            with_avg_pool=False,
        ),
        loss_cfg=dict(type="DetConLoss", temperature=0.1),
    ),
    decode_head=dict(
        type="FCNHead",
        in_channels=[40, 80, 160, 320],
        in_index=[0, 1, 2, 3],
        input_transform="multiple_select",
        channels=40,
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=2,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        enable_aggregator=True,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
        ],
    ),
)

load_from = None

resume_from = None

fp16 = dict(loss_scale=512.0)
