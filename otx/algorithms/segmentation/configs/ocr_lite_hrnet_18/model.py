"""Model configuration of OCR-Lite-HRnet-18 model for Segmentation Task."""

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
    "../../../../recipes/stages/segmentation/incremental.py",
    "../../../common/adapters/mmcv/configs/backbones/lite_hrnet_18.py",
]

model = dict(
    type="ClassIncrEncoderDecoder",
    pretrained=None,
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
        enable_out_norm=False,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
        ],
    ),
)

load_from = "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions\
/models/custom_semantic_segmentation/litehrnet18_imagenet1k_rsc.pth"

fp16 = dict(loss_scale=512.0)
