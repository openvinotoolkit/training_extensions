"""Model configuration of X3D model for Action Classification Task."""

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

_base_ = ["../base/supervised.py"]

num_classes = 400
num_samples = 12
model = dict(
    type="Recognizer3D",
    backbone=dict(type="X3D", gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type="X3DHead", in_channels=432, num_classes=num_classes, spatial_type="avg", dropout_ratio=0.5, fc1_bias=False
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips="prob"),
)

resume_from = None
load_from = (
    "https://download.openmmlab.com/mmaction/recognition/x3d/facebook/"
    "x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth"
)
