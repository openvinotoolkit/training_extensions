"""Model configuration of Fast RCNN with X3D for Action Detection Task."""

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

# model setting
model = dict(
    type="AVAFastRCNN",
    backbone=dict(type="X3D", gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    roi_head=dict(
        type="AVARoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor3D", roi_layer_type="RoIAlign", output_size=8, with_temporal_pool=True
        ),
        bbox_head=dict(type="BBoxHeadAVA", in_channels=432, num_classes=81, multilabel=False, dropout_ratio=0.5),
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(type="MaxIoUAssignerAVA", pos_iou_thr=0.9, neg_iou_thr=0.9, min_pos_iou=0.9),
            sampler=dict(type="RandomSampler", num=32, pos_fraction=1, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False,
        )
    ),
    test_cfg=dict(rcnn=dict(action_thr=0.002)),
)

load_from = (
    "https://download.openmmlab.com/mmaction/recognition/x3d/facebook/"
    "x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth"
)
resume_from = None
