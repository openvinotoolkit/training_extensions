"""Single Stage detector configuration of OTX Detection."""

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

_base_ = "./detector.py"

model = dict(
    type="SingleStageDetector",
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False,
        ),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,
    ),
    test_cfg=dict(
        nms=dict(type="nms", iou_threshold=0.5),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200,
    ),
)
