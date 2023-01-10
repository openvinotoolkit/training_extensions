"""Data configuration for default action detection dataset."""

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

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type="SampleAVAFrames", clip_len=32, frame_interval=2),
    dict(type="RawFrameDecode"),
    dict(type="RandomRescale", scale_range=(256, 320)),
    dict(type="RandomCrop", size=256),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type="Rename", mapping=dict(imgs="img")),
    dict(type="ToTensor", keys=["img", "proposals", "gt_bboxes", "gt_labels"]),
    dict(type="ToDataContainer", fields=[dict(key=["proposals", "gt_bboxes", "gt_labels"], stack=False)]),
    dict(type="Collect", keys=["img", "proposals", "gt_bboxes", "gt_labels"], meta_keys=["scores"]),
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type="SampleAVAFrames", clip_len=32, frame_interval=2, test_mode=True),
    dict(type="RawFrameDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW", collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type="Rename", mapping=dict(imgs="img")),
    dict(type="ToTensor", keys=["img", "proposals"]),
    dict(type="ToDataContainer", fields=[dict(key="proposals", stack=False)]),
    dict(type="Collect", keys=["img", "proposals"], meta_keys=["scores", "img_shape"], nested=True),
]

data = dict(
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        pipeline=train_pipeline,
        person_det_score_thr=0.5,
        fps=1,
    ),
    val=dict(
        pipeline=val_pipeline,
        person_det_score_thr=0.5,
        fps=1,
        test_mode=True,
    ),
)
data["test"] = data["val"]
