"""Data configuration for AVA dataset."""

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


dataset_type = "AVADataset"

# These will be assigned through otx cli
data_root = None
ann_file_train = None
ann_file_val = None

# These also should be assigned otx cli, but harded-coded
# These wii be changed when annotation format is CVAT
anno_root = "/home/jaeguk/workspace/data/ava/annotations"
exclude_file_train = f"{anno_root}/ava_train_excluded_timestamps_v2.2.csv"
exclude_file_val = f"{anno_root}/ava_val_excluded_timestamps_v2.2.csv"

label_file = f"{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt"

proposal_file_train = f"{anno_root}/ava_dense_proposals_train.FAIR." "recall_93.9.pkl"
proposal_file_val = f"{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl"

# Data pipeline
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
    dict(type="Collect", keys=["img", "proposals", "gt_bboxes", "gt_labels"], meta_keys=["scores", "entity_ids"]),
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

# Dataset structure
# TODO Sync with latest mmaction2
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=0,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=0),
    train=dict(
        type=dataset_type,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        proposal_file=proposal_file_train,
        label_file=label_file,
        person_det_score_thr=0.9,
    ),
    val=dict(
        type=dataset_type,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        proposal_file=proposal_file_val,
        label_file=label_file,
        person_det_score_thr=0.9,
    ),
)
data["test"] = data["val"]
