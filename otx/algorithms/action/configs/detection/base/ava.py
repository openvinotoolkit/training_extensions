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


_base_ = ["./data.py"]
# These should be assigned otx cli, but harded-coded
# These wii be changed when annotation format is changed to CVAT
anno_root = "/home/jaeguk/workspace/data/ava/annotations"
exclude_file_train = f"{anno_root}/ava_train_excluded_timestamps_v2.2.csv"
exclude_file_val = f"{anno_root}/ava_val_excluded_timestamps_v2.2.csv"
proposal_file_train = f"{anno_root}/ava_dense_proposals_train.FAIR." "recall_93.9.pkl"
proposal_file_val = f"{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl"

# Data pipeline
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

# Dataset structure
# TODO Sync with latest mmaction2
# pylint: disable=unhashable-member, no-member
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=0,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=0),
    train=dict(
        exclude_file=exclude_file_train,
        pipeline={{_base_.train_pipeline}},  # type: ignore[attr-defined]
        proposal_file=proposal_file_train,
        filename_tmpl="_{:06}.jpg",
        person_det_score_thr=0.9,
        timestamp_start=900,
        timestamp_end=1800,
        fps=30,
    ),
    val=dict(
        exclude_file=exclude_file_val,
        pipeline={{_base_.val_pipeline}},  # type: ignore[attr-defined]
        proposal_file=proposal_file_val,
        filename_tmpl="_{:06}.jpg",
        person_det_score_thr=0.9,
        timestamp_start=900,
        timestamp_end=1800,
        fps=30,
    ),
)
data["test"] = data["val"]
