"""MMDeploy config partitioning Swin-T MaskRCNN model to tile classifier and MaskRCNN model."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

_base_ = ["./deployment.py"]

ir_config = dict(
    output_names=["boxes", "labels", "masks", "tile_prob"],
)

partition_config = dict(
    type="tile_classifier",
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file="tile_classifier.onnx",
            start=["tile_classifier:input"],
            end=["tile_classifier:output"],
            output_names=["tile_prob"],
        )
    ],
)
