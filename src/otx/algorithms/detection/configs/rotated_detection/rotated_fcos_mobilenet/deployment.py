"""Deployment of rotated_atss_obb_r50 for Rotated-Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

_base_ = ["../../base/deployments/base_rotated_detection_dynamic.py"]

ir_config = dict(
    output_names=["boxes", "labels"],
)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 1024, 1024]))],
)
