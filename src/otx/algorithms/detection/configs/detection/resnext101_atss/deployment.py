"""MMDeploy config of ResNeXt101-ATSS model for Detection Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

_base_ = ["../../base/deployments/base_detection_dynamic.py"]

ir_config = dict(
    output_names=["boxes", "labels"],
)

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[-1, 3, 736, 992]))],
)
