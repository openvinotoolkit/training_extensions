"""MobileNet-V3-Small for multi-label config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/multilabel/incremental.yaml", "../base/models/mobilenet_v3.py"]

model = dict(
    type="CustomImageClassifier",
    task="classification",
    head=dict(
        type="CustomMultiLabelNonLinearClsHead",
        loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=0.0,
        ),
    ),
)
