"""EfficientNet-B0 for hierarchical config."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name

_base_ = ["../../../../recipes/stages/classification/incremental.yaml", "../base/models/efficientnet.py"]

model = dict(
    type="CustomImageClassifier",
    task="classification",
    backbone=dict(
        version="b0",
    ),
    head=dict(
        type="CustomHierarchicalLinearClsHead",
        multilabel_loss=dict(
            type="AsymmetricLossWithIgnore",
            gamma_pos=0.0,
            gamma_neg=4.0,
        ),
    ),
)
