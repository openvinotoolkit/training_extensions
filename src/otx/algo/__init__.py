# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom algorithms, e.g., model, losses, hook, etc..."""

from . import (
    accelerators,
    action_classification,
    classification,
    detection,
    plugins,
    segmentation,
    strategies,
    visual_prompting,
    instance_segmentation,
)

__all__ = [
    "action_classification",
    "classification",
    "detection",
    "segmentation",
    "visual_prompting",
    "strategies",
    "accelerators",
    "plugins",
    "instance_segmentation",
]
