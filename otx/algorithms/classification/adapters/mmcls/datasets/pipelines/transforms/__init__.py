"""Module to init transforms for OTX classification."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# flake8: noqa

from .augmix import AugMixAugment
from .otx_transforms import PILToTensor, RandomRotate, OTXNormalize
from .random_augment import OTXRandAugment
from .twocrop_transform import TwoCropTransform

__all__ = [
    "AugMixAugment",
    "PILToTensor",
    "OTXNormalize",
    "RandomRotate",
    "OTXRandAugment",
    "TwoCropTransform",
]
