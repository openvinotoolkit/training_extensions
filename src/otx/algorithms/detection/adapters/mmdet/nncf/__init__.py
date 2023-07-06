"""NNCF utils for mmdet."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa

from .builder import build_nncf_detector

__all__ = [
    "build_nncf_detector",
]
