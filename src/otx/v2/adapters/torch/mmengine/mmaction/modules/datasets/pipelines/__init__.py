"""Collection of data pipelines for OTX Action Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .loading import OTXRawFrameDecode

__all__ = ["OTXRawFrameDecode"]
