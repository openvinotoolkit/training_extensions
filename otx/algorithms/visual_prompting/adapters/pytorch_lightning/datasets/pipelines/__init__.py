"""OTX Algorithms - Visual prompting pipelines."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .transforms import ResizeAndPad, collate_fn

__all__ = ["ResizeAndPad", "collate_fn"]
