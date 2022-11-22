"""OTX Adapters - openvino.utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .fake_input import get_fake_input

from .utils import (
    prepare_mmdet_model_for_execution,
    build_val_dataloader,
)

__all__ = [
    "get_fake_input"
    "prepare_mmdet_model_for_execution",
    "build_val_dataloader",
]
