"""OTX Adapters - openvino.utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .fake_input import get_fake_input

from .utils import (
    build_dataloader,
)

__all__ = [
    "get_fake_input"
    "build_dataloader",
]
