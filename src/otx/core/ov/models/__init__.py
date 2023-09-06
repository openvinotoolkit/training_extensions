"""Module for otx.core.ov.models."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from .mmov_model import MMOVModel
from .ov_model import OVModel  # type: ignore[attr-defined]
from .parser_mixin import ParserMixin  # type: ignore[attr-defined]

__all__ = [
    "MMOVModel",
    "OVModel",
    "ParserMixin",
]
