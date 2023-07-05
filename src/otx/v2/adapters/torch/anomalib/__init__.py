"""Adapter of Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .dataset import Dataset
from .engine import AnomalibEngine as Engine
from .model import build_model_from_config

__all__ = ["Dataset", "Engine", "build_model_from_config"]
