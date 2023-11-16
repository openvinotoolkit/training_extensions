"""OTX Algorithms - Detection Dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from . import pipelines
from .dataset import OTXDetDataset

__all__ = ["OTXDetDataset", "pipelines"]
