"""OTX Algorithms - Action Classification Dataset."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from . import pipelines
from .otx_action_cls_dataset import OTXActionClsDataset

__all__ = ["OTXActionClsDataset", "pipelines"]
