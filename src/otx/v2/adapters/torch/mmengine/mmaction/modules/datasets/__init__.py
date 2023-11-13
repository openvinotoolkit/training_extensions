"""OTX Algorithms - Action Classification Dataset."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .otx_action_cls_dataset import OTXActionClsDataset
from . import pipelines
__all__ = ["OTXActionClsDataset", "pipelines"]
