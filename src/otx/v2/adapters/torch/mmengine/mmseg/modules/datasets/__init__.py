"""OTX Algorithms - Segmentation Dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .otx_datasets import OTXSegDataset
from .pipeline import LoadAnnotationFromOTXDataset

__all__ = [
    "OTXSegDataset",
    "LoadAnnotationFromOTXDataset",
]
