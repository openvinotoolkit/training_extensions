"""OTX Algorithms - Segmentation Dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .otx_datasets import OTXSegDataset
from .pipeline import LoadAnnotationFromOTXDataset, get_annotation_mmseg_format

__all__ = [
    "get_annotation_mmseg_format",
    "OTXSegDataset",
    "LoadAnnotationFromOTXDataset",
]
