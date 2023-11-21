"""OTX Ligthning adapter modules - Dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# [TODO]: This should change with anomalib v1.0.
from otx.v2.adapters.torch.lightning.registry import DATASETS

from .anomaly_dataset import OTXAnomalyDataModule, OTXAnomalyDataset
from .pipelines import MultipleInputsCompose, Pad, ResizeLongestSide
from .visual_prompting_dataset import OTXVisualPromptingDataModule, OTXVisualPromptingDataset, get_transform

__all__ = [
    "OTXVisualPromptingDataModule",
    "OTXVisualPromptingDataset",
    "get_transform",
    "MultipleInputsCompose",
    "Pad",
    "ResizeLongestSide",
    "OTXAnomalyDataModule",
    "OTXAnomalyDataset",
    "DATASETS",
]

DATASETS.register_module(name="visual_prompting", module=OTXVisualPromptingDataset)
DATASETS.register_module(name="anomaly", module=OTXAnomalyDataset)
