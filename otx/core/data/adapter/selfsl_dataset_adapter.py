"""Self-SL Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional, Dict
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter
from otx.api.entities.subset import Subset
from datumaro.components.dataset import Dataset as DatumaroDataset


class SelfSLSegmentationDatasetAdapter(BaseDatasetAdapter):
    """Self-SL for semantic segmentation adapter inherited from BaseDatasetAdapter."""

    def _import_dataset(
        self,
        train_data_roots: Optional[str] = None,
    ) -> Dict[Subset, DatumaroDataset]:
        """Import custom Self-SL dataset for using DetCon."""
        pass