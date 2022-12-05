"""Anomaly Classification / Detection / Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
from otx.core.base_dataset_adapter import BaseDatasetAdapter
from otx.api.entities.datasets import DatasetEntity
from otx.utils.logger import get_logger

class AnomalyDatasetAdapter(BaseDatasetAdapter):
    def convert_to_otx_format(self, datumaro_dataset: dict) -> DatasetEntity:
        """ Conver DatumaroDataset to DatasetEntity for Anomalytasks. """
        # Prepare
        