# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of classification datasets."""
import pytest

from otx.core.data.dataset.classification import OTXHlabelClsDataset
from otx.core.data.dataset.classification import HLabelInfo
from datumaro import Label
from datumaro.components.dataset import DatasetSubset
from unittest.mock import MagicMock

class TestOTXHlabelClsDataset:
    def test_add_ancestors(self, fxt_hlabel_dataset_subset):
        original_anns = fxt_hlabel_dataset_subset.get(id=0, subset="train").annotations
        assert len(original_anns) == 1
        
        hlabel_dataset = OTXHlabelClsDataset(
            dm_subset=fxt_hlabel_dataset_subset,
            transforms=MagicMock(),
        )
        # Added the ancestor 
        adjusted_anns = hlabel_dataset.dm_subset.get(id=0, subset="train").annotations
        assert len(adjusted_anns) == 2
            