# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of classification datasets."""
from otx.core.data.dataset.classification import OTXHlabelClsDataset
from otx.core.data.dataset.classification import HLabelInfo
from datumaro import Label
from datumaro.components.dataset import DatasetSubset
from unittest.mock import MagicMock

class TestOTXHlabelClsDataset:
    def test_add_ancestors(self, fxt_hlabel_dataset_subset, fxt_hlabel_multilabel_info):
        label_anns = [Label(label=1, id=0, group=0)]
        
        hlabel_dataset = MagicMock(spec=OTXHlabelClsDataset)
        hlabel_dataset.dm_subset = fxt_hlabel_dataset_subset
        hlabel_dataset.label_info = fxt_hlabel_multilabel_info
        
        hlabel_dataset.add_ancestors()
    