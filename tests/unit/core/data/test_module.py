# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest
from otx.core.config.data import (
    DataModuleConfig,
    SubsetConfig,
)
from otx.core.data.module import (
    OTXDataModule,
    OTXTaskType,
)


class TestModule:
    @pytest.fixture()
    def fxt_config(self) -> DataModuleConfig:
        mock = MagicMock(spec=DataModuleConfig)
        mock.data_format = "coco_instances"
        mock.data_root = "."
        mock.mem_cache_size = "1GB"
        mock.train_subset = MagicMock(spec=SubsetConfig)
        mock.train_subset.num_workers = 0
        mock.val_subset = MagicMock(spec=SubsetConfig)
        mock.val_subset.num_workers = 0
        mock.test_subset = MagicMock(spec=SubsetConfig)
        mock.test_subset.num_workers = 0

        return mock

    @patch("otx.core.data.module.OTXDatasetFactory")
    @patch("otx.core.data.module.DmDataset.import_from")
    @pytest.mark.parametrize(
        "task",
        [
            OTXTaskType.MULTI_CLASS_CLS,
            OTXTaskType.DETECTION,
            OTXTaskType.SEMANTIC_SEGMENTATION,
        ],
    )
    def test_init(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        task,
        fxt_config,
    ) -> None:
        # Our query for subset name for train, val, test
        fxt_config.train_subset.subset_name = "train_1"
        fxt_config.val_subset.subset_name = "val_1"
        fxt_config.test_subset.subset_name = "test_1"

        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        OTXDataModule(task=task, config=fxt_config)

        assert mock_otx_dataset_factory.create.call_count == 3
