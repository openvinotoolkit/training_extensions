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
        return DataModuleConfig(
            data_format="coco_instances",
            data_root=".",
            subsets={
                "train": MagicMock(spec=SubsetConfig),
                "val": MagicMock(spec=SubsetConfig),
                "test": MagicMock(spec=SubsetConfig),
            },
            train_subset_name="train_1",
            val_subset_name="val_1",
            test_subset_name="test_1",
        )

    @patch("otx.core.data.module.OTXDatasetFactory")
    @patch("otx.core.data.module.DmDataset.import_from")
    @pytest.mark.parametrize(
        "task", [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.DETECTION],
    )
    def test_init(self, mock_dm_dataset, mock_otx_dataset_factory, task, fxt_config) -> None:
        mock_dm_subsets = {
            f"{name}_{idx}": MagicMock()
            for name in ["train", "val", "test"]
            for idx in range(2)
        }
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        OTXDataModule(task=task, config=fxt_config)

        assert mock_otx_dataset_factory.create.call_count == 3
