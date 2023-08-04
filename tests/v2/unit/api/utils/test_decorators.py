"""OTX V2 API-utils Unit-Test codes (decorators)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.api.core.dataset import BaseDataset
from otx.v2.api.utils.decorators import add_subset_dataloader


class TestAddSubsetDataloader:
    """Test suite for the `add_subset_dataloader` decorator."""

    def test_adds_dataloader_methods_to_base_dataset(self) -> None:
        """Test that the `add_subset_dataloader` decorator adds dataloader methods to the base dataset."""
        @add_subset_dataloader(subsets=["train", "val"])
        class MyDataset(BaseDataset):
            def subset_dataloader(self, subset: str) -> str:
                return f"{subset} dataloader"

        dataset = MyDataset()
        assert dataset.train_dataloader() == "train dataloader"
        assert dataset.val_dataloader() == "val dataloader"

    def test_raises_not_implemented_error_if_subset_dataloader_not_implemented(self) -> None:
        """Test that the `add_subset_dataloader` decorator raises a `NotImplementedError` if the `subset_dataloader` function is not implemented."""
        with pytest.raises(NotImplementedError, match="In order to use this decorator, the class must have the subset_dataloader function implemented."):
            @add_subset_dataloader(subsets=["train", "val"])
            class MyDataset:
                pass
