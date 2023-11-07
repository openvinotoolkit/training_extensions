"""Tests dataset and datamodule used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
from otx.v2.adapters.torch.lightning.modules.datasets.pipelines import (
    MultipleInputsCompose,
    Pad,
    ResizeLongestSide,
    collate_fn,
)
from otx.v2.adapters.torch.lightning.modules.datasets.visual_prompting_dataset import (
    OTXVisualPromptingDataModule,
    OTXVisualPromptingDataset,
    get_transform,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from tests.v2.unit.adapters.torch.lightning.test_helpers import (
    MockDatasetConfig,
    generate_visual_prompting_dataset,
)

if TYPE_CHECKING:
    from otx.v2.api.entities.datasets import DatasetEntity


@pytest.fixture()
def transform() -> Callable:
    return lambda items: items


@pytest.fixture()
def image_size() -> int:
    return 32


@pytest.fixture()
def mean() -> list[float]:
    return [1.0, 1.0, 1.0]


@pytest.fixture()
def std() -> list[float]:
    return [0.0, 0.0, 0.0]


@pytest.fixture()
def dataset_polygon() -> DatasetEntity:
    """Set dataset with polygon."""
    return generate_visual_prompting_dataset(use_mask=False)


@pytest.fixture()
def dataset_mask() -> DatasetEntity:
    """Set dataset with mask."""
    return generate_visual_prompting_dataset(use_mask=True)


def test_get_transform(image_size: int, mean: list[float], std: list[float]) -> None:
    """Test get_transform."""
    transform = get_transform(image_size=image_size, mean=mean, std=std)

    assert isinstance(transform, MultipleInputsCompose)
    assert isinstance(transform.transforms[0], ResizeLongestSide)
    assert transform.transforms[0].target_length == 32
    assert isinstance(transform.transforms[1], Pad)
    assert isinstance(transform.transforms[2], transforms.Normalize)
    assert transform.transforms[2].mean == mean
    assert transform.transforms[2].std == std


class TestOTXVIsualPromptingDataset:
    def test_len(self, mocker, dataset_polygon, transform, image_size, mean, std) -> None:
        """Test __len__."""
        mocker.patch(
            "otx.v2.adapters.torch.lightning.modules.datasets.visual_prompting_dataset.get_transform",
            return_value=transform,
        )
        otx_dataset = OTXVisualPromptingDataset(dataset_polygon, image_size, mean, std)
        assert len(otx_dataset) == 4

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_getitem(
        self, mocker, dataset_polygon, dataset_mask, transform, image_size, mean, std, use_mask: bool,
    ) -> None:
        """Test __getitem__."""
        mocker.patch(
            "otx.v2.adapters.torch.lightning.modules.datasets.visual_prompting_dataset.get_transform",
            return_value=transform,
        )
        dataset = dataset_mask if use_mask else dataset_polygon
        otx_dataset = OTXVisualPromptingDataset(dataset=dataset, image_size=image_size, mean=mean, std=std)

        item = otx_dataset[0]

        # Check the returned item's keys
        expected_keys = {"index", "original_size", "images", "path", "gt_masks", "bboxes", "points", "labels"}
        assert set(item.keys()) == expected_keys

        # Check specific values in the item
        assert item["index"] == 0
        assert (item["images"] == dataset[0].media.numpy).all()
        assert item["original_size"] == dataset[0].media.numpy.shape[:2]
        assert item["path"] == dataset[0].media.path
        assert isinstance(item["gt_masks"], list)
        assert isinstance(item["gt_masks"][0], np.ndarray)
        assert isinstance(item["bboxes"], np.ndarray)
        assert item["points"] == []


class TestOTXVisualPromptingDataModule:
    @pytest.fixture()
    def datamodule(self) -> OTXVisualPromptingDataModule:
        dataset = generate_visual_prompting_dataset()

        # Create a mock config
        config = MockDatasetConfig()

        # Create an instance of OTXVisualPromptingDataModule
        return OTXVisualPromptingDataModule(config, dataset)

    def test_setup(self, mocker, datamodule) -> None:
        """Test setup."""
        mocker.patch.object(datamodule, "summary", return_value=None)

        datamodule.setup()

        assert isinstance(datamodule.train_dataset, OTXVisualPromptingDataset)
        assert isinstance(datamodule.val_dataset, OTXVisualPromptingDataset)

    def test_train_dataloader(self, mocker, datamodule) -> None:
        """Test train_dataloader."""
        mocker.patch.object(datamodule, "summary", return_value=None)
        datamodule.setup(stage="fit")

        # Call the train_dataloader method
        dataloader = datamodule.train_dataloader()

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == datamodule.config.train_batch_size
        assert dataloader.num_workers == datamodule.config.num_workers
        assert dataloader.collate_fn == collate_fn

    def test_val_dataloader(self, mocker, datamodule) -> None:
        """Test val_dataloader."""
        mocker.patch.object(datamodule, "summary", return_value=None)
        datamodule.setup(stage="fit")

        # Call the val_dataloader method
        dataloader = datamodule.val_dataloader()

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == datamodule.config.val_batch_size
        assert dataloader.num_workers == datamodule.config.num_workers
        assert dataloader.collate_fn == collate_fn

    def test_test_dataloader(self, mocker, datamodule) -> None:
        """Test test_dataloader."""
        mocker.patch.object(datamodule, "summary", return_value=None)
        datamodule.setup(stage="test")

        # Call the test_dataloader method
        dataloader = datamodule.test_dataloader()

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == datamodule.config.test_batch_size
        assert dataloader.num_workers == datamodule.config.num_workers
        assert dataloader.collate_fn == collate_fn

    def test_predict_dataloader(self, datamodule) -> None:
        """Test predict_dataloader."""
        datamodule.setup(stage="predict")

        # Call the predict_dataloader method
        dataloader = datamodule.predict_dataloader()

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 1
        assert dataloader.num_workers == datamodule.config.num_workers
        assert dataloader.collate_fn == collate_fn
