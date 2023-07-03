"""Tests dataset and datamodule used for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from torch.utils.data import DataLoader

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.dataset import (
    OTXVisualPromptingDataModule,
    OTXVisualPromptingDataset,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import (
    collate_fn,
)
from otx.api.entities.image import Image
from otx.api.entities.shapes.polygon import Point, Polygon
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.visual_prompting.test_helpers import (
    MockDatasetConfig,
    generate_visual_prompting_dataset,
)
from otx.api.entities.datasets import DatasetEntity


class TestOTXVIsualPromptingDataset:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.transform = lambda items: items

    @pytest.fixture
    def dataset_polygon(self) -> DatasetEntity:
        """Set dataset with polygon."""
        return generate_visual_prompting_dataset(use_mask=False)

    @pytest.fixture
    def dataset_mask(self) -> DatasetEntity:
        """Set dataset with mask."""
        return generate_visual_prompting_dataset(use_mask=True)

    @e2e_pytest_unit
    def test_len(self, dataset_polygon) -> None:
        """Test __len__."""
        otx_dataset = OTXVisualPromptingDataset(dataset_polygon, self.transform)
        assert len(otx_dataset) == 4

    @e2e_pytest_unit
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_getitem(self, dataset_polygon, dataset_mask, use_mask: bool) -> None:
        """Test __getitem__."""
        dataset = dataset_mask if use_mask else dataset_polygon
        otx_dataset = OTXVisualPromptingDataset(dataset=dataset, transform=self.transform)

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

    @e2e_pytest_unit
    def test_convert_polygon_to_mask(self, dataset_polygon) -> None:
        """Test convert_polygon_to_mask."""
        otx_dataset = OTXVisualPromptingDataset(dataset_polygon, self.transform)

        polygon = Polygon(points=[Point(x=0.1, y=0.1), Point(x=0.2, y=0.2), Point(x=0.3, y=0.3)])
        width = 100
        height = 100

        mask = otx_dataset.convert_polygon_to_mask(polygon, width, height)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (height, width)
        assert mask.sum() == 21

    @e2e_pytest_unit
    def test_generate_bbox(self, dataset_polygon) -> None:
        """Test generate_bbox."""
        otx_dataset = OTXVisualPromptingDataset(dataset_polygon, self.transform)

        x1, y1, x2, y2 = 10, 20, 30, 40
        width = 100
        height = 100

        bbox = otx_dataset.generate_bbox(x1, y1, x2, y2, width, height)

        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert bbox[0] >= 0 and bbox[0] <= width
        assert bbox[1] >= 0 and bbox[1] <= height
        assert bbox[2] >= 0 and bbox[2] <= width
        assert bbox[3] >= 0 and bbox[3] <= height

    @e2e_pytest_unit
    def test_generate_bbox_from_mask(self, dataset_polygon) -> None:
        """Test generate_bbox_from_mask."""
        otx_dataset = OTXVisualPromptingDataset(dataset_polygon, self.transform)

        gt_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        width = 3
        height = 3

        bbox = otx_dataset.generate_bbox_from_mask(gt_mask, width, height)

        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert bbox[0] >= 0 and bbox[0] <= width
        assert bbox[1] >= 0 and bbox[1] <= height
        assert bbox[2] >= 0 and bbox[2] <= width
        assert bbox[3] >= 0 and bbox[3] <= height


class TestOTXVisualPromptingDataModule:
    @pytest.fixture
    def datamodule(self) -> OTXVisualPromptingDataModule:
        dataset = generate_visual_prompting_dataset()

        # Create a mock config
        config = MockDatasetConfig()

        # Create an instance of OTXVisualPromptingDataModule
        return OTXVisualPromptingDataModule(config, dataset)

    @e2e_pytest_unit
    def test_setup(self, mocker, datamodule) -> None:
        """Test setup."""
        mocker.patch.object(datamodule, "summary", return_value=None)

        datamodule.setup()

        assert isinstance(datamodule.train_dataset, OTXVisualPromptingDataset)
        assert isinstance(datamodule.val_dataset, OTXVisualPromptingDataset)

    @e2e_pytest_unit
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

    @e2e_pytest_unit
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

    @e2e_pytest_unit
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

    @e2e_pytest_unit
    def test_predict_dataloader(self, datamodule) -> None:
        """Test predict_dataloader."""
        datamodule.setup(stage="predict")

        # Call the predict_dataloader method
        dataloader = datamodule.predict_dataloader()

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 1
        assert dataloader.num_workers == datamodule.config.num_workers
        assert dataloader.collate_fn == collate_fn
