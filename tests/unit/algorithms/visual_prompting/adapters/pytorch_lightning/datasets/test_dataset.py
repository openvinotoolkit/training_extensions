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
    OTXVIsualPromptingDataset,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import (
    collate_fn,
)
from otx.api.entities.image import Image
from otx.api.entities.shapes.polygon import Point, Polygon
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockConfig:
    def __init__(self):
        self.image_size: Tuple[int] = (4, 4)
        self.use_mask: bool = False
        self.num_workers: int = 1
        self.train_batch_size: int = 1
        self.val_batch_size: int = 1
        self.test_batch_size: int = 1
        self.offset_bbox: int = 0


class MockAnnotation:
    def __init__(self, annotations):
        for k, v in annotations.items():
            setattr(self, k, v)

    def get_labels(self, *args, **kwargs):
        return self.labels


class MockDatasetItemEntity:
    def __init__(self, items):
        for k, v in items.items():
            setattr(self, k, v)

    def get_annotations(self, *args, **kwargs):
        for annotation in self.annotations:
            yield annotation

    @property
    def numpy(self):
        return self.media

    def __getitem__(self, index):
        return self.annotations[index]


class MockDatasetEntity:
    def __init__(self, items: Optional[List[MockDatasetItemEntity]] = None):
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def get_labels(self):
        return ['label1', 'label2']

    def get_subset(self, subset):
        return self


class MockMultipleInputsCompose:
    def __call__(self, item):
        return item
    

class TestOTXVIsualPromptingDataset:
    @pytest.fixture
    def dataset_polygon(self):
        # Create a mock dataset with some sample items
        items = [
            MockDatasetItemEntity(dict(
                media=Image(data=np.empty((4, 4))),
                path=None,
                width=4,
                height=4,
                annotations=[
                    MockAnnotation(annotations=dict(
                        shape=Polygon(points=[
                                Point(x=0.1, y=0.1),
                                Point(x=0.2, y=0.2),
                                Point(x=0.3, y=0.3)
                            ]),
                        labels=['label1']
                    ))
                ]
            ))
        ]
        return MockDatasetEntity(items)

    @pytest.fixture
    def dataset_mask(self):
        # Create a mock dataset with some sample items
        items = [
            MockDatasetItemEntity(dict(
                media=Image(data=np.empty((4, 4))),
                path=None,
                width=4,
                height=4,
                annotations=[
                    MockAnnotation(annotations=dict(
                        shape=Image(data=np.array([[0, 1, 1, 0] for _ in range(4)])),
                        labels=['label2']
                    ))
                ]
            ))
        ]
        return MockDatasetEntity(items)

    @e2e_pytest_unit
    def test_len(self, dataset_polygon):
        transform = MockMultipleInputsCompose()
        otx_dataset = OTXVIsualPromptingDataset(dataset_polygon, transform)
        assert len(otx_dataset) == 1

    @e2e_pytest_unit
    @pytest.mark.parametrize("use_mask,expected_labels",
        [
            (False, ["label1"]),
            (True, ["label2"])
        ]
    )
    def test_getitem(self, dataset_polygon, dataset_mask, use_mask: bool, expected_labels: List[str]):
        transform = MockMultipleInputsCompose()
        dataset = dataset_mask if use_mask else dataset_polygon
        otx_dataset = OTXVIsualPromptingDataset(
            dataset=dataset,
            transform=transform,
            use_mask=use_mask)

        item = otx_dataset[0]

        # Check the returned item's keys
        expected_keys = {'index', 'original_size', 'images', 'path', 'gt_masks', 'bboxes', 'points', 'labels'}
        assert set(item.keys()) == expected_keys

        # Check specific values in the item
        assert item['index'] == 0
        assert item['original_size'] == (4, 4)
        assert item['images'] == dataset[0].media
        assert item['path'] == dataset[0].path
        assert isinstance(item['gt_masks'], np.ndarray)
        assert isinstance(item['bboxes'], np.ndarray)
        assert item['points'] == []
        assert item['labels'] == expected_labels

    @e2e_pytest_unit
    def test_convert_polygon_to_mask(self, dataset_polygon):
        transform = MockMultipleInputsCompose()
        otx_dataset = OTXVIsualPromptingDataset(dataset_polygon, transform)

        polygon = Polygon(points=[
            Point(x=0.1, y=0.1),
            Point(x=0.2, y=0.2),
            Point(x=0.3, y=0.3)
        ])
        width = 100
        height = 100

        mask = otx_dataset.convert_polygon_to_mask(polygon, width, height)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (height, width)
        assert mask.sum() == 21

    @e2e_pytest_unit
    def test_generate_bbox(self, dataset_polygon):
        transform = MockMultipleInputsCompose()
        otx_dataset = OTXVIsualPromptingDataset(dataset_polygon, transform)

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
    def test_generate_bbox_from_mask(self, dataset_polygon):
        transform = MockMultipleInputsCompose()
        otx_dataset = OTXVIsualPromptingDataset(dataset_polygon, transform)

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
    def datamodule(self):
        # Create a mock dataset
        dataset = MockDatasetEntity()

        # Create a mock config
        config = MockConfig()

        # Create an instance of OTXVisualPromptingDataModule
        return OTXVisualPromptingDataModule(config, dataset)

    @e2e_pytest_unit
    def test_setup(self, mocker, datamodule):
        """Test setup."""
        mocker.patch.object(datamodule, "summary", return_value=None)

        datamodule.setup()

        assert isinstance(datamodule.train_dataset, OTXVIsualPromptingDataset)
        assert isinstance(datamodule.val_dataset, OTXVIsualPromptingDataset)

    @e2e_pytest_unit
    def test_train_dataloader(self, mocker, datamodule):
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
    def test_val_dataloader(self, mocker, datamodule):
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
    def test_test_dataloader(self, mocker, datamodule):
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
    def test_predict_dataloader(self, datamodule):
        """Test predict_dataloader."""
        datamodule.setup(stage="predict")

        # Call the predict_dataloader method
        dataloader = datamodule.predict_dataloader()

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 1
        assert dataloader.num_workers == datamodule.config.num_workers
        assert dataloader.collate_fn == collate_fn
