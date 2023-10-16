# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import math

from otx.v2.api.core.dataset import BaseDataset
from torch.utils.data.dataloader import DataLoader

TASK_CONFIGURATION = {
    "classification": {
        "train_data_roots": "tests/assets/classification_dataset",
        "val_data_roots": "tests/assets/classification_dataset",
        "test_data_roots": "tests/assets/classification_dataset",
        "sample": "tests/assets/classification_dataset/0/11.jpg",
    },
    "anomaly_classification": {
        "train_data_roots": "tests/assets/anomaly/hazelnut/train",
        "val_data_roots": "tests/assets/anomaly/hazelnut/test",
        "test_data_roots": "tests/assets/anomaly/hazelnut/test",
        "sample": "tests/assets/anomaly/hazelnut/test/colour/01.jpg",
    },
}


def assert_torch_dataset_api_is_working(dataset: BaseDataset, train_data_size: int, val_data_size: int, test_data_size: int) -> None:
    """
    Asserts that the PyTorch dataset API is working correctly by checking the length and type of the dataloaders
    generated by the dataset object.

    Args:
        dataset (BaseDataset): The PyTorch dataset object to test.
        data_size (int): The expected size of the dataset.

    Returns:
        None
    """
    # Train Dataloader batch = 1
    train_dataloader = dataset.train_dataloader(batch_size=1)
    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader.dataset) == train_data_size
    assert len(train_dataloader) == train_data_size

    # Train Dataloader batch = 2
    train_dataloader = dataset.train_dataloader(batch_size=2, drop_last=False)
    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader) == math.ceil(train_data_size / 2)

    # Validation Dataloader
    val_dataloader = dataset.val_dataloader(batch_size=1)
    assert isinstance(val_dataloader, DataLoader)
    assert len(val_dataloader.dataset) == val_data_size
    assert len(val_dataloader) == val_data_size

    # Test Dataloader
    test_dataloader = dataset.test_dataloader(batch_size=1)
    assert isinstance(test_dataloader, DataLoader)
    assert len(test_dataloader.dataset) == test_data_size
    assert len(test_dataloader) == test_data_size
