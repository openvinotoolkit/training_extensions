import pytest

from otx.api.dataset import Dataset

def test_dataset():
    dataset = Dataset.create("/home/yunchu/data/cifar10", "cifar")

    assert isinstance(dataset, Dataset)
