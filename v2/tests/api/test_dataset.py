import pytest
import os

from otx import OTXConstants
from otx.api.dataset import Dataset

def test_dataset():
    data_path = os.path.join(OTXConstants.PACKAGE_ROOT, "..", "data")
    dataset = Dataset.create(os.path.join(data_path, "classification/train"), "imagenet", subset="train")
    assert isinstance(dataset, Dataset)
    print(dataset.categories())
    dataset_val = Dataset.create(os.path.join(data_path, "classification/val"), "imagenet", subset="val")
    assert isinstance(dataset_val, Dataset)

    dataset = dataset.update(dataset_val)
    print(f"categories = {dataset.categories()}")
    print(f"subsets = {dataset.subsets()}")

    train_set = dataset.get_subset("train")
    assert train_set is not None

    val_set = dataset.get_subset("val")
    assert val_set is not None
