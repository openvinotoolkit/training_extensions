from .electricity_dataset import *

DATASETS = {
    "electricity": ElectricityDataset
}

def get_dataset(dataset_name):
    assert dataset_name in DATASETS
    return DATASETS[dataset_name]
