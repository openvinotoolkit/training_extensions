from typing import List

import datumaro as dm
import numpy as np


def generate_datumaro_dataset_item(
    item_id: str,
    subset: str,
    task: str,
    image_shape: np.array = np.array((5, 5, 3)),
    mask_shape: np.array = np.array((5, 5)),
) -> dm.DatasetItem:
    """Generate Datumaro DatasetItem.

    Args:
        item_id (str): The ID of dataset item
        subset (str): subset of item, e.g. "train" or "val"
        task (str): task type, e.g. "classification"
        image_shape (np.array): the shape of image.
        image_shape (np.array): the shape of mask.

    Returns:
        dm.DatasetItem: Datumaro DatasetItem
    """
    ann_task_dict = {
        "classification": dm.Label(label=0),
        "detection": dm.Bbox(1, 2, 3, 4, label=0),
        "segmentation": dm.Mask(np.zeros(mask_shape)),
    }

    return dm.DatasetItem(id=item_id, subset=subset, image=np.ones(image_shape), annotations=[ann_task_dict[task]])


def generate_datumaro_dataset(
    subsets: List[str],
    task: str,
    num_data: int = 1,
    image_shape: np.array = np.array((5, 5, 3)),
    mask_shape: np.array = np.array((5, 5)),
) -> dm.Dataset:
    """Generate Datumaro Dataset.

    Args:
        subsets (List): the list of subset, e.g. ["train", "val"]
        task (str): task name, e.g. "classification", "segmentation", ..
        num_data (int): the number of dataset to make.
        image_shape (np.array): the shape of image.
        mask_shape (np.array): the shape of mask.

    Returns:
        dm.Dataset: Datumaro Dataset
    """
    dataset_items: dm.DatasetItem = []
    for subset in subsets:
        for idx in range(num_data):
            dataset_items.append(
                generate_datumaro_dataset_item(
                    item_id=f"{subset}/image{idx}",
                    subset=subset,
                    task=task,
                    image_shape=image_shape,
                    mask_shape=mask_shape,
                )
            )
    return dm.Dataset.from_iterable(dataset_items, categories=["cat", "dog"])
