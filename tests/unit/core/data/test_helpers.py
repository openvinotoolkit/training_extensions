"""Test Helpers for otx.core.data."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Optional
import os

import cv2
import datumaro as dm
import numpy as np

from otx.api.entities.model_template import TaskType

TASK_NAME_TO_TASK_TYPE = {
    "classification": TaskType.CLASSIFICATION,
    "detection": TaskType.DETECTION,
    "rotated_detection": TaskType.ROTATED_DETECTION,
    "instance_segmentation": TaskType.INSTANCE_SEGMENTATION,
    "segmentation": TaskType.SEGMENTATION,
    "anomaly_classification": TaskType.ANOMALY_CLASSIFICATION,
    "anomaly_detection": TaskType.ANOMALY_DETECTION,
    "anomaly_segmentation": TaskType.ANOMALY_SEGMENTATION,
    "action_classification": TaskType.ACTION_CLASSIFICATION,
    "action_detection": TaskType.ACTION_DETECTION,
    "visual_prompting": TaskType.VISUAL_PROMPTING,
}

TASK_NAME_TO_DATA_ROOT = {
    "classification": {
        "train": "tests/assets/classification_dataset",
        "val": "tests/assets/classification_dataset",
        "test": "tests/assets/classification_dataset",
        "unlabeled": "tests/assets/classification_dataset",
    },
    "detection": {
        "train": "tests/assets/car_tree_bug",
        "val": "tests/assets/car_tree_bug",
        "test": "tests/assets/car_tree_bug",
        "unlabeled": "tests/assets/car_tree_bug",
    },
    "rotated_detection": {
        "train": "tests/assets/car_tree_bug",
        "val": "tests/assets/car_tree_bug",
        "test": "tests/assets/car_tree_bug",
        "unlabeled": "tests/assets/car_tree_bug",
    },
    "instance_segmentation": {
        "train": "tests/assets/car_tree_bug",
        "val": "tests/assets/car_tree_bug",
        "test": "tests/assets/car_tree_bug",
    },
    "segmentation": {
        "train": "tests/assets/common_semantic_segmentation_dataset/train",
        "val": "tests/assets/common_semantic_segmentation_dataset/val",
        "test": "tests/assets/common_semantic_segmentation_dataset/val",
        "unlabeled": "tests/assets/common_semantic_segmentation_dataset/val",
    },
    "anomaly_classification": {
        "train": "tests/assets/anomaly/hazelnut",
        "val": "tests/assets/anomaly/hazelnut",
        "test": "tests/assets/anomaly/hazelnut",
    },
    "anomaly_detection": {
        "train": "tests/assets/anomaly/hazelnut",
        "val": "tests/assets/anomaly/hazelnut",
        "test": "tests/assets/anomaly/hazelnut",
    },
    "anomaly_segmentation": {
        "train": "tests/assets/anomaly/hazelnut",
        "val": "tests/assets/anomaly/hazelnut",
        "test": "tests/assets/anomaly/hazelnut",
    },
    "action_classification": {
        "train": "tests/assets/cvat_dataset/action_classification/train",
        "val": "tests/assets/cvat_dataset/action_classification/train",
        "test": "tests/assets/cvat_dataset/action_classification/train",
    },
    "action_detection": {
        "train": "tests/assets/cvat_dataset/action_detection/train",
        "val": "tests/assets/cvat_dataset/action_detection/train",
        "test": "tests/assets/cvat_dataset/action_detection/train",
    },
    "visual_prompting": {
        "coco": {
            "train": "tests/assets/car_tree_bug",
            "val": "tests/assets/car_tree_bug",
            "test": "tests/assets/car_tree_bug",
        },
        "voc": {
            "train": "tests/assets/voc_dataset/voc_dataset1",
            "val": "tests/assets/voc_dataset/voc_dataset1",
            "test": "tests/assets/voc_dataset/voc_dataset1",
        },
        "common_semantic_segmentation": {
            "train": "tests/assets/common_semantic_segmentation_dataset/train",
            "val": "tests/assets/common_semantic_segmentation_dataset/val",
            "test": "tests/assets/common_semantic_segmentation_dataset/val",
        },
    },
}


def generate_datumaro_dataset_item(
    item_id: str,
    subset: str,
    task: str,
    image_shape: np.array = np.array((5, 5, 3)),
    mask_shape: np.array = np.array((5, 5)),
    temp_dir: Optional[str] = None,
) -> dm.DatasetItem:
    """Generate Datumaro DatasetItem.

    Args:
        item_id (str): The ID of dataset item
        subset (str): subset of item, e.g. "train" or "val"
        task (str): task type, e.g. "classification"
        image_shape (np.array): the shape of image.
        image_shape (np.array): the shape of mask.
        temp_dir (str): directory to save image data

    Returns:
        dm.DatasetItem: Datumaro DatasetItem
    """
    ann_task_dict = {
        "classification": dm.Label(label=0),
        "detection": dm.Bbox(1, 2, 3, 4, label=0),
        "segmentation": dm.Mask(np.zeros(mask_shape)),
    }

    if temp_dir:
        path = os.path.join(temp_dir, "image.png")
        cv2.imwrite(path, np.ones(image_shape))
        return dm.DatasetItem(id=item_id, subset=subset, image=path, annotations=[ann_task_dict[task]])

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
