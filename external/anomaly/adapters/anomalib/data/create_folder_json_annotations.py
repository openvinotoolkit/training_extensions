"""Create JSON Annotations for OTE CLI from Folder Dataset.

Example:
    Assume that Folder dataset is located in "./data/anomaly/Hazelnut_Toy/" from the root
    directory in training_extensions. JSON annotations could be created by running the
    following:

    The following script will generate the classification, detection and segmentation
    JSON annotations to each category in ./data/anomaly/Hazelnut_Toy dataset.

    >>> python external/anomaly/adapters/anomalib/data/create_folder_json_annotations.py \
    ... --data_path ./data/anomaly/Hazelnut_Toy/hazelnut \
    ... --annotation_path ./data/anomaly/Hazelnut_Toy/ \
    ... --normal_dir good \
    ... --abnormal_dir colour \
"""


# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Optional

import pandas as pd
from adapters.anomalib.data.utils import (
    create_bboxes_from_mask,
    create_polygons_from_mask,
    save_json_items,
)
from anomalib.data.folder import make_dataset


def create_json_items(pd_items: pd.DataFrame, data_root: str, task: str) -> Dict[str, Any]:
    """Create JSON items for the given task.

    Args:
        pd_items (pd.DataFrame): Folder samples in pandas DataFrame object.
        data_root (str): Path to the folder containing the dataset.
        task (str): Task name. One of "classification", "detection", "segmentation".

    Returns:
        Dict[str, Any]: Folder segmentation JSON items
    """
    if task not in ("classification", "detection", "segmentation"):
        raise ValueError(f"Unsupported task: {task}")
    
    json_items: Dict[str, Any] = {"image_path": {}, "label": {}}
    if task in ("classification", "segmentation"):
        json_items["masks"] = {}
    else:  # detection
        json_items["bboxes"] = {}

    
    for index, pd_item in pd_items.iterrows():
        json_items["image_path"][str(index)] = pd_item.image_path.replace(data_root, "")[1:]
        json_items["label"][str(index)] = pd_item.label if pd_item.label != "normal" else "good"
        if pd_item.label != "normal":
            if task == "classification":
                json_items["masks"][str(index)] = pd_item.mask_path.replace(data_root, "")[1:]
            elif task == "detection":
                json_items["bboxes"][str(index)] = create_bboxes_from_mask(pd_item.mask_path)
            else:  # segmentation
                json_items["masks"][str(index)] = create_polygons_from_mask(pd_item.mask_path)

    return json_items


def create_task_annotations(
    task: str, data_path: str, annotation_path: str, normal_dir: str, abnormal_dir: str
) -> None:
    """Generate annotations for a given task.

    Args:
        task (str): Task type to save annotations.
        data_path (str): Path to MVTec AD category.
        annotation_path (str): Path to save MVTec AD category JSON annotation items.
        normal_dir (str): Path to normal samples.
        abnormal_dir (str): Path to abnormal samples.

    Raises:
        ValueError: When task is not classification, detection or segmentation.
    """
    annotation_path = os.path.join(annotation_path, task)
    os.makedirs(annotation_path, exist_ok=True)

    for split in ["train", "val", "test"]:
        df_items = make_dataset(
            normal_dir=data_path + f"/{normal_dir}",
            abnormal_dir=data_path + f"/{abnormal_dir}",
            split=split,
            mask_dir=data_path + f"/mask/{abnormal_dir}",
            seed=42,
        )

        json_items = create_json_items(df_items, data_path, task)
        save_json_items(json_items, f"{annotation_path}/{split}.json")


def create_folder_annotations(
    data_path: str, normal_dir: str, abnormal_dir: str, annotation_path: Optional[str] = None
) -> None:
    """Create JSON annotations for Folder dataset.

    Args:
        data_path (str): Path to Folder dataset.
        annotation_path (Optional[str], optional): Path to save JSON annotations. Defaults to None.
        normal_dir (str): Path to normal samples.
        abnormal_dir (str): Path to abnormal samples.
    """
    if annotation_path is None:
        annotation_path = data_path

    for task in ["classification", "detection", "segmentation"]:
        create_task_annotations(task, data_path, annotation_path, normal_dir, abnormal_dir)


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/anomaly/MVTec/", help="Path to Mvtec AD dataset.")
    parser.add_argument("--annotation_path", type=str, required=False, help="Path to create OTE CLI annotations.")
    parser.add_argument("--normal_dir", type=str, required=True, help="Path to normal samples.")
    parser.add_argument("--abnormal_dir", type=str, required=True, help="Path to abnormal samples.")
    return parser.parse_args()


def main():
    """Create Folder Annotations."""
    args = get_args()
    create_folder_annotations(
        data_path=args.data_path,
        annotation_path=args.annotation_path,
        normal_dir=args.normal_dir,
        abnormal_dir=args.abnormal_dir,
    )


if __name__ == "__main__":
    main()
