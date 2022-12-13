# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Create MVTec AD (CC BY-NC-SA 4.0) JSON Annotations for OTE CLI.

Description:
    This script converts MVTec AD dataset masks to OTE CLI annotation format for
        classification, detection and segmentation tasks.

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
      Unsupervised Anomaly Detection; in: International Journal of Computer Vision
      129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

    - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
      A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
      in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
      9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.

Example:
    Assume that MVTec AD dataset is located in "./data/anomaly/MVTec/" from the root
    directory in training_extensions. JSON annotations could be created by running the
    following:

    >>> import os
    '~/training_extensions'
    >>> os.listdir("./data/anomaly")
    ['detection', 'shapes', 'segmentation', 'MVTec', 'classification']

    The following script will generate the classification, detection and segmentation
    JSON annotations to each category in ./data/anomaly/MVTec dataset.

    >>> python external/anomaly/adapters/anomalib/data/create_mvtec_ad_json_annotations.py \
    ...     --data_path ./data/anomaly/MVTec/
"""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from adapters.anomalib.data.utils import (
    create_bboxes_from_mask,
    create_polygons_from_mask,
    save_json_items,
)
from anomalib.data.mvtec import make_mvtec_dataset


def create_json_items(pd_items: pd.DataFrame, task: str) -> Dict[str, Any]:
    """Create JSON items for the given task.

    Args:
        pd_items (pd.DataFrame): MVTec AD samples in pandas DataFrame object.
        task (str): Task name. One of "classification", "detection", "segmentation".

    Returns:
        Dict[str, Any]: MVTec AD JSON items
    """
    if task not in ("classification", "detection", "segmentation"):
        raise ValueError(f"Unsupported task: {task}")

    json_items: Dict[str, Any] = {"image_path": {}, "label": {}}
    if task in ("classification", "segmentation"):
        json_items["masks"] = {}
    else:  # detection
        json_items["bboxes"] = {}

    for index, pd_item in pd_items.iterrows():
        json_items["image_path"][str(index)] = pd_item.image_path.replace(pd_item.path, "")[1:]
        json_items["label"][str(index)] = pd_item.label
        if pd_item.label != "good":
            if task == "classification":
                json_items["masks"][str(index)] = pd_item.mask_path.replace(pd_item.path, "")[1:]
            elif task == "detection":
                json_items["bboxes"][str(index)] = create_bboxes_from_mask(pd_item.mask_path)
            else:  # segmentation
                json_items["masks"][str(index)] = create_polygons_from_mask(pd_item.mask_path)
    return json_items


def create_task_annotations(task: str, data_path: str, annotation_path: str) -> None:
    """Create MVTec AD categories for a given task.

    Args:
        task (str): Task type to save annotations.
        data_path (str): Path to MVTec AD category.
        annotation_path (str): Path to save MVTec AD category JSON annotation items.

    Raises:
        ValueError: When task is not classification, detection or segmentation.
    """
    annotation_path = os.path.join(annotation_path, task)
    os.makedirs(annotation_path, exist_ok=True)

    for split in ["train", "val", "test"]:
        df_items = make_mvtec_dataset(path=Path(data_path), create_validation_set=True, split=split)
        json_items = create_json_items(df_items, task)
        save_json_items(json_items, f"{annotation_path}/{split}.json")


def create_mvtec_ad_category_annotations(data_path: str, annotation_path: str) -> None:
    """Create MVTec AD category annotations for classification, detection and segmentation tasks.

    Args:
        data_path (str): Path to MVTec AD category.
        annotation_path (str): Path to save MVTec AD category JSON annotation items.
    """
    for task in ["classification", "detection", "segmentation"]:
        create_task_annotations(task, data_path, annotation_path)


def create_mvtec_ad_annotations(mvtec_data_path: str, mvtec_annotation_path: Optional[str] = None) -> None:
    """Create JSON annotations for MVTec AD dataset.

    Args:
        mvtec_data_path (str): Path to MVTec AD dataset.
        mvtec_annotation_path (Optional[str], optional): Path to save JSON annotations. Defaults to None.
    """
    if mvtec_annotation_path is None:
        mvtec_annotation_path = mvtec_data_path

    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    for category in categories:
        print(f"Creating annotations for {category}")
        category_data_path = os.path.join(mvtec_data_path, category)
        category_annotation_path = os.path.join(mvtec_annotation_path, category)
        create_mvtec_ad_category_annotations(category_data_path, category_annotation_path)


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/anomaly/MVTec/", help="Path to Mvtec AD dataset.")
    parser.add_argument("--annotation_path", type=str, required=False, help="Path to create OTE CLI annotations.")
    return parser.parse_args()


def main():
    """Create MVTec AD Annotations."""
    args = get_args()
    create_mvtec_ad_annotations(mvtec_data_path=args.data_path, mvtec_annotation_path=args.annotation_path)


if __name__ == "__main__":
    main()
