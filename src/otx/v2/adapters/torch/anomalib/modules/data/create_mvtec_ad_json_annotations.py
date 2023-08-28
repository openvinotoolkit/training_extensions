# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Create MVTec AD (CC BY-NC-SA 4.0) JSON Annotations for OTX CLI.

Description:
    This script converts MVTec AD dataset masks to OTX CLI annotation format for
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

    >>> python external/anomaly/adapters.anomalib/data/create_mvtec_ad_json_annotations.py \
    ...     --data_path ./data/anomaly/MVTec/
"""

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
from anomalib.data.mvtec import make_mvtec_dataset
from anomalib.data.utils import Split


def create_bboxes_from_mask(mask_path: str) -> List[List[float]]:
    """Create bounding box from binary mask.

    Args:
        mask_path (str): Path to binary mask.

    Returns:
        List[List[float]]: Bounding box coordinates.
    """
    # pylint: disable-msg=too-many-locals

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    bboxes: List[List[float]] = []
    _, _, coordinates, _ = cv2.connectedComponentsWithStats(mask)
    for i, coordinate in enumerate(coordinates):
        # First row of the coordinates is always backround,
        # so should be ignored.
        if i == 0:
            continue

        # Last column of the coordinates is the area of the connected component.
        # It could therefore be ignored.
        comp_x, comp_y, comp_w, comp_h, _ = coordinate
        x1 = comp_x / width
        y1 = comp_y / height
        x2 = (comp_x + comp_w) / width
        y2 = (comp_y + comp_h) / height

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def create_polygons_from_mask(mask_path: str) -> List[List[List[float]]]:
    """Create polygons from binary mask.

    Args:
        mask_path (str): Path to binary mask.

    Returns:
        List[List[float]]: Polygon coordinates.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    polygons = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygons = [[[point[0][0] / width, point[0][1] / height] for point in polygon] for polygon in polygons]

    return polygons


def create_classification_json_items(pd_items: pd.DataFrame) -> Dict[str, Any]:
    """Create JSON items for the classification task.

    Args:
        pd_items (pd.DataFrame): MVTec AD samples in pandas DataFrame object.

    Returns:
        Dict[str, Any]: MVTec AD classification JSON items
    """
    json_items: Dict[str, Any] = {"image_path": {}, "label": {}, "masks": {}}
    for index, pd_item in pd_items.iterrows():
        json_items["image_path"][str(index)] = pd_item.image_path.replace(pd_item.path, "")[1:]
        json_items["label"][str(index)] = pd_item.label
        if pd_item.label != "good":
            json_items["masks"][str(index)] = pd_item.mask_path.replace(pd_item.path, "")[1:]

    return json_items


def create_detection_json_items(pd_items: pd.DataFrame) -> Dict[str, Any]:
    """Create JSON items for the detection task.

    Args:
        pd_items (pd.DataFrame): MVTec AD samples in pandas DataFrame object.

    Returns:
        Dict[str, Any]: MVTec AD detection JSON items
    """
    json_items: Dict[str, Any] = {"image_path": {}, "label": {}, "bboxes": {}}
    for index, pd_item in pd_items.iterrows():
        json_items["image_path"][str(index)] = pd_item.image_path.replace(pd_item.path, "")[1:]
        json_items["label"][str(index)] = pd_item.label
        if pd_item.label != "good":
            json_items["bboxes"][str(index)] = create_bboxes_from_mask(pd_item.mask_path)

    return json_items


def create_segmentation_json_items(pd_items: pd.DataFrame) -> Dict[str, Any]:
    """Create JSON items for the segmentation task.

    Args:
        pd_items (pd.DataFrame): MVTec AD samples in pandas DataFrame object.

    Returns:
        Dict[str, Any]: MVTec AD segmentation JSON items
    """
    json_items: Dict[str, Any] = {"image_path": {}, "label": {}, "masks": {}}
    for index, pd_item in pd_items.iterrows():
        json_items["image_path"][str(index)] = pd_item.image_path.replace(pd_item.path, "")[1:]
        json_items["label"][str(index)] = pd_item.label
        if pd_item.label != "good":
            json_items["masks"][str(index)] = create_polygons_from_mask(pd_item.mask_path)

    return json_items


def save_json_items(json_items: Dict[str, Any], file: str) -> None:
    """Save JSON items to file.

    Args:
        json_items (Dict[str, Any]): MVTec AD JSON items
        file (str): Path to save as a JSON file.
    """
    with open(file=file, mode="w", encoding="utf-8") as f:
        json.dump(json_items, f)


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

        if task == "classification":
            create_json_items = create_classification_json_items
        elif task == "detection":
            create_json_items = create_detection_json_items
        elif task == "segmentation":
            create_json_items = create_segmentation_json_items
        else:
            raise ValueError(f"Unknown task {task}. Available tasks are classification, detection and segmentation.")

        if split == "train":
            df_items = make_mvtec_dataset(root=Path(data_path), split=Split.TRAIN)
        else:
            df_items = make_mvtec_dataset(root=Path(data_path), split=Split.TEST)
        json_items = create_json_items(df_items)
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
    parser.add_argument("--annotation_path", type=str, required=False, help="Path to create OTX CLI annotations.")
    return parser.parse_args()


def main():
    """Create MVTec AD Annotations."""
    args = get_args()
    create_mvtec_ad_annotations(mvtec_data_path=args.data_path, mvtec_annotation_path=args.annotation_path)


if __name__ == "__main__":
    main()
