# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Pre filtering data for OTX."""

from __future__ import annotations

import warnings
from random import sample
from typing import TYPE_CHECKING

from datumaro.components.annotation import Annotation, Bbox, Polygon
from datumaro.components.dataset import Dataset as DmDataset

if TYPE_CHECKING:
    from datumaro.components.dataset_base import DatasetItem


def pre_filtering(dataset: DmDataset, data_format: str, unannotated_items_ratio: float) -> DmDataset:
    """Pre-filtering function to filter the dataset based on certain criteria.

    Args:
        dataset (DmDataset): The input dataset to be filtered.
        data_format (str): The format of the dataset.
        unannotated_items_ratio (float): The ratio of background unannotated items to be used.
            This must be a float between 0 and 1.

    Returns:
        DmDataset: The filtered dataset.
    """
    used_background_items = set()
    msg = f"There are empty annotation items in train set, Of these, only {unannotated_items_ratio*100}% are used."
    warnings.warn(msg, stacklevel=2)
    if unannotated_items_ratio > 0:
        empty_items = [item.id for item in dataset if item.subset == "train" and len(item.annotations) == 0]
        used_background_items = set(sample(empty_items, int(len(empty_items) * unannotated_items_ratio)))

    dataset = DmDataset.filter(
        dataset,
        lambda item: not (
            item.subset == "train" and len(item.annotations) == 0 and item.id not in used_background_items
        ),
    )
    dataset = DmDataset.filter(dataset, is_valid_annot, filter_annotations=True)

    return remove_unused_labels(dataset, data_format)


def is_valid_annot(item: DatasetItem, annotation: Annotation) -> bool:  # noqa: ARG001
    """Return whether DatasetItem's annotation is valid."""
    if isinstance(annotation, Bbox):
        x1, y1, x2, y2 = annotation.points
        if x1 < x2 and y1 < y2:
            return True
        msg = "There are bounding box which is not `x1 < x2 and y1 < y2`, they will be filtered out before training."
        warnings.warn(msg, stacklevel=2)
        return False
    if isinstance(annotation, Polygon):
        # TODO(JaegukHyun): This process is computationally intensive.  # noqa: TD003
        # We should make pre-filtering user-configurable.
        x_points = [annotation.points[i] for i in range(0, len(annotation.points), 2)]
        y_points = [annotation.points[i + 1] for i in range(0, len(annotation.points), 2)]
        if min(x_points) < max(x_points) and min(y_points) < max(y_points) and annotation.get_area() > 0:
            return True
        msg = "There are invalid polygon, they will be filtered out before training."
        return False
    return True


def remove_unused_labels(dataset: DmDataset, data_format: str) -> DmDataset:
    """Remove unused labels in Datumaro dataset."""
    original_categories: list[str] = dataset.get_label_cat_names()
    used_labels: list[int] = list({ann.label for item in dataset for ann in item.annotations})
    if data_format == "ava":
        used_labels = [0, *used_labels]
    elif data_format == "common_semantic_segmentation_with_subset_dirs":
        if 0 in used_labels:
            used_labels = [label - 1 for label in used_labels[1:]]
        else:
            used_labels = [label - 1 for label in used_labels]
    if len(used_labels) == len(original_categories):
        return dataset
    msg = "There are unused labels in dataset, they will be filtered out before training."
    warnings.warn(msg, stacklevel=2)
    mapping = {original_categories[idx]: original_categories[idx] for idx in used_labels}
    return dataset.transform("remap_labels", mapping=mapping, default="delete")
