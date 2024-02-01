# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Pre filtering data for OTX."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from datumaro.components.annotation import Annotation, Bbox, Polygon
from datumaro.components.dataset import Dataset as DmDataset

if TYPE_CHECKING:
    from datumaro.components.dataset_base import DatasetItem


def pre_filtering(dataset: DmDataset) -> DmDataset:
    """Filtering invalid data in datumaro dataset."""
    dataset = DmDataset.filter(dataset, is_non_empty_item)
    dataset = DmDataset.filter(dataset, is_valid_annot, filter_annotations=True)

    return remove_unused_labels(dataset)


def is_non_empty_item(item: DatasetItem) -> bool:
    """Return whether DatasetItem's annotation is non-empty."""
    return not (item.subset == "train" and len(item.annotations) == 0)


def is_valid_annot(item: DatasetItem, annotation: Annotation) -> bool:
    """Return whether DatasetItem's annotation is valid."""
    del item
    if isinstance(annotation, Bbox):
        msg = "If there are bounding box which is not `x1 < x2 and y1 < y2`, they will be filtered out before training."
        warnings.warn(msg, stacklevel=2)
        x1, y1, x2, y2 = annotation.points
        return x1 < x2 and y1 < y2
    if isinstance(annotation, Polygon):
        x_points = [annotation.points[i] for i in range(0, len(annotation.points), 2)]
        y_points = [annotation.points[i + 1] for i in range(0, len(annotation.points), 2)]
        return min(x_points) < max(x_points) and min(y_points) < max(y_points) and annotation.get_area() > 0
    return True


def remove_unused_labels(dataset: DmDataset) -> DmDataset:
    """Remove unused labels in Datumaro dataset."""
    original_categories: list[str] = dataset.get_label_cat_names()
    used_labels: list[int] = list({ann.label for item in dataset for ann in item.annotations})
    mapping = {original_categories[idx]: original_categories[idx] for idx in used_labels}
    return dataset.transform("remap_labels", mapping=mapping, default="delete")
