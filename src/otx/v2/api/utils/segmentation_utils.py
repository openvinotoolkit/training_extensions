"""This module implements segmentation related utilities."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import cv2
import numpy as np
from typing import List

from otx.v2.api.entities.annotation import Annotation
from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.label import LabelEntity
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.shapes.polygon import Polygon
from otx.v2.api.entities.utils.shape_factory import ShapeFactory


def mask_from_dataset_item(
    dataset_item: DatasetItemEntity, labels: List[LabelEntity], use_otx_adapter: bool = True
) -> np.ndarray:
    """Creates a mask from dataset item.

    The mask will be two dimensional, and the value of each pixel matches the class index with offset 1. The background
    class index is zero. labels[0] matches pixel value 1, etc. The class index is
    determined based on the order of 'labels'.

    Args:
        dataset_item: Item to make mask for
        labels: The labels to use for creating the mask. The order of
            the labels determines the class index.

    Returns:
        Numpy array of mask
    """
    # todo: cache this so that it does not have to be redone for all the same media
    if use_otx_adapter:
        mask = mask_from_annotation(dataset_item.get_annotations(), labels, dataset_item.width, dataset_item.height)
    else:
        mask = mask_from_file(dataset_item)
    return mask


def mask_from_file(dataset_item: DatasetItemEntity) -> np.ndarray:
    """Loads masks directly from annotation image.

    Only Common Sematic Segmentation format is supported.
    """

    mask_form_file = dataset_item.media.path
    if mask_form_file is None:
        raise ValueError("Mask file doesn't exist or corrupted")
    mask_form_file = mask_form_file.replace("images", "masks")
    mask = cv2.imread(mask_form_file, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims(mask, axis=2)
    return mask


def mask_from_annotation(
    annotations: List[Annotation], labels: List[LabelEntity], width: int, height: int
) -> np.ndarray:
    """Generate a segmentation mask of a numpy image, and a list of shapes.

    The mask is will be two dimensional and the value of each pixel matches the class
    index with offset 1. The background class index is zero. labels[0] matches pixel
    value 1, etc. The class index is determined based on the order of `labels`:

    Args:
        annotations: List of annotations to plot in mask
        labels: List of labels. The index position of the label
            determines the class number in the segmentation mask.
        width: Width of the mask
        height: Height of the mask

    Returns:
        2d numpy array of mask
    """

    mask = np.zeros(shape=(height, width), dtype=np.uint8)
    for annotation in annotations:
        shape = annotation.shape
        if not isinstance(shape, Polygon):
            shape = ShapeFactory.shape_as_polygon(annotation.shape)
        known_labels = [
            label for label in annotation.get_labels() if isinstance(label, ScoredLabel) and label.get_label() in labels
        ]
        if len(known_labels) == 0:
            # Skip unknown shapes
            continue

        label_to_compare = known_labels[0].get_label()

        class_idx = labels.index(label_to_compare) + 1
        contour = []
        for point in shape.points:
            contour.append([int(point.x * width), int(point.y * height)])

        mask = cv2.drawContours(mask, np.asarray([contour]), 0, (class_idx, class_idx, class_idx), -1)

    mask = np.expand_dims(mask, axis=2)

    return mask
