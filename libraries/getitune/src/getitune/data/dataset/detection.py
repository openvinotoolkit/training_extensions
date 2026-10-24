# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for DetectionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from getitune import TaskType
from getitune.data.dataset.base import Transforms, VisionDataset
from getitune.data.dataset.mixins import DataAugSwitchMixin
from getitune.data.entity.sample import DetectionSample
from getitune.data.entity.utils import with_image_dtype
from getitune.types.label import LabelInfo

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class DetectionDataset(VisionDataset, DataAugSwitchMixin):
    """getitune Dataset for object detection tasks.

    This dataset handles object detection where each image contains multiple objects with
    bounding box annotations. It processes Datumaro dataset items and converts them into
    BaseSample format suitable for object detection training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms | None, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".


    Example:
        >>> from getitune.data.dataset.detection import DetectionDataset
        >>> dataset = DetectionDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item with bounding boxes
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(DetectionSample, storage_dtype)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
        )
        labels = dm_subset.label_categories.labels  # type: ignore[missing-attribute]
        self.label_info = LabelInfo(
            label_names=list(labels),
            label_groups=[list(labels)],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int | str, list[int]] = {}
        for idx in range(len(self)):
            item = self.dm_subset[idx]
            labels = item.label.tolist()
            if use_string_label:
                labels = [self.label_info.label_names[label] for label in labels]
            for label in labels:
                if label not in idx_list_per_classes:
                    idx_list_per_classes[label] = []
                idx_list_per_classes[label].append(idx)
        return idx_list_per_classes

    def _apply_transforms(self, entity: DetectionSample) -> DetectionSample | None:
        if self.has_dynamic_augmentation:
            self._apply_augmentation_switch()
        return super()._apply_transforms(entity)

    @property
    def task_type(self) -> TaskType:
        """Getitune Task Type for the dataset.

        Returns:
            TaskType: The object detection task type.
        """
        return TaskType.DETECTION
