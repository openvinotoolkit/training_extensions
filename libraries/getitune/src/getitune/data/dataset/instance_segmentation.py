# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for InstanceSegDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from getitune import LabelInfo
from getitune.data.dataset.base import Transforms, VisionDataset
from getitune.data.entity.sample import InstanceSegmentationSample
from getitune.data.entity.utils import with_image_dtype
from getitune.types import TaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class InstanceSegDataset(VisionDataset):
    """Dataset class for instance segmentation tasks in getitune.

    This class handles loading images and their masks.

    Args:
        dm_subset (DmDataset): The subset of the dataset to use.
        transforms (Transforms, optional): Data transformations to be applied.
        task_type (TaskType, optional): The task type. Defaults to INSTANCE_SEGMENTATION.
        max_refetch (int, optional): Maximum number of times to refetch data. Defaults to 1000.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".

    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        task_type: TaskType = TaskType.INSTANCE_SEGMENTATION,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(InstanceSegmentationSample, storage_dtype)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
        )

        labels = list(dm_subset.label_categories.labels)  # type: ignore[missing-attribute]
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )
        self._task_type = task_type

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

    @property
    def task_type(self) -> TaskType:
        """Getitune Task Type for the dataset.

        Returns:
            TaskType: The instance segmentation task type.
        """
        return self._task_type
