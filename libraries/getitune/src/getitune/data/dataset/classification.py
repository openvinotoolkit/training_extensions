# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for ClassificationDatasets using new Datumaro experimental Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from getitune import LabelInfo
from getitune.data.dataset.base import Transforms, VisionDataset
from getitune.data.entity.sample import (
    ClassificationMultiLabelSample,
    ClassificationSample,
)
from getitune.data.entity.utils import with_image_dtype
from getitune.types import TaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class MulticlassClsDataset(VisionDataset):
    """getitune Dataset for multi-class classification tasks.

    This dataset handles single-label classification where each image belongs to exactly one class.
    It processes Datumaro dataset items and converts them into BaseSample format suitable for
    multi-class classification training and inference.

    Args:
        dm_subset (Dataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transformations to apply to the data.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".


    Raises:
        ValueError: If an image has multiple labels (multi-label case).

    Example:
        >>> from getitune.data.dataset.classification import MulticlassClsDataset
        >>> dataset = MulticlassClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(ClassificationSample, storage_dtype)
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

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int | str, list[int]] = {}
        for idx in range(len(self)):
            item = self.dm_subset[idx]
            label_id = item.label.item()
            if use_string_label:
                label_id = self.label_info.label_names[label_id]
            if label_id not in idx_list_per_classes:
                idx_list_per_classes[label_id] = []
            idx_list_per_classes[label_id].append(idx)
        return idx_list_per_classes

    @property
    def task_type(self) -> TaskType:
        """Getitune Task Type for the dataset.

        Returns:
            TaskType: The multi-class classification task type.
        """
        return TaskType.MULTI_CLASS_CLS


class MultilabelClsDataset(VisionDataset):
    """getitune Dataset for multi-label classification tasks.

    This dataset handles multi-label classification where each image can belong to multiple classes
    simultaneously. It processes Datumaro dataset items and converts them into BaseSample format
    with one-hot encoded labels suitable for multi-label classification training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".


    Attributes:
        num_classes (int): Number of classes in the dataset.

    Example:
        >>> from getitune.data.dataset.classification import MultilabelClsDataset
        >>> dataset = MultilabelClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item with one-hot encoded labels
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(ClassificationMultiLabelSample, storage_dtype)
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
        self.num_classes = len(labels)

    def _get_item_impl(self, index: int) -> ClassificationMultiLabelSample | None:
        item = self._read_dm_item(index)
        item.label = self._convert_to_onehot(torch.as_tensor(list(item.label)), ignored_labels=[])
        return self._apply_transforms(item)

    def _convert_to_onehot(self, labels: torch.tensor, ignored_labels: list[int]) -> torch.tensor:
        """Convert label to one-hot vector format.

        Args:
            labels: Input label tensor to convert.
            ignored_labels: List of label indices to ignore.

        Returns:
            torch.tensor: One-hot encoded label tensor where ignored labels are set to -1.
        """
        # Torch's one_hot() expects the input to be of type long
        # However, when labels are empty, they are of type float32
        onehot = functional.one_hot(labels.long(), self.num_classes).sum(0).clamp_max_(1)
        if ignored_labels:
            for ignore_label in ignored_labels:
                onehot[ignore_label] = -1
        return onehot

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
            TaskType: The multi-label classification task type.
        """
        return TaskType.MULTI_LABEL_CLS
