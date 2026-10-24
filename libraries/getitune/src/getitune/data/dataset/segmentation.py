# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for SegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from getitune import SegLabelInfo
from getitune.data.dataset.base import Transforms, VisionDataset
from getitune.data.entity.sample import SegmentationSample
from getitune.data.entity.utils import with_image_dtype
from getitune.types import TaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class SegmentationDataset(VisionDataset):
    """getitune Dataset for semantic segmentation tasks.

    This dataset handles semantic segmentation where each pixel in an image is classified
    into one of multiple classes. It processes Datumaro dataset items and converts them
    into BaseSample format suitable for semantic segmentation training and inference.

    Args:
        dm_subset: Datumaro dataset subset containing the data items.
        transforms: Transform operations to apply to the data items.
        max_refetch: Maximum number of retries when fetching a data item fails.
        ignore_index: Index value for pixels to be ignored during training.
        data_format: Source data format (e.g. "coco", "voc"). Defaults to "".
        storage_dtype: Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".

    Attributes:
        ignore_index: Index value for pixels to be ignored during training.

    Example:
        >>> from getitune.data.dataset.segmentation import SegmentationDataset
        >>> dataset = SegmentationDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ...     ignore_index=255
        ... )
        >>> item = dataset[0]  # Get first item with segmentation masks
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        ignore_index: int = 255,
        data_format: str = "",
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(SegmentationSample, storage_dtype)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
        )

        labels = list(dm_subset.schema.attributes["masks"].categories.labels)
        self.label_info = SegLabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )

    @property
    def task_type(self) -> TaskType:
        """Getitune Task Type for the dataset.

        Returns:
            TaskType: The semantic segmentation task type.
        """
        return TaskType.SEMANTIC_SEGMENTATION
