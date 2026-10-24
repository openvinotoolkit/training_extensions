# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for VisionDataset using new Datumaro experimental Dataset."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Union

import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import Compose
from torchvision.transforms.v2 import functional as f

from getitune import LabelInfo, NullLabelInfo
from getitune.data.augmentation.pipeline import CPUAugmentationPipeline
from getitune.data.entity.sample import BaseSample, SampleBatch
from getitune.types import TaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset

Transforms = Union[
    Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]], "CPUAugmentationPipeline"
]


def _ensure_chw_format(img: torch.Tensor) -> torch.Tensor:
    """Ensure image tensor is in CHW format with 3 channels.

    Args:
        img: Image tensor that may be in HWC, HCW, or CHW format

    Returns:
        Image tensor in CHW format (C, H, W) for 3D or (B, C, H, W) for 4D with 3 channels
    """
    # Handle 2D grayscale images (H, W) - add channel dimension
    if img.ndim == 2:
        img = img.unsqueeze(0)  # (H, W) -> (1, H, W)
        img = img.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
    elif img.ndim == 3:
        # Check if last dimension is likely channels (small value like 1, 3, or 4)
        # and first dimension is not (larger, like image height)
        if img.shape[-1] in (1, 3, 4) and img.shape[0] > 4:
            # HWC format detected, convert to CHW
            img = img.permute(2, 0, 1)
        # Check for HCW format: channels in the middle dimension
        elif img.shape[1] in (1, 3, 4) and img.shape[0] > 4 and img.shape[2] > 4:
            img = img.permute(1, 0, 2)
        # If 4 channels (RGBA), convert to 3 channels (RGB)
        if img.shape[0] == 4:
            img = img[:3]
        # If 1 channel (grayscale/palette), convert to 3 channels (RGB)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
    return img


def _collect_optional_attr(items: list[BaseSample], attr_name: str) -> list | None:
    if not items or not all(hasattr(item, attr_name) for item in items):
        return None
    values = [getattr(item, attr_name) for item in items]
    return values if any(value is not None for value in values) else None


def _default_collate_fn(items: list[BaseSample]) -> SampleBatch:
    """Collate BaseSample items into an SampleBatch.

    Args:
        items: List of BaseSample items to batch

    Returns:
        Batched BaseSample items with stacked tensors
    """
    # Convert images to float32 tensors before stacking
    image_tensors = []
    for item in items:
        img = item.image
        # All images should already be tensors from the pipeline
        if not isinstance(img, torch.Tensor):
            msg = (
                f"Expected torch.Tensor but got {type(img)}. "
                "Images should be converted to tensors in the dataset pipeline."
            )
            raise TypeError(msg)
        # Convert to float32 if not already.
        # For int32/int16 tensors (16-bit images) the intensity transform should
        # have already produced float32 in [0,1].  If we get here with an integer
        # dtype it means the intensity transform is missing — raise an error
        # instead of silently producing wrong values.
        if img.dtype != torch.float32:
            if img.dtype in (torch.int32, torch.int16, torch.int64):
                msg = (
                    f"Image tensor has dtype {img.dtype} which looks like a high-bit-depth image "
                    "that was not converted to float32. Please configure an intensity transform "
                    "(IntensityConfig) in the recipe to map raw pixel values to [0, 1] float32."
                )
                raise TypeError(msg)
            # uint8 → float32 [0, 1]
            img = img.float().div_(255.0)
        image_tensors.append(img)

    if len(image_tensors) == 0:
        msg = "No images found in batch. Ensure that the dataset and pipeline are configured correctly."
        raise ValueError(msg)
    images = torch.stack(image_tensors)

    # Convert labels to long
    labels = _collect_optional_attr(items, "label")
    if labels is not None:
        labels = [label.long() for label in labels]

    return SampleBatch(
        images=images,
        labels=labels,
        masks=_collect_optional_attr(items, "masks"),
        bboxes=_collect_optional_attr(items, "bboxes"),
        keypoints=_collect_optional_attr(items, "keypoints"),
        imgs_info=_collect_optional_attr(items, "img_info"),
    )


class VisionDataset(TorchDataset):
    """Base VisionDataset using new Datumaro experimental Dataset.

    This class defines the basic logic and interface for getitune datasets, providing
    functionality for data transformation, image decoding, and label handling.

    Args:
        dm_subset (Dataset): Datumaro subset of a dataset.
        transforms (Transforms, optional): Transformations to apply to the data.
        max_refetch (int, optional): Maximum number of times to attempt fetching a valid image. Defaults to 1000.

    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
    ) -> None:
        self.transforms = transforms
        self.max_refetch = max_refetch
        self.label_info: LabelInfo = NullLabelInfo()
        self.dm_subset = dm_subset

    def __len__(self) -> int:
        return len(self.dm_subset)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_samples={len(self)}, num_classes={self.label_info.num_classes})"

    def describe(self) -> dict[str, Any]:
        """Return a summary of the dataset.

        Returns:
            A dictionary containing dataset summary information.
        """
        return {
            "class_name": type(self).__name__,
            "num_samples": len(self),
            "num_classes": self.label_info.num_classes,
            "label_names": list(self.label_info.label_names),
            "has_transforms": self.transforms is not None,
        }

    def _apply_transforms(self, entity: BaseSample) -> BaseSample | None:
        # Intensity mapping: convert raw pixels to float32 [0, 1].
        #
        # When a CPUAugmentationPipeline is used the pipeline itself prepends
        # the correct intensity transform (built from IntensityConfig), so we
        # must NOT scale here — the intensity transform will do it.
        #
        # For legacy paths (Compose, callable) or when no transforms are set we
        # keep the original uint8-only scaling as a safe default.
        if not isinstance(self.transforms, CPUAugmentationPipeline):
            # Legacy path: always scale assuming uint8 input (backward-compat)
            entity.image = f.to_dtype(entity.image, dtype=torch.float32, scale=True)

        if self.transforms is None:
            return entity

        if isinstance(self.transforms, CPUAugmentationPipeline):
            return self.transforms(entity)

        # Legacy path: Compose
        if isinstance(self.transforms, Compose):
            return self.transforms(entity)

        # Legacy path: Iterable of transforms
        if isinstance(self.transforms, Iterable):
            return self._iterable_transforms(entity)

        # Legacy path: Single callable
        if callable(self.transforms):
            return self.transforms(entity)
        return None

    def _iterable_transforms(self, item: BaseSample) -> BaseSample | None:
        if not isinstance(self.transforms, list):
            raise TypeError(item)

        results = item
        for transform in self.transforms:
            results = transform(results)
            if results is None:
                return None

        return results

    def _read_dm_item(self, index: int) -> BaseSample:
        """Read an item from the datumaro subset with guaranteed CHW image format."""
        item = self.dm_subset[index]
        # Workaround for a datumaro bug: ``TensorField.from_polars()`` applies
        # ``np.transpose(data, (2, 0, 1))`` to undo the export transpose, but
        # the correct inverse of ``(2, 0, 1)`` is ``(1, 2, 0)``.  As a result,
        # images come back as HWC instead of the original CHW.
        item.image = _ensure_chw_format(item.image)
        return item

    def __getitem__(self, index: int) -> BaseSample:
        for _ in range(self.max_refetch):
            results = self._get_item_impl(index)

            if results is not None:
                return results

            index = torch.randint(0, len(self), (1,)).item()
        msg = f"Reach the maximum refetch number ({self.max_refetch})"
        raise RuntimeError(msg)

    def _get_item_impl(self, index: int) -> BaseSample | None:
        dm_item = self._read_dm_item(index)
        return self._apply_transforms(dm_item)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect samples into a batch in data loader."""
        return _default_collate_fn

    @abc.abstractmethod
    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels to lists of corresponding sample indices.

        Args:
            use_string_label (bool, optional): If True, use string labels as keys; otherwise, use integer labels.

        Returns:
            dict[int | str, list[int]]: A dictionary where each key is a class label (int or str) and each value is a
            list of sample indices belonging to that class.
        """

    @property
    def task_type(self) -> TaskType | None:
        """Getitune Task Type for the dataset. Can be None if no task is defined."""
        return None
