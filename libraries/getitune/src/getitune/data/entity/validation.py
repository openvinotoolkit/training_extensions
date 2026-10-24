# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validation functions for getitune data entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from getitune.data.entity.base import ImageInfo

if TYPE_CHECKING:
    from torchvision.tv_tensors import BoundingBoxes, Mask


def validate_images(image_batch: torch.Tensor | list[torch.Tensor]) -> None:
    """Validate the image batch."""
    if not isinstance(image_batch, (torch.Tensor, list)):
        msg = f"Image batch must be a torch.Tensor or a list of torch.Tensors. Got {type(image_batch)}"
        raise TypeError(msg)
    if isinstance(image_batch, torch.Tensor):
        if image_batch.dtype != torch.float32:
            msg = f"Image batch must have dtype float32. Found {image_batch.dtype}"
            raise ValueError(msg)
        if image_batch.ndim != 4:
            msg = f"Image batch must have 4 dimensions (BCHW), got {image_batch.ndim}"
            raise ValueError(msg)
        if image_batch.shape[1] not in [1, 3]:
            msg = f"Image batch must have 1 or 3 channels, got {image_batch.shape[1]}"
            raise ValueError(msg)
    if isinstance(image_batch, list):
        if not all(isinstance(img, torch.Tensor) for img in image_batch):
            msg = "All items in image batch list must be torch.Tensors"
            raise TypeError(msg)
        if not all(img.dtype == torch.float32 for img in image_batch):
            msg = "All images in batch must have dtype float32"
            raise ValueError(msg)
        if not all(img.ndim == 3 for img in image_batch):
            msg = "All images in batch must have 3 dimensions (CHW)"
            raise ValueError(msg)
        if not all(img.shape[0] in [1, 3] for img in image_batch):
            msg = "All images in batch must have 1 or 3 channels"
            raise ValueError(msg)


def validate_labels(label_batch: list[torch.Tensor | None]) -> None:
    """Validate the label batch."""
    if all(label is None for label in label_batch):
        return
    first_non_none = next((label for label in label_batch if label is not None), None)
    if first_non_none is None:
        return
    if not isinstance(first_non_none, torch.Tensor):
        msg = f"Label batch must be a list of torch tensors. Got {type(first_non_none)}"
        raise TypeError(msg)
    if first_non_none.dtype != torch.long:
        msg = "Label batch must have dtype torch.long"
        raise ValueError(msg)
    if first_non_none.ndim > 2:
        msg = f"Label batch must have shape of (N, 1) or (N,), but got {first_non_none.shape}"
        raise ValueError(msg)


def validate_bboxes(boxes_batch: list[BoundingBoxes | None]) -> None:
    """Validate the bboxes batch."""
    if all(box is None for box in boxes_batch):
        return
    first_non_none = next((box for box in boxes_batch if box is not None), None)
    if first_non_none is None:
        return
    if not isinstance(first_non_none, torch.Tensor):
        msg = f"Boxes batch must be a list of torch tensors. Got {type(first_non_none)}"
        raise TypeError(msg)
    if not first_non_none.dtype.is_floating_point:
        msg = f"Boxes batch must have a floating point dtype. Got {first_non_none.dtype}"
        raise ValueError(msg)
    if first_non_none.ndim != 2:
        msg = "Boxes batch must have 2 dimensions"
        raise ValueError(msg)
    if first_non_none.shape[1] != 4:
        msg = "Boxes batch must have 4 coordinates"
        raise ValueError(msg)


def validate_keypoints(keypoints_batch: list[torch.Tensor | None]) -> None:
    """Validate the keypoints batch."""
    if all(keypoints is None for keypoints in keypoints_batch):
        return
    first_non_none = next((kp for kp in keypoints_batch if kp is not None), None)
    if first_non_none is None:
        return
    if not isinstance(first_non_none, torch.Tensor):
        msg = f"Keypoints batch must be a list of torch tensors. Got {type(first_non_none)}"
        raise TypeError(msg)
    if first_non_none.dtype != torch.float32:
        msg = "Keypoints batch must have dtype torch.float32"
        raise ValueError(msg)
    if first_non_none.ndim != 2:
        msg = "Keypoints batch must have 2 dimensions"
        raise ValueError(msg)
    if first_non_none.shape[1] != 3:
        msg = "Keypoints batch must have 2 coordinates and 1 visibility value"
        raise ValueError(msg)
    if any(first_non_none[:, 2] > 1) or any(first_non_none[:, 2] < 0):
        msg = "Keypoints visibility must be between 0 and 1"
        raise ValueError(msg)


def validate_masks(masks_batch: list[Mask | None]) -> None:
    """Validate the masks batch."""
    if all(mask is None for mask in masks_batch):
        return
    first_non_none = next((mask for mask in masks_batch if mask is not None), None)
    if first_non_none is None:
        return
    if not isinstance(first_non_none, torch.Tensor):
        msg = f"Masks batch must be a list of torch tensors. Got {type(first_non_none)}"
        raise TypeError(msg)


def validate_imgs_info(imgs_info_batch: Sequence[ImageInfo | None]) -> None:
    """Validate the image info batch."""
    if all(img_info is None for img_info in imgs_info_batch):
        return
    first_non_none = next((info for info in imgs_info_batch if info is not None), None)
    if first_non_none is None:
        return
    if not isinstance(first_non_none, ImageInfo):
        msg = "Image info batch must be a list of getitune.data.entity.ImageInfo"
        raise TypeError(msg)


def validate_scores(scores_batch: list[torch.Tensor | None]) -> None:
    """Validate the scores batch."""
    if all(score is None for score in scores_batch):
        return
    first_non_none = next((score for score in scores_batch if score is not None), None)
    if first_non_none is None:
        return
    if not isinstance(first_non_none, torch.Tensor):
        msg = f"Scores batch must be a list of torch tensors. Got {type(first_non_none)}"
        raise TypeError(msg)
    if not first_non_none.dtype.is_floating_point:
        msg = f"Scores batch must have a floating point dtype. Got {first_non_none.dtype}"
        raise ValueError(msg)
    if first_non_none.ndim > 1:
        msg = "Scores batch must have 1 or 2 dimensions"
        raise ValueError(msg)


def validate_feature_vectors(
    feature_vector_batch: list[torch.Tensor | np.ndarray | None],
) -> None:
    """Validate the feature vector batch.

    Numpy is mixed for this round as it is used in OV Classification.
    """
    first_non_none = next((fv for fv in feature_vector_batch if fv is not None), None)
    if first_non_none is None:
        return
    if not isinstance(first_non_none, (torch.Tensor, np.ndarray)):
        msg = f"Feature vector batch must be a list of torch tensors or numpy arrays. Got {type(first_non_none)}"
        raise TypeError(msg)
    if isinstance(first_non_none, torch.Tensor) and not first_non_none.dtype.is_floating_point:
        msg = f"Feature vector must have a floating point dtype. Got {first_non_none.dtype}"
        raise ValueError(msg)
    if isinstance(first_non_none, np.ndarray) and first_non_none.dtype.kind != "f":
        msg = f"Feature vector must have a floating point dtype. Got {first_non_none.dtype}"
        raise ValueError(msg)
    if isinstance(first_non_none, torch.Tensor) and first_non_none.ndim != 2:
        msg = "Feature vector must have 2 dimensions"
        raise ValueError(msg)
