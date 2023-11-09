"""Collection of transfrom pipelines."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms.functional import pad


def collate_fn(batch: list) -> dict:
    """Collate function for dataloader.

    Args:
        batch (list): List of batch data.

    Returns:
        Dict: Collated batch data.
    """

    def _convert_empty_to_none(x: str) -> list | None:
        """Convert empty list to None.

        Args:
            x (str): Key of batch data.

        Returns:
            list | None: List of batch data.
        """
        func = torch.stack if x == "gt_masks" else torch.tensor
        items = [func(item[x]) for item in batch if np.asarray(item[x]).size != 0]
        return None if len(items) == 0 else items

    index = [item["index"] for item in batch]
    images = torch.stack([item["images"] for item in batch])
    bboxes = _convert_empty_to_none("bboxes")
    points = None  # TBD
    gt_masks = _convert_empty_to_none("gt_masks")
    original_size = [item["original_size"] for item in batch]
    path = [item["path"] for item in batch]
    padding = [item["padding"] for item in batch]
    labels = [item["labels"] for item in batch]
    if gt_masks:
        return {
            "index": index,
            "images": images,
            "bboxes": bboxes,
            "points": points,
            "gt_masks": gt_masks,
            "original_size": original_size,
            "path": path,
            "labels": labels,
            "padding": padding,
        }
    return {
        "index": -1,
        "images": [],
        "bboxes": [],
        "points": [],
        "gt_masks": [],
        "original_size": [],
        "path": [],
        "labels": [],
        "padding": [],
    }


class Pad:
    """Pad images, gt_masks, bboxes, and points to the same size."""

    def __call__(self, item: dict[str, Tensor | tuple]) -> dict[str, Tensor | tuple]:
        """Pad images, gt_masks, bboxes, and points to the same size.

        Args:
            item (dict[str, Tensor | tuple]): Input item.

        Returns:
            dict[str, Tensor | tuple]: Padded item.
        """
        if isinstance(item["images"], Tensor):
            _, h, w = item["images"].shape
        else:
            _, h, w = item["images"][0].shape
        max_dim = max(w, h)
        pad_w = max_dim - w
        pad_h = max_dim - h
        padding = (0, 0, pad_w, pad_h)

        item["padding"] = padding
        item["images"] = pad(item["images"], padding, fill=0, padding_mode="constant")

        return item


class MultipleInputsCompose(Compose):
    """Composes several transforms have multiple inputs together."""

    def __call__(self, item: dict[str, int | Tensor]) -> dict[str, int | Tensor]:
        """Composes several transforms have multiple inputs together.

        Args:
            item (dict[str, int | Tensor]): Input item.

        Returns:
            dict[str, int | Tensor]: Transformed item.
        """
        for t in self.transforms:
            if isinstance(t, transforms.Normalize):
                item["images"] = t(item["images"])
            else:
                item = t(item)
        return item
