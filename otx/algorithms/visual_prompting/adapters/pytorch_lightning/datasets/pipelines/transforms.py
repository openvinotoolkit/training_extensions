"""Collection of transfrom pipelines for visual prompting task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, List, Union

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms.functional import pad  # type: ignore


def collate_fn(batch: List[Any]) -> Dict:
    """Collate function for dataloader.

    Args:
        batch (List): List of batch data.

    Returns:
        Dict: Collated batch data.
    """

    def _convert_empty_to_none(x: str) -> List:
        """Convert empty list to None.

        Args:
            x (str): Key of batch data.

        Returns:
            List: List of batch data.
        """
        func = torch.stack if x == "gt_masks" else torch.tensor
        items = [func(item[x]) for item in batch if item[x]]
        return None if len(items) == 0 else items

    index = [item["index"] for item in batch]
    images = torch.stack([item["images"] for item in batch])
    bboxes = _convert_empty_to_none("bboxes")
    points = _convert_empty_to_none("points")
    gt_masks = _convert_empty_to_none("gt_masks")
    original_size = [item["original_size"] for item in batch]
    padding = [item["padding"] for item in batch]
    path = [item["path"] for item in batch]
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

    def __call__(self, item: Dict[str, Union[List[Any], Tensor]]) -> Dict[str, Union[int, Tensor]]:
        """Pad images, gt_masks, bboxes, and points to the same size.

        Args:
            item (Dict[str, Union[int, Tensor]]): Input item.

        Returns:
            Dict[str, Union[int, Tensor]]: Padded item.
        """
        _, h, w = item["images"].shape  # type: ignore
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

        item["padding"] = padding
        item["images"] = pad(item["images"], padding, fill=0, padding_mode="constant")
        item["gt_masks"] = [pad(gt_mask, padding, fill=0, padding_mode="constant") for gt_mask in item["gt_masks"]]
        item["bboxes"] = [
            [bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in item["bboxes"]
        ]
        if item["points"]:
            item["points"] = [
                [point[0] + pad_w, point[1] + pad_h, point[2] + pad_w, point[3] + pad_h] for point in item["points"]
            ]
        return item


class MultipleInputsCompose(Compose):
    """Composes several transforms have multiple inputs together."""

    def __call__(self, item: Dict[str, Union[int, Tensor]]) -> Dict[str, Union[int, Tensor]]:
        """Composes several transforms have multiple inputs together.

        Args:
            item (Dict[str, Union[int, Tensor]]): Input item.

        Returns:
            Dict[str, Union[int, Tensor]]: Transformed item.
        """
        for t in self.transforms:
            if isinstance(t, transforms.Normalize):
                item["images"] = t(item["images"])
            else:
                item = t(item)
        return item
