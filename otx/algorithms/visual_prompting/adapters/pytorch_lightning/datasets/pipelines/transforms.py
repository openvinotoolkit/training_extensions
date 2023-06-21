"""Collection of transfrom pipelines for visual prompting task."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Tuple, Union, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image, pad  # type: ignore
from torchvision.transforms import Compose


def collate_fn(batch):
    def _convert_empty_to_none(x):
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
        return {"index": index, "images": images, "bboxes": bboxes, "points": points, "gt_masks": gt_masks, "original_size": original_size, "path": path, "labels": labels, "padding": padding}
    return {"index": -1, "images": [], "bboxes": [], "points": [], "gt_masks": [], "original_size": [], "path": [], "labels": [], "padding": []}


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.

    Args:
        target_length (int): ...
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __call__(self, item: Dict[str, Union[int, Tensor]]):
        item["images"] = torch.as_tensor(
            self.apply_image(item["images"]).transpose((2, 0, 1)),
            dtype=torch.get_default_dtype())
        item["gt_masks"] = [torch.as_tensor(self.apply_image(gt_mask)) for gt_mask in item["gt_masks"]]
        item["bboxes"] = self.apply_boxes(item["bboxes"], item["original_size"])
        if item["points"]:
            item["points"] = self.apply_coords(item["points"], item["original_size"])
        return item

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format."""
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """Expects a numpy array of length 2 in the final dimension.
        Requires the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """Expects a numpy array shape Bx4. Requires the original image size in (H, W) format."""
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class Pad:
    """"""
    def __call__(self, item: Dict[str, Union[int, Tensor]]):
        _, h, w = item["images"].shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

        item["padding"] = padding
        item["images"] = pad(item["images"], padding, fill=0, padding_mode="constant")
        item["gt_masks"] = [pad(gt_mask, padding, fill=0, padding_mode="constant") for gt_mask in item["gt_masks"]]
        item["bboxes"] = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in item["bboxes"]]
        if item["points"]:
            item["points"] = [[point[0] + pad_w, point[1] + pad_h, point[2] + pad_w, point[3] + pad_h] for point in item["points"]]
        return item


class MultipleInputsCompose(Compose):
    """Composes several transforms have multiple inputs together."""
    def __call__(self, item: Dict[str, Union[int, Tensor]]):
        for t in self.transforms:
            if isinstance(t, transforms.Normalize):
                item["images"] = t(item["images"])
            else:
                item = t(item)
        return item
