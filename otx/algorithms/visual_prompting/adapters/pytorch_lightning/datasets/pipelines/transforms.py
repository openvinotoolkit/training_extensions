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
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from torchvision.transforms import Compose


def collate_fn(batch):
    index = [item['index'] for item in batch]
    image = torch.stack([item['image'] for item in batch])
    bbox = [torch.tensor(item['bbox']) for item in batch]
    mask = [torch.stack(item['mask']) for item in batch if item['mask'] != []]
    label = [item['label'] for item in batch] if batch else []
    if mask:
        return {'index': index, 'image': image, 'bbox': bbox, 'mask': mask, 'label': label}
    return {'index': -1, 'image': [], 'bbox': [], 'mask': [], 'label': []}


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
        item["image"] = torch.as_tensor(
            self.apply_image(item["image"]).transpose((2, 0, 1)),
            dtype=torch.get_default_dtype())
        item["mask"] = [torch.as_tensor(self.apply_image(mask)) for mask in item["mask"]]
        item["bbox"] = self.apply_boxes(item["bbox"], item["original_size"])
        if item["point"]:
            item["point"] = self.apply_coords(item["point"], item["original_size"])

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
        _, h, w = item["image"].shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

        item["image"] = transforms.functional.pad(item["image"], padding, fill=0, padding_mode="constant")
        item["mask"] = [transforms.functional.pad(mask, padding, fill=0, padding_mode="constant") for mask in item["mask"]]
        item["bbox"] = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in item["bbox"]]
        return item


class MultipleInputsCompose(Compose):
    """Composes several transforms have multiple inputs together."""
    def __call__(self, item: Dict[str, Union[int, Tensor]]):
        for t in self.transforms:
            if isinstance(t, transforms.Normalize):
                item["image"] = t(item["image"])
            else:
                item = t(item)
        return item
