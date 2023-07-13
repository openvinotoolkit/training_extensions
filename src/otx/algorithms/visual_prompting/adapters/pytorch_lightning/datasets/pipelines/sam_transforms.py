"""SAM transfrom pipeline for visual prompting task."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore


class ResizeLongestSide:
    """Resizes images to the longest side target_length, as well as provides methods for resizing coordinates and boxes.

    Provides methods for transforming both numpy array and batched torch tensors.

    Args:
        target_length (int): The length of the longest side of the image.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __call__(self, item: Dict[str, Union[List, Tensor]]) -> Dict[str, Union[List, Tensor]]:
        """Applies the transformation to a single sample.

        Args:
            item (Dict[str, Union[List, Tensor]]): Dictionary of batch data.

        Returns:
        Dict[str, Union[List, Tensor]]: Dictionary of batch data.
        """
        item["images"] = torch.as_tensor(
            self.apply_image(item["images"], self.target_length).transpose((2, 0, 1)), dtype=torch.get_default_dtype()
        )
        item["gt_masks"] = [torch.as_tensor(gt_mask) for gt_mask in item["gt_masks"]]
        item["bboxes"] = self.apply_boxes(item["bboxes"], item["original_size"])
        if item["points"]:
            item["points"] = self.apply_coords(item["points"], item["original_size"])
        return item

    @classmethod
    def apply_image(cls, image: np.ndarray, target_length: int) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format.

        Args:
            image (np.ndarray): Image array.
            target_length (int): The length of the longest side of the image.

        Returns:
            np.ndarray: Resized image.
        """
        target_size = cls.get_preprocess_shape(image.shape[0], image.shape[1], target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Union[List[Any], Tensor]) -> np.ndarray:
        """Expects a numpy array of length 2 in the final dimension.

        Requires the original image size in (H, W) format.

        Args:
            coords (np.ndarray): Coordinates array.
            original_size (Union[List[Any], Tensor]): Original size of image.

        Returns:
            np.ndarray: Resized coordinates.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Union[List[Any], Tensor]) -> np.ndarray:
        """Expects a numpy array shape Bx4. Requires the original image size in (H, W) format.

        Args:
            boxes (np.ndarray): Boxes array.
            original_size (Union[List[Any], Tensor]): Original size of image.

        Returns:
            np.ndarray: Resized boxes.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """Compute the output size given input size and target long side length.

        Args:
            oldh (int): Original height.
            oldw (int): Original width.
            long_side_length (int): Target long side length.

        Returns:
            Tuple[int, int]: Output size.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
