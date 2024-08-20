# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Util functions related to ignored_labels."""

from __future__ import annotations

import torch

from otx.core.data.entity.base import ImageInfo


def get_valid_label_mask(img_metas: list[ImageInfo], num_classes: int) -> torch.Tensor:
    """Get valid label mask using ignored_label.

    Args:
        img_metas (list[ImageInfo]): The metadata of the input images.

    Returns:
        torch.Tensor: The valid label mask.
    """
    valid_label_mask = []
    for meta in img_metas:
        mask = torch.Tensor([1 for _ in range(num_classes)])
        if meta.ignored_labels:
            mask[meta.ignored_labels] = 0
        valid_label_mask.append(mask)
    return torch.stack(valid_label_mask, dim=0)
