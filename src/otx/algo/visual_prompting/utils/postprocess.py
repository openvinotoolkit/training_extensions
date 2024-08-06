# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-process methods for the OTX visual prompting."""

import torch
from torch import Tensor, nn


def postprocess_masks(masks: Tensor, input_size: int, orig_size: Tensor) -> Tensor:
    """Postprocess the predicted masks.

    Args:
        masks (Tensor): A batch of predicted masks with shape Bx1xHxW.
        input_size (int): The size of the image input to the model, in (H, W) format.
            Used to remove padding.
        orig_size (Tensor): The original image size with shape Bx2.

    Returns:
        masks (Tensor): The postprocessed masks with shape Bx1xHxW.
    """
    orig_size = orig_size.squeeze()
    masks = nn.functional.interpolate(masks, size=(input_size, input_size), mode="bilinear", align_corners=False)

    prepadded_size = get_prepadded_size(orig_size, input_size)  # type: ignore[arg-type]
    masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

    orig_size = orig_size.to(torch.int64)
    h, w = orig_size[0], orig_size[1]
    return nn.functional.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)


def get_prepadded_size(input_image_size: Tensor, longest_side: int) -> Tensor:
    """Get pre-padded size."""
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    return torch.floor(transformed_size + 0.5).to(torch.int64)
