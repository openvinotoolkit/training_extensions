# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Copy from mmpretrain/models/utils/embed.py."""
import torch
from torch.nn import functional


def resize_pos_embed(
    pos_embed: torch.Tensor,
    src_shape: tuple,
    dst_shape: tuple,
    mode: str = "bicubic",
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, "shape of pos_embed must be [1, L, C]"  # noqa: S101
    _, L, C = pos_embed.shape  # noqa: N806
    src_h, src_w = src_shape
    assert src_h * src_w + num_extra_tokens == L, (  # noqa: S101
        f"The length of `pos_embed` ({L}) doesn't match the expected "
        f"shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the "
        "`img_size` argument."
    )
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = functional.interpolate(src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)
