"""Channel shuffle method."""
# Copyright (c) 2018-2020 Open-MMLab.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
                      in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """
    batch_size, num_channels, height, width = x.size()
    if num_channels % groups != 0:
        msg = "num_channels should be divisible by groups"
        raise ValueError(msg)

    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    return x.view(batch_size, -1, height, width)
