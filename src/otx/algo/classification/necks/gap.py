# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""GlobalAveragePooling Implementation.

The original source code is mmpretrain.models.necks.gap.GlobalAveragePooling.
you can refer https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/necks/gap.py

"""

from __future__ import annotations

import torch
from torch import nn


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        self.gap: nn.Module
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self) -> None:
        """Initializes the weights of the model.

        This method is responsible for initializing the weights of the model's layers.
        """

    def forward(self, inputs: tuple | torch.Tensor) -> tuple | torch.Tensor:
        """Forward pass of the GAP (Global Average Pooling) layer.

        Args:
            inputs (tuple | torch.Tensor): The input tensor or tuple of tensors.

        Returns:
            tuple | torch.Tensor: The output tensor or tuple of tensors after applying GAP.

        Raises:
            TypeError: If the inputs are neither a tuple nor a torch.Tensor.
        """
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            _outs = self.gap(inputs)
            outs = _outs.view(inputs.size(0), -1)
        else:
            msg = "neck inputs should be tuple or torch.tensor"
            raise TypeError(msg)
        return outs
