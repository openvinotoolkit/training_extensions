# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base head for getitune segmentation models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import torch
from torch import nn

from getitune.backend.lightning.models.segmentation.modules import resize


class BaseSegmentationHead(nn.Module):
    """Base class for segmentation heads.

    Args:
        in_channels (int | list[int]): Number of input channels.
        channels (int): Number of channels in the feature map.
        num_classes (int): Number of classes for segmentation.
        dropout_ratio (float, optional): The dropout ratio. Defaults to 0.1.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to ``nn.ReLU``.
        in_index (int, list[int], optional): Input index. Defaults to -1.
        input_transform (Optional[str], optional): Input transform type.
            Defaults to None.
        align_corners (bool, optional): Whether to align corners. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int | list[int],
        channels: int,
        num_classes: int,
        dropout_ratio: float = 0.1,
        normalization: Callable[..., nn.Module] | None = None,
        activation: Callable[..., nn.Module] | None = nn.ReLU,
        in_index: int | list[int] = -1,
        input_transform: str | None = None,
        align_corners: bool = False,
    ) -> None:
        """Initialize the BaseSegmHead."""
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.input_transform = input_transform
        self.dropout_ratio = dropout_ratio
        self.normalization = normalization
        self.activation = activation
        if self.input_transform is not None and not isinstance(in_index, list):
            msg = f'"in_index" expects a list, but got {type(in_index)}'
            raise TypeError(msg)
        self.in_index = in_index
        self.align_corners = align_corners

        if input_transform == "resize_concat":
            if not isinstance(in_channels, list):
                msg = f'"in_channels" expects a list, but got {type(in_channels)}'
                raise TypeError(msg)
            self.in_channels = sum(in_channels)
        else:
            self.in_channels = in_channels  # type: ignore[assignment]
        self.conv_seg = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _transform_inputs(
        self,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Transform inputs for decoder.

        Args:
            inputs (List[torch.Tensor]): List of multi-level img features.

        Returns:
            torch.Tensor: The transformed inputs
        """
        if self.input_transform == "resize_concat" and isinstance(self.in_index, list):
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input_tensor=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select" and isinstance(self.in_index, list):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        else:
            msg = f"Unsupported input_transform type: {self.input_transform}"
            raise ValueError(msg)

        return inputs

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward function for segmentation head.

        Args:
            inputs (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes, H, W).
        """

    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        """Classify each pixel.

        Args:
            feat (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The output tensor containing the classified pixel values.
        """
        if self.dropout is not None:
            feat = self.dropout(feat)
        output: torch.Tensor = self.conv_seg(feat)
        return output
