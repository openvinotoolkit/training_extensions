
from future import __annotations__

from typing import Dict

import torch
import torch.nn as nn
from otx.algo.segmentation.modules import resize
from abc import ABCMeta, abstractmethod


class BaseSegmHead(nn.Module, metaclass=ABCMeta):
    """Base class for segmentation heads."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_classes: int,
        dropout_ratio: float = 0.1,
        conv_cfg: Dict[str, str] | None = None,
        norm_cfg: Dict[str, str] | None = None,
        act_cfg: Dict[str, str] = dict(type='ReLU'),
        in_index: int = -1,
        input_transform: str | None = None,
        ignore_index: int = 255,
        align_corners: bool = False,
    ) -> None:
        """Initialize the BaseSegmHead.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of channels in the feature map.
            num_classes (int): Number of classes for segmentation.
            dropout_ratio (float, optional): The dropout ratio. Defaults to 0.1.
            conv_cfg (Optional[ConfigType], optional): Config for convolution layer.
                Defaults to None.
            norm_cfg (Optional[ConfigType], optional): Config for normalization layer.
                Defaults to None.
            act_cfg (Dict[str, Union[str, Dict]], optional): Activation config.
                Defaults to dict(type='ReLU').
            in_index (int, optional): Input index. Defaults to -1.
            input_transform (Optional[str], optional): Input transform type.
                Defaults to None.
            ignore_index (int, optional): The index to be ignored. Defaults to 255.
            align_corners (bool, optional): Whether to align corners. Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.input_transform = input_transform
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(
            channels, self.num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
