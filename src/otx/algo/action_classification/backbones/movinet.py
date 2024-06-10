# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Code modified from: https://github.com/Atze00/MoViNet-pytorch/blob/main/movinets/models.py."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
from torch.nn.modules.utils import _pair, _triple


class Conv2dBNActivation(nn.Sequential):
    """A base module that applies a 2D Conv-BN-Activation.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (Union[int, tuple[int, int]]): Size of the convolution kernel.
        padding (Union[int, tuple[int, int]]): Size of the padding applied to the input.
        stride (Union[int, tuple[int, int]], optional): Stride of the convolution. Default: 1.
        groups (int, optional): Number of groups in the convolution. Default: 1.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use.
            If None, identity is used. Default: None.
        activation_layer (Optional[Callable[..., nn.Module]], optional): Activation layer to use.
            If None, identity is used. Default: None.
        **kwargs (Any): Additional keyword arguments passed to nn.Conv2d.

    Attributes:
        kernel_size (tuple[int, int]): Size of the convolution kernel.
        stride (tuple[int, int]): Stride of the convolution.
        out_channels (int): Number of output channels.

    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
        **kwargs,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = OrderedDict(
            {
                "conv2d": nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    **kwargs,
                ),
                "norm": norm_layer(out_planes, eps=0.001),
                "act": activation_layer(),
            },
        )

        self.out_channels = out_planes
        super().__init__(dict_layers)


class Conv3DBNActivation(nn.Sequential):
    """A base module that applies a 3D Conv-BN-Activation.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (Union[int, tuple[int, int, int]]): Size of the convolution kernel.
        padding (Union[int, tuple[int, int, int]]): Size of the padding applied to the input.
        stride (Union[int, tuple[int, int, int]], optional): Stride of the convolution. Default: 1.
        groups (int, optional): Number of groups in the convolution. Default: 1.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use.
            If None, identity is used. Default: None.
        activation_layer (Optional[Callable[..., nn.Module]], optional): Activation layer to use.
            If None, identity is used. Default: None.
        **kwargs (Any): Additional keyword arguments passed to nn.Conv3d.

    Attributes:
        kernel_size (tuple[int, int, int]): Size of the convolution kernel.
        stride (tuple[int, int, int]): Stride of the convolution.
        out_channels (int): Number of output channels.

    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: int | tuple[int, int, int],
        padding: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
        **kwargs,
    ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride

        dict_layers = OrderedDict(
            {
                "conv3d": nn.Conv3d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    **kwargs,
                ),
                "norm": norm_layer(out_planes, eps=0.001),
                "act": activation_layer(),
            },
        )

        self.out_channels = out_planes
        super().__init__(dict_layers)


class ConvBlock3D(nn.Module):
    """A module that applies a 2+1D or 3D Conv-BN-activation sequential.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (tuple[int, int, int]): Size of the convolution kernel.
        tf_like (bool): Whether to use TensorFlow-like padding and convolution.
        conv_type (str): Type of 3D convolution to use. Must be "2plus1d" or "3d".
        padding (tuple[int, int, int], optional): Size of the padding applied to the input.
            Default: (0, 0, 0).
        stride (tuple[int, int, int], optional): Stride of the convolution. Default: (1, 1, 1).
        norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use.
            If None, identity is used. Default: None.
        activation_layer (Optional[Callable[..., nn.Module]], optional): Activation layer to use.
            If None, identity is used. Default: None.
        bias (bool, optional): Whether to use bias in the convolution. Default: False.
        **kwargs (Any): Additional keyword arguments passed to nn.Conv2d or nn.Conv3d.

    Attributes:
        conv_1 (Union[Conv2dBNActivation, Conv3DBNActivation]): Convolutional layer.
        conv_2 (Optional[Conv2dBNActivation]): Convolutional layer for 2+1D convolution.
        padding (tuple[int, int, int]): Size of the padding applied to the input.
        kernel_size (tuple[int, int, int]): Size of the convolution kernel.
        dim_pad (int): Padding along the temporal dimension.
        stride (tuple[int, int, int]): Stride of the convolution.
        conv_type (str): Type of 3D convolution used.
        tf_like (bool): Whether to use TensorFlow-like padding and convolution.

    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: tuple[int, int, int],
        tf_like: bool,
        conv_type: str,
        padding: tuple[int, int, int] = (0, 0, 0),
        stride: tuple[int, int, int] = (1, 1, 1),
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv_2 = None
        if tf_like:
            # We need odd kernel to have even padding
            # and stride == 1 to precompute padding,
            if kernel_size[0] % 2 == 0:
                raise ValueError("tf_like supports only odd" + " kernels for temporal dimension")
            padding = ((kernel_size[0] - 1) // 2, 0, 0)
            if stride[0] != 1:
                raise ValueError("illegal stride value, tf like supports" + " only stride == 1 for temporal dimension")
            if stride[1] > kernel_size[1] or stride[2] > kernel_size[2]:
                # these values are not tested so should be avoided
                raise ValueError("tf_like supports only" + "  stride <= of the kernel size")

        if conv_type not in ["2plus1d", "3d"]:
            raise ValueError("only 2plus2d or 3d are " + "allowed as 3d convolutions")

        if conv_type == "2plus1d":
            self.conv_1 = Conv2dBNActivation(
                in_planes,
                out_planes,
                kernel_size=(kernel_size[1], kernel_size[2]),
                padding=(padding[1], padding[2]),
                stride=(stride[1], stride[2]),
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                bias=bias,
                **kwargs,
            )
            if kernel_size[0] > 1:
                self.conv_2 = Conv2dBNActivation(
                    in_planes,
                    out_planes,
                    kernel_size=(kernel_size[0], 1),
                    padding=(padding[0], 0),
                    stride=(stride[0], 1),
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    bias=bias,
                    **kwargs,
                )
        elif conv_type == "3d":
            self.conv_1 = Conv3DBNActivation(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=padding,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                stride=stride,
                bias=bias,
                **kwargs,
            )
        self.padding = padding
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0] - 1
        self.stride = stride
        self.conv_type = conv_type
        self.tf_like = tf_like

    def _forward(self, x: Tensor) -> Tensor:
        shape_with_buffer = x.shape
        if self.conv_type == "2plus1d":
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.conv_1(x)
        if self.conv_type == "2plus1d":
            x = rearrange(x, "(b t) c h w -> b c t h w", t=shape_with_buffer[2])
            if self.conv_2 is not None:
                w = x.shape[-1]
                x = rearrange(x, "b c t h w -> b c t (h w)")
                x = self.conv_2(x)
                x = rearrange(x, "b c t (h w) -> b c t h w", w=w)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of ConvBlock3D."""
        if self.tf_like:
            x = same_padding(
                x,
                x.shape[-2],
                x.shape[-1],
                self.stride[-2],
                self.stride[-1],
                self.kernel_size[-2],
                self.kernel_size[-1],
            )
        return self._forward(x)


class SqueezeExcitation(nn.Module):
    """Implements the Squeeze-and-Excitation (SE) block.

    Args:
        input_channels (int): Number of input channels.
        activation_2 (nn.Module): Activation function applied after the second convolutional block.
        activation_1 (nn.Module): Activation function applied after the first convolutional block.
        conv_type (str): Convolutional block type ("2plus1d" or "3d").
        squeeze_factor (int, optional): The reduction factor for the number of channels (default: 4).
        bias (bool, optional): Whether to add a bias term to the convolutional blocks (default: True).
    """

    def __init__(
        self,
        input_channels: int,
        activation_2: nn.Module,
        activation_1: nn.Module,
        conv_type: str,
        squeeze_factor: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        se_multiplier = 1
        squeeze_channels = _make_divisible(input_channels // squeeze_factor * se_multiplier, 8)
        self.fc1 = ConvBlock3D(
            input_channels * se_multiplier,
            squeeze_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            tf_like=False,
            conv_type=conv_type,
            bias=bias,
        )
        self.activation_1 = activation_1()
        self.activation_2 = activation_2()
        self.fc2 = ConvBlock3D(
            squeeze_channels,
            input_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            tf_like=False,
            conv_type=conv_type,
            bias=bias,
        )

    def _scale(self, x: Tensor) -> Tensor:
        """Computes the scaling factor for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            torch.Tensor: Scaling factor for the input tensor of shape (batch_size, channels, 1, 1, 1).
        """
        scale = F.adaptive_avg_pool3d(x, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of SqueezeExcitation."""
        scale = self._scale(x)
        return scale * x


def _make_divisible(value: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * value:
        new_v += divisor
    return new_v


def same_padding(
    x: Tensor,
    in_height: int,
    in_width: int,
    stride_h: int,
    stride_w: int,
    filter_height: int,
    filter_width: int,
) -> Tensor:
    """Applies padding to the input tensor to ensure that the output tensor size is the same as the input tensor size.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).
        in_height (int): Height of the input tensor.
        in_width (int): Width of the input tensor.
        stride_h (int): Stride in the height dimension.
        stride_w (int): Stride in the width dimension.
        filter_height (int): Height of the filter (kernel).
        filter_width (int): Width of the filter (kernel).

    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, channels, time, height + pad_h, width + pad_w), where
        pad_h and pad_w are the heights and widths of the top, bottom, left, and right padding applied to the tensor.

    """
    if in_height % stride_h == 0:
        pad_along_height = max(filter_height - stride_h, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_h), 0)
    if in_width % stride_w == 0:
        pad_along_width = max(filter_width - stride_w, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_w), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding_pad = (pad_left, pad_right, pad_top, pad_bottom)
    return torch.nn.functional.pad(x, padding_pad)


class TFAvgPool3D(nn.Module):
    """3D average pooling layer with padding."""

    def __init__(self) -> None:
        super().__init__()
        self.avgf = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x: Tensor) -> Tensor:
        """Applies 3D average pooling with padding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            torch.Tensor: Pooled tensor of shape (batch_size, channels, time, height', width'), where
            height' and width' are the heights and widths of the pooled tensor after padding is applied.

        """
        use_padding = x.shape[-1] % 2 != 0
        padding_pad = (0, 0, 0, 0) if use_padding else (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, padding_pad)
        if use_padding:
            x = torch.nn.functional.avg_pool3d(
                x,
                (1, 3, 3),
                stride=(1, 2, 2),
                count_include_pad=False,
                padding=(0, 1, 1),
            )
        else:
            x = self.avgf(x)
            x[..., -1] = x[..., -1] * 9 / 6
            x[..., -1, :] = x[..., -1, :] * 9 / 6
        return x


class BasicBneck(nn.Module):
    """Basic bottleneck block of MoViNet network.

    Args:
        cfg (DictConfig): configuration object containing block's hyperparameters.
        tf_like (bool): A boolean indicating whether to use TensorFlow like convolution
            padding or not.
        conv_type (str): A string indicating the type of convolutional layer to use.
            Can be "2d" or "3d".
        norm_layer (Callable[..., nn.Module], optional): A callable normalization layer
            to use. Defaults to None.
        activation_layer (Callable[..., nn.Module], optional): A callable activation
            layer to use. Defaults to None.

    Attributes:
        expand (ConvBlock3D, optional): An optional expansion convolutional block.
        deep (ConvBlock3D): A convolutional block with kernel size, stride, padding,
            and groups as specified in the configuration object.
        squeeze_excitation (SqueezeExcitation): A squeeze-and-excitation block.
        project (ConvBlock3D): A projection convolutional block.
        res (nn.Sequential, optional): An optional residual convolutional block.
        alpha (nn.Parameter): A learnable parameter used in the ReZero operation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tf_like: bool,
        conv_type: str,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.res = None

        layers = []
        if cfg.expanded_channels != cfg.out_channels:
            self.expand = ConvBlock3D(
                in_planes=cfg.input_channels,
                out_planes=cfg.expanded_channels,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
                conv_type=conv_type,
                tf_like=tf_like,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        self.deep = ConvBlock3D(
            in_planes=cfg.expanded_channels,
            out_planes=cfg.expanded_channels,
            kernel_size=cfg.kernel_size,
            padding=cfg.padding,
            stride=cfg.stride,
            groups=cfg.expanded_channels,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        self.se = SqueezeExcitation(
            cfg.expanded_channels,
            activation_1=activation_layer,
            activation_2=(nn.Sigmoid if conv_type == "3d" else nn.Hardsigmoid),
            conv_type=conv_type,
        )
        self.project = ConvBlock3D(
            cfg.expanded_channels,
            cfg.out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=nn.Identity,
        )

        if not (cfg.stride == (1, 1, 1) and cfg.input_channels == cfg.out_channels):
            if cfg.stride != (1, 1, 1):
                if tf_like:
                    layers.append(TFAvgPool3D())
                else:
                    layers.append(nn.AvgPool3d((1, 3, 3), stride=cfg.stride, padding=cfg.padding_avg))
            layers.append(
                ConvBlock3D(
                    in_planes=cfg.input_channels,
                    out_planes=cfg.out_channels,
                    kernel_size=(1, 1, 1),
                    padding=(0, 0, 0),
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity,
                    conv_type=conv_type,
                    tf_like=tf_like,
                ),
            )
            self.res = nn.Sequential(*layers)
        # ReZero
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of BasicBneck."""
        residual = self.res(x) if self.res is not None else x
        if hasattr(self, "expand"):
            x = self.expand(x)
        x = self.deep(x)
        x = self.se(x)
        x = self.project(x)
        return residual + self.alpha * x


class MoViNetBackboneBase(nn.Module):
    """MoViNet class used for video classification.

    Args:
        cfg (DictConfig): configuration object containing network's hyperparameters.
        conv_type (str, optional): A string indicating the type of convolutional layer
            to use. Can be "2d" or "3d". Defaults to "3d".
        tf_like (bool, optional): A boolean indicating whether to use TensorFlow like
            convolution padding or not. Defaults to False.

    Attributes:
        conv1 (ConvBlock3D): A convolutional block for the first layer.
        blocks (nn.Sequential): A sequence of basic bottleneck blocks.
        conv7 (ConvBlock3D): A convolutional block for the final layer.

    Methods:
        avg(x: Tensor) -> Tensor: A static method that returns the adaptive average pool
            of the input tensor.
        _init_weights(module): A private method that initializes the weights of the network's
            convolutional, batch normalization, and linear layers.
        forward(x: Tensor) -> Tensor: The forward pass of the network.

    """

    def __init__(
        self,
        cfg: DictConfig,
        conv_type: str = "3d",
        tf_like: bool = False,
    ) -> None:
        super().__init__()
        tf_like = True
        blocks_dic = OrderedDict()

        norm_layer = nn.BatchNorm3d if conv_type == "3d" else nn.BatchNorm2d
        activation_layer = nn.SiLU if conv_type == "3d" else nn.Hardswish

        self.conv1 = ConvBlock3D(
            in_planes=cfg.conv1.input_channels,
            out_planes=cfg.conv1.out_channels,
            kernel_size=cfg.conv1.kernel_size,
            stride=cfg.conv1.stride,
            padding=cfg.conv1.padding,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        for i, block in enumerate(cfg.blocks):
            for j, basicblock in enumerate(block):
                blocks_dic[f"b{i}_l{j}"] = BasicBneck(
                    basicblock,
                    conv_type=conv_type,
                    tf_like=tf_like,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
        self.blocks = nn.Sequential(blocks_dic)
        self.conv7 = ConvBlock3D(
            in_planes=cfg.conv7.input_channels,
            out_planes=cfg.conv7.out_channels,
            kernel_size=cfg.conv7.kernel_size,
            stride=cfg.conv7.stride,
            padding=cfg.conv7.padding,
            conv_type=conv_type,
            tf_like=tf_like,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

    def avg(self, x: Tensor) -> Tensor:
        """Returns the adaptive average pool of the input tensor.

        Args:
            x (Tensor): A tensor to be averaged.

        Returns:
            Tensor: A tensor with the averaged values.

        """
        return F.adaptive_avg_pool3d(x, 1)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MoViNet."""
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        return self.avg(x)

    def init_weights(self) -> None:
        """Initializes the weights of network."""
        self.apply(self._init_weights)


class MoViNetBackbone(MoViNetBackboneBase):
    """MoViNet wrapper class for OTX."""

    def __init__(self, **kwargs) -> None:
        cfg = DictConfig({})
        cfg.name = "A0"
        cfg.conv1 = DictConfig({})
        MoViNetBackbone.fill_conv(cfg.conv1, 3, 8, (1, 3, 3), (1, 2, 2), (0, 1, 1))

        cfg.blocks = [
            [DictConfig({})],
            [DictConfig({}) for _ in range(3)],
            [DictConfig({}) for _ in range(3)],
            [DictConfig({}) for _ in range(4)],
            [DictConfig({}) for _ in range(4)],
        ]

        # block 2
        MoViNetBackbone.fill_se_config(cfg.blocks[0][0], 8, 8, 24, (1, 5, 5), (1, 2, 2), (0, 2, 2), (0, 1, 1))

        # block 3
        MoViNetBackbone.fill_se_config(cfg.blocks[1][0], 8, 32, 80, (3, 3, 3), (1, 2, 2), (1, 0, 0), (0, 0, 0))
        MoViNetBackbone.fill_se_config(cfg.blocks[1][1], 32, 32, 80, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[1][2], 32, 32, 80, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

        # block 4
        MoViNetBackbone.fill_se_config(cfg.blocks[2][0], 32, 56, 184, (5, 3, 3), (1, 2, 2), (2, 0, 0), (0, 0, 0))
        MoViNetBackbone.fill_se_config(cfg.blocks[2][1], 56, 56, 112, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[2][2], 56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

        # block 5
        MoViNetBackbone.fill_se_config(cfg.blocks[3][0], 56, 56, 184, (5, 3, 3), (1, 1, 1), (2, 1, 1), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[3][1], 56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[3][2], 56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[3][3], 56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1))

        # block 6
        MoViNetBackbone.fill_se_config(cfg.blocks[4][0], 56, 104, 384, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[4][1], 104, 104, 280, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[4][2], 104, 104, 280, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))
        MoViNetBackbone.fill_se_config(cfg.blocks[4][3], 104, 104, 344, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1))

        cfg.conv7 = DictConfig({})
        MoViNetBackbone.fill_conv(cfg.conv7, 104, 480, (1, 1, 1), (1, 1, 1), (0, 0, 0))

        cfg.dense9 = DictConfig({"hidden_dim": 2048})
        super().__init__(cfg)

    @staticmethod
    def fill_se_config(
        conf: DictConfig,
        input_channels: int,
        out_channels: int,
        expanded_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        padding_avg: tuple[int, int, int],
    ) -> None:
        """Set the values of a given DictConfig object to SE module.

        Args:
            conf (DictConfig): The DictConfig object to be updated.
            input_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            expanded_channels (int): The number of channels after expansion in the basic block.
            kernel_size (tuple[int]): The size of the kernel.
            stride (tuple[int]): The stride of the kernel.
            padding (tuple[int]): The padding of the kernel.
            padding_avg (tuple[int]): The padding for the average pooling operation.

        Returns:
            None.
        """
        conf.expanded_channels = expanded_channels
        conf.padding_avg = padding_avg
        MoViNetBackbone.fill_conv(
            conf,
            input_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )

    @staticmethod
    def fill_conv(
        conf: DictConfig,
        input_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
    ) -> None:
        """Set the values of a given DictConfig object to conv layer.

        Args:
            conf (DictConfig): The DictConfig object to be updated.
            input_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (tuple[int]): The size of the kernel.
            stride (tuple[int]): The stride of the kernel.
            padding (tuple[int]): The padding of the kernel.

        Returns:
            None.
        """
        conf.input_channels = input_channels
        conf.out_channels = out_channels
        conf.kernel_size = kernel_size
        conf.stride = stride
        conf.padding = padding
