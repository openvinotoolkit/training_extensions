# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""GELAN implementation for YOLOv9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar

import torch
from torch import Tensor, nn

from otx.algo.detection.utils.yolov7_v9_utils import auto_pad, set_info_into_instance
from otx.algo.modules import Conv2dModule, build_activation_layer

logger = logging.getLogger(__name__)


class ELAN(nn.Module):
    """ELAN  structure.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        part_channels (int): The number of part channels.
        process_channels (int | None, optional): The number of process channels. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv2dModule(
            in_channels,
            part_channels,
            1,
            normalization=nn.BatchNorm2d(part_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv2 = Conv2dModule(
            part_channels // 2,
            process_channels,
            3,
            padding=1,
            normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv3 = Conv2dModule(
            process_channels,
            process_channels,
            3,
            padding=1,
            normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv4 = Conv2dModule(
            part_channels + 2 * process_channels,
            out_channels,
            1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for ELAN."""
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class Pool(nn.Module):
    """A generic pooling block supporting 'max' and 'avg' pooling methods.

    Args:
        method (str, optional): The pooling method. Defaults to "max".
        kernel_size (int | tuple[int, int], optional): The kernel size. Defaults to 2.
    """

    def __init__(self, method: str = "max", kernel_size: int | tuple[int, int] = 2, **kwargs) -> None:
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        pool_classes = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}
        self.pool = pool_classes[method.lower()](kernel_size=kernel_size, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for Pool."""
        return self.pool(x)


class AConv(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv = Conv2dModule(
            in_channels,
            out_channels,
            3,
            stride=2,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for AConv."""
        x = self.avg_pool(x)
        return self.conv(x)


class RepConv(nn.Module):
    """A convolutional block that combines two convolution layers (kernel and point-wise).

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (tuple[int, int], optional): The kernel size. Defaults to 3.
        activation (Callable[..., nn.Module], optional): The activation function. Defaults to
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = 3,
        *,
        activation: Callable[..., nn.Module] = nn.SiLU,
        **kwargs,
    ) -> None:
        super().__init__()
        self.act = build_activation_layer(activation)
        self.conv1 = Conv2dModule(
            in_channels,
            out_channels,
            kernel_size,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=None,
            **kwargs,
        )
        self.conv2 = Conv2dModule(
            in_channels,
            out_channels,
            1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=None,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RepConv."""
        return self.act(self.conv1(x) + self.conv2(x))


class Bottleneck(nn.Module):
    """A bottleneck block with optional residual connections.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (tuple[int, int], optional): The kernel size. Defaults to (3, 3).
        residual (bool, optional): Whether to use residual connections. Defaults to True.
        expand (float, optional): The expansion factor. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int] = (3, 3),
        residual: bool = True,
        expand: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0], **kwargs)
        self.conv2 = Conv2dModule(
            neck_channels,
            out_channels,
            kernel_size[1],
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.residual = residual

        if residual and (in_channels != out_channels):
            self.residual = False
            msg = f"Residual connection disabled: in_channels ({in_channels}) != out_channels ({out_channels})"
            logger.warning(msg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for Bottleneck."""
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class RepNCSP(nn.Module):
    """RepNCSP block with convolutions, split, and bottleneck processing.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int, optional): The kernel size. Defaults to 1.
        csp_expand (float, optional): The expansion factor for CSP. Defaults to 0.5.
        repeat_num (int, optional): The number of repetitions. Defaults to 1.
        neck_args (dict[str, Any] | None, optional): The configuration for the bottleneck blocks. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        neck_args = neck_args or {}
        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv2dModule(
            in_channels,
            neck_channels,
            kernel_size,
            normalization=nn.BatchNorm2d(neck_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv2 = Conv2dModule(
            in_channels,
            neck_channels,
            kernel_size,
            normalization=nn.BatchNorm2d(neck_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv3 = Conv2dModule(
            2 * neck_channels,
            out_channels,
            kernel_size,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )

        self.bottleneck = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)],
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RepNCSP."""
        x1 = self.bottleneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class RepNCSPELAN(nn.Module):
    """RepNCSPELAN block combining RepNCSP blocks with ELAN structure.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        part_channels (int): The number of part channels.
        process_channels (int | None, optional): The number of process channels. Defaults to None.
        csp_args (dict[str, Any] | None, optional): The configuration for the CSP blocks. Defaults to None.
        csp_neck_args (dict[str, Any] | None, optional): The configuration for the CSP neck blocks. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: int | None = None,
        csp_args: dict[str, Any] | None = None,
        csp_neck_args: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        csp_args = csp_args or {}
        csp_neck_args = csp_neck_args or {}
        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv2dModule(
            in_channels,
            part_channels,
            1,
            normalization=nn.BatchNorm2d(part_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv2dModule(
                process_channels,
                process_channels,
                3,
                padding=1,
                normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
                **kwargs,
            ),
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv2dModule(
                process_channels,
                process_channels,
                3,
                padding=1,
                normalization=nn.BatchNorm2d(process_channels, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
                **kwargs,
            ),
        )
        self.conv4 = Conv2dModule(
            part_channels + 2 * process_channels,
            out_channels,
            1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for RepNCSPELAN."""
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class ADown(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv1 = Conv2dModule(
            half_in_channels,
            half_out_channels,
            3,
            stride=2,
            normalization=nn.BatchNorm2d(half_out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )
        self.max_pool = Pool("max", kernel_size=3, stride=2)
        self.conv2 = Conv2dModule(
            half_in_channels,
            half_out_channels,
            kernel_size=1,
            normalization=nn.BatchNorm2d(half_out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for ADown."""
        x = self.avg_pool(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x2)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)


class GELANModule(nn.Module):
    """Generalized Efficient Layer Aggregation Network (GELAN) implementation for YOLOv9.

    Args:
        first_dim (int): The number of input channels.
        block_entry_cfg (dict[str, Any]): The configuration for the entry block.
        csp_channels (list[list[int]]): The configuration for the CSP blocks.
        csp_args (dict[str, Any], optional): The configuration for the CSP blocks. Defaults to None.
        is_aconv_adown (str, optional): The type of downsampling module. Defaults to "AConv".
    """

    def __init__(
        self,
        first_dim: int,
        block_entry_cfg: dict[str, Any],
        csp_channels: list[list[int]],
        csp_args: dict[str, Any] | None = None,
        is_aconv_adown: str = "AConv",
    ) -> None:
        super().__init__()

        self.first_dim = first_dim
        self.block_entry_cfg = block_entry_cfg
        self.csp_channels = csp_channels
        self.csp_args = csp_args or {}

        self.module = nn.ModuleList()
        self.module.append(
            set_info_into_instance(
                {
                    "module": Conv2dModule(
                        3,
                        first_dim,
                        3,
                        stride=2,
                        normalization=nn.BatchNorm2d(first_dim, eps=1e-3, momentum=3e-2),
                        activation=nn.SiLU(inplace=True),
                    ),
                    "source": 0,
                },
            ),
        )
        self.module.append(
            Conv2dModule(
                first_dim,
                first_dim * 2,
                3,
                stride=2,
                normalization=nn.BatchNorm2d(first_dim * 2, eps=1e-3, momentum=3e-2),
                activation=nn.SiLU(inplace=True),
            ),
        )

        block_entry_layer = ELAN if block_entry_cfg["type"] == "ELAN" else RepNCSPELAN
        self.module.append(block_entry_layer(**block_entry_cfg["args"]))

        aconv_adown_layer = AConv if is_aconv_adown == "AConv" else ADown
        for idx, csp_channel in enumerate(csp_channels):
            prev_output_channel = csp_channels[idx - 1][1] if idx > 0 else block_entry_cfg["args"]["out_channels"]
            self.module.append(aconv_adown_layer(prev_output_channel, csp_channel[0]))
            self.module.append(
                set_info_into_instance(
                    {
                        "module": RepNCSPELAN(
                            csp_channel[0],
                            csp_channel[1],
                            part_channels=csp_channel[2],
                            csp_args=self.csp_args,
                        ),
                        "tags": f"B{idx+3}",
                    },
                ),
            )

    def forward(self, x: Tensor | dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
        """Forward pass for GELAN."""
        outputs: dict[str, Tensor] = {0: x} if isinstance(x, Tensor) else x
        for layer in self.module:
            if hasattr(layer, "source") and isinstance(layer.source, list):
                model_input = [outputs[idx] for idx in layer.source]
            else:
                model_input = outputs[getattr(layer, "source", -1)]
            x = layer(model_input)
            outputs[-1] = x
            if hasattr(layer, "tags"):
                outputs[layer.tags] = x
        return outputs


class GELAN:
    """GELAN factory for detection."""

    GELAN_CFG: ClassVar[dict[str, Any]] = {
        "yolov9-s": {
            "first_dim": 32,
            "block_entry_cfg": {"type": "ELAN", "args": {"in_channels": 64, "out_channels": 64, "part_channels": 64}},
            "csp_channels": [[128, 128, 128], [192, 192, 192], [256, 256, 256]],
            "csp_args": {"repeat_num": 3},
        },
        "yolov9-m": {
            "first_dim": 32,
            "block_entry_cfg": {
                "type": "RepNCSPELAN",
                "args": {"in_channels": 64, "out_channels": 128, "part_channels": 128},
            },
            "csp_channels": [[240, 240, 240], [360, 360, 360], [480, 480, 480]],
        },
        "yolov9-c": {
            "first_dim": 64,
            "block_entry_cfg": {
                "type": "RepNCSPELAN",
                "args": {"in_channels": 128, "out_channels": 256, "part_channels": 128},
            },
            "csp_channels": [[256, 512, 256], [512, 512, 512], [512, 512, 512]],
            "is_aconv_adown": "ADown",
        },
    }

    def __new__(cls, model_name: str) -> GELANModule:
        """Constructor for GELAN for v7 and v9."""
        if model_name not in cls.GELAN_CFG:
            msg = f"model type '{model_name}' is not supported"
            raise KeyError(msg)

        return GELANModule(**cls.GELAN_CFG[model_name])
