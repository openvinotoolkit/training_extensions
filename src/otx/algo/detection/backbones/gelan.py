# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""GELAN implementation for YOLOv9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from otx.algo.detection.utils.yolov7_v9_utils import set_info_into_module

# TODO (sungchul): Update docstring
# TODO (sungchul): replace `build_activation_layer` in src/otx/algo/modules/activation.py with `create_activation_function`


def auto_pad(kernel_size: _size_2_t, dilation: _size_2_t = 1, **kwargs) -> tuple[int, int]:
    """Auto Padding for the convolution blocks."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def create_activation_function(activation: str) -> nn.Module:
    """Retrieves an activation function from the PyTorch nn module based on its name, case-insensitively.

    TODO (sungchul): change to use `build_activation_layer` in src/otx/algo/modules/activation.py.
    """
    if not activation or activation.lower() in ["false", "none"]:
        return nn.Identity()

    activation_map = {
        name.lower(): obj
        for name, obj in nn.modules.activation.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, nn.Module)
    }
    if activation.lower() in activation_map:
        return activation_map[activation.lower()](inplace=True)
    msg = f"Activation function '{activation}' is not found in torch.nn"
    raise ValueError(msg)


class Conv(nn.Module):
    """A basic convolutional block that includes convolution, batch normalization, and activation.

    TODO (sungchul): replace it to `ConvModule` in src/otx/algo/modules/conv_module.py.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        activation: str = "SiLU",
        **kwargs,
    ):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
        self.act = create_activation_function(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class ELAN(nn.Module):
    """ELAN  structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: int | None = None,
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = Conv(part_channels // 2, process_channels, 3, padding=1, **kwargs)
        self.conv3 = Conv(process_channels, process_channels, 3, padding=1, **kwargs)
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class Pool(nn.Module):
    """A generic pooling block supporting 'max' and 'avg' pooling methods."""

    def __init__(self, method: str = "max", kernel_size: _size_2_t = 2, **kwargs):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        pool_classes = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}
        self.pool = pool_classes[method.lower()](kernel_size=kernel_size, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class AConv(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv = Conv(in_channels, out_channels, **mid_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        return self.conv(x)


class RepConv(nn.Module):
    """A convolutional block that combines two convolution layers (kernel and point-wise)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        *,
        activation: str = "SiLU",
        **kwargs,
    ):
        super().__init__()
        self.act = create_activation_function(activation)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, activation=False, **kwargs)
        self.conv2 = Conv(in_channels, out_channels, 1, activation=False, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv1(x) + self.conv2(x))


class Bottleneck(nn.Module):
    """A bottleneck block with optional residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int] = (3, 3),
        residual: bool = True,
        expand: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0], **kwargs)
        self.conv2 = Conv(neck_channels, out_channels, kernel_size[1], **kwargs)
        self.residual = residual

        if residual and (in_channels != out_channels):
            self.residual = False
            logger.warning(
                "Residual connection disabled: in_channels ({}) != out_channels ({})",
                in_channels,
                out_channels,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class RepNCSP(nn.Module):
    """RepNCSP block with convolutions, split, and bottleneck processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()

        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size, **kwargs)

        self.bottleneck = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.bottleneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class RepNCSPELAN(nn.Module):
    """RepNCSPELAN block combining RepNCSP blocks with ELAN structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: int | None = None,
        csp_args: dict[str, Any] = {},
        csp_neck_args: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class ADown(nn.Module):
    """Downsampling module combining average and max pooling with convolution for feature reduction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv1 = Conv(half_in_channels, half_out_channels, **mid_layer)
        self.max_pool = Pool("max", **mid_layer)
        self.conv2 = Conv(half_in_channels, half_out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x2)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)


class GELANModule(nn.Module):
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
        self.module.append(set_info_into_module({"module": Conv(3, first_dim, 3, stride=2), "source": 0}))
        self.module.append(Conv(first_dim, first_dim * 2, 3, stride=2))

        block_entry_layer = ELAN if block_entry_cfg["type"] == "ELAN" else RepNCSPELAN
        self.module.append(block_entry_layer(**block_entry_cfg["args"]))

        aconv_adown_layer = AConv if is_aconv_adown == "AConv" else ADown
        for idx, csp_channel in enumerate(csp_channels):
            prev_output_channel = csp_channels[idx - 1][1] if idx > 0 else block_entry_cfg["args"]["out_channels"]
            self.module.append(aconv_adown_layer(prev_output_channel, csp_channel[0]))
            self.module.append(
                set_info_into_module(
                    {
                        "module": RepNCSPELAN(
                            csp_channel[0], csp_channel[1], part_channels=csp_channel[2], csp_args=self.csp_args
                        ),
                        "tags": f"B{idx+3}",
                    },
                )
            )

    def forward(self, x: Tensor | dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
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
    """Generalized Efficient Layer Aggregation Network (GELAN) implementation for YOLOv9."""

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
