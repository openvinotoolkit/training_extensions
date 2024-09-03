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


class GELAN:
    """Generalized Efficient Layer Aggregation Network (GELAN) implementation for YOLOv9."""

    GELAN_CFG: ClassVar[dict[str, Any]] = {
        "yolov9-s": {
            "fitst_dim": 32,
            "rep_ncsp_elan_dim": [128, 192, 256],
            "csp_args": {"repeat_num": 3},
        },
        "yolov9_m": {
            "fitst_dim": 32,
            "rep_ncsp_elan_dim": [128, 240, 360, 480],
        },
        "yolov9_c": {
            "fitst_dim": 64,
            "rep_ncsp_elan_dim": [256, 512, 512, 512],
        },
    }
    # GELAN_CFG: ClassVar[list[dict[str, Any]]] = {
    #     "yolov9_s": [
    #         {"module": Conv(3, 32, 3, stride=2), "source": 0},
    #         {"module": Conv(32, 64, 3, stride=2)},
    #         {"module": ELAN(64, 64, part_channels=64)},
    #         {"module": AConv(64, 128)},
    #         {"module": RepNCSPELAN(128, 128, part_channels=128, csp_args={"repeat_num": 3}), "tags": "B3"},
    #         {"module": AConv(128, 192)},
    #         {"module": RepNCSPELAN(192, 192, part_channels=192, csp_args={"repeat_num": 3}), "tags": "B4"},
    #         {"module": AConv(192, 256)},
    #         {"module": RepNCSPELAN(256, 256, part_channels=256, csp_args={"repeat_num": 3}), "tags": "B5"},
    #     ],
    #     "yolov9_m": [
    #         {"module": Conv(3, 32, 3, stride=2), "source": 0},
    #         {"module": Conv(32, 64, 3, stride=2)},
    #         {"module": RepNCSPELAN(64, 128, part_channels=128)},
    #         {"module": AConv(128, 240)},
    #         {"module": RepNCSPELAN(240, 240, part_channels=240), "tags": "B3"},
    #         {"module": AConv(240, 360)},
    #         {"module": RepNCSPELAN(360, 360, part_channels=360), "tags": "B4"},
    #         {"module": AConv(360, 480)},
    #         {"module": RepNCSPELAN(480, 480, part_channels=480), "tags": "B5"},
    #     ],
    #     "yolov9_c": [
    #         {"module": Conv(3, 64, 3, stride=2), "source": 0},
    #         {"module": Conv(64, 128, 3, stride=2)},
    #         {"module": RepNCSPELAN(128, 256, part_channels=128)},
    #         {"module": ADown(256, 256)},
    #         {"module": RepNCSPELAN(256, 512, part_channels=256), "tags": "B3"},
    #         {"module": ADown(512, 512)},
    #         {"module": RepNCSPELAN(512, 512, part_channels=512), "tags": "B4"},
    #         {"module": ADown(512, 512)},
    #         {"module": RepNCSPELAN(512, 512, part_channels=512), "tags": "B5"},
    #     ],
    # }

    def __new__(self, model_name: str) -> nn.ModuleList:
        modules = nn.ModuleList()

        # first convs
        fitst_dim: int = self.GELAN_CFG[model_name]["fitst_dim"]
        modules.append(Conv(3, fitst_dim, 3, stride=2))

        return modules

        # for layer_dict in self.GELAN_CFG[model_name]:
        #     layer = layer_dict.pop("module")
        #     for k, v in layer_dict.items():
        #         setattr(layer, k, v)
