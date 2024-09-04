# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Neck implementation of YOLOv7 and YOLOv9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from otx.algo.detection.backbones.gelan import Pool, RepNCSPELAN
from otx.algo.detection.utils.yolov7_v9_utils import set_info_into_module
from otx.algo.modules import Conv2dModule


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        neck_channels (int | None): The number of neck channels. Defaults to None.
    """

    def __init__(self, in_channels: int, out_channels: int, neck_channels: int | None = None) -> None:
        super().__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv2dModule(
            in_channels,
            neck_channels,
            kernel_size=1,
            normalization=nn.BatchNorm2d(neck_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )
        self.pools = nn.ModuleList([Pool("max", 5, stride=1) for _ in range(3)])
        self.conv5 = Conv2dModule(
            4 * neck_channels,
            out_channels,
            kernel_size=1,
            normalization=nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2),
            activation=nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class Concat(nn.Module):
    """Concat module.

    Args:
        dim (int): The dimension to concatenate. Defaults to 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return torch.cat(x, self.dim)


class YOLONeckModule(nn.Module):
    """Neck module for YOLOv7 and v9.

    Args:
        elan_channels (list[dict[str, int]]): The ELAN channels.
        concat_sources (list[list[str, int]]): The sources to concatenate.
        csp_args (dict[str, Any] | None): The arguments for CSP. Defaults to None.
    """

    def __init__(
        self,
        elan_channels: list[dict[str, int]],
        concat_sources: list[list[str, int]],
        csp_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.elan_channels = elan_channels
        self.concat_sources = concat_sources
        self.csp_args = csp_args or {}

        self.module = nn.ModuleList()
        for idx, elan_channel in enumerate(elan_channels):
            layer = SPPELAN if elan_channel["type"] == "SPPELAN" else RepNCSPELAN
            _csp_args = {"csp_args": self.csp_args} if elan_channel["type"] == "RepNCSPELAN" else {}
            self.module.append(
                set_info_into_module(
                    {
                        "module": layer(**elan_channel["args"], **_csp_args),
                        "tags": elan_channel["tags"],
                    },
                ),
            )
            if len(concat_sources) > idx:
                self.module.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.module.append(set_info_into_module({"module": Concat(), "source": concat_sources[idx]}))

    def forward(self, x: Tensor | dict[str, Tensor], *args, **kwargs) -> dict[str, Tensor]:
        """Forward function."""
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


class YOLONeck:
    """YOLONeck factory for detection."""

    YOLONECK_CFG: ClassVar[dict[str, Any]] = {
        "yolov9-s": {
            "elan_channels": [
                {"type": "SPPELAN", "args": {"in_channels": 256, "out_channels": 256}, "tags": "N3"},
                {
                    "type": "RepNCSPELAN",
                    "args": {"in_channels": 448, "out_channels": 192, "part_channels": 192},
                    "tags": "N4",
                },
            ],
            "concat_sources": [[-1, "B4"]],
            "csp_args": {"repeat_num": 3},
        },
        "yolov9-m": {
            "elan_channels": [
                {"type": "SPPELAN", "args": {"in_channels": 480, "out_channels": 480}, "tags": "N3"},
                {
                    "type": "RepNCSPELAN",
                    "args": {"in_channels": 840, "out_channels": 360, "part_channels": 360},
                    "tags": "N4",
                },
            ],
            "concat_sources": [[-1, "B4"], [-1, "B3"]],
        },
        "yolov9-c": {
            "elan_channels": [
                {"type": "SPPELAN", "args": {"in_channels": 512, "out_channels": 512}, "tags": "N3"},
                {
                    "type": "RepNCSPELAN",
                    "args": {"in_channels": 1024, "out_channels": 512, "part_channels": 512},
                    "tags": "N4",
                },
            ],
            "concat_sources": [[-1, "B4"], [-1, "B3"]],
        },
    }

    def __new__(cls, model_name: str) -> YOLONeckModule:
        """Constructor for YOLONeck for v7 and v9."""
        if model_name not in cls.YOLONECK_CFG:
            msg = f"model type '{model_name}' is not supported"
            raise KeyError(msg)

        return YOLONeckModule(**cls.YOLONECK_CFG[model_name])
