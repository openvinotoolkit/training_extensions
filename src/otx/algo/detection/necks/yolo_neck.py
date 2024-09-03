# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Neck implementation of YOLOv7 and YOLOv9."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from otx.algo.detection.backbones.gelan import RepNCSPELAN
from otx.algo.detection.backbones.yolo_v7_v9_backbone import Conv, Pool, insert_io_info_into_module


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, neck_channels: int | None = None):
        super().__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv(in_channels, neck_channels, kernel_size=1)
        self.pools = nn.ModuleList([Pool("max", 5, stride=1) for _ in range(3)])
        self.conv5 = Conv(4 * neck_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class UpSample(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.UpSample = nn.Upsample(**kwargs)

    def forward(self, x):
        return self.UpSample(x)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class YOLONeckModule(nn.Module):
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
                insert_io_info_into_module(
                    {
                        "module": layer(**elan_channel["args"], **_csp_args),
                        "tags": elan_channel["tags"],
                    }
                )
            )
            if len(concat_sources) > idx:
                self.module.append(UpSample(scale_factor=2, mode="nearest"))
                self.module.append(insert_io_info_into_module({"module": Concat(), "source": concat_sources[idx]}))

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


class YOLONeck:
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
