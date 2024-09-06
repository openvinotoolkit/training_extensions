# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Neck implementation of YOLOv7 and YOLOv9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from torch import Tensor, nn

from otx.algo.detection.layers import SPPELAN, Concat, RepNCSPELAN
from otx.algo.detection.utils.utils import set_info_into_instance


class YOLONeckModule(nn.Module):
    """Neck module for YOLOv7 and v9.

    Args:
        elan_channels (list[dict[str, int]]): The ELAN channels.
        concat_sources (list[list[str | int]]): The sources to concatenate.
        csp_args (dict[str, Any] | None): The arguments for CSP. Defaults to None.
    """

    def __init__(
        self,
        elan_channels: list[dict[str, int]],
        concat_sources: list[list[str | int]],
        csp_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.elan_channels = elan_channels
        self.concat_sources = concat_sources
        self.csp_args = csp_args or {}

        self.module = nn.ModuleList()
        for idx, elan_channel in enumerate(elan_channels):
            layer = SPPELAN if elan_channel["type"] == "SPPELAN" else RepNCSPELAN
            _csp_args: Mapping = {"csp_args": self.csp_args} if elan_channel["type"] == "RepNCSPELAN" else {}
            self.module.append(
                set_info_into_instance(
                    {
                        "module": layer(**elan_channel["args"], **_csp_args),  # type: ignore[arg-type]
                        "tags": elan_channel["tags"],
                    },
                ),
            )
            if len(concat_sources) > idx:
                self.module.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.module.append(set_info_into_instance({"module": Concat(), "source": concat_sources[idx]}))

    def forward(self, outputs: dict[int | str, Tensor], *args, **kwargs) -> dict[int | str, Tensor]:
        """Forward function."""
        for layer in self.module:
            if hasattr(layer, "source") and isinstance(layer.source, list):
                model_input = [outputs[idx] for idx in layer.source]
            else:
                model_input = outputs[getattr(layer, "source", -1)]  # type: ignore[arg-type]
            x = layer(model_input)
            outputs[-1] = x
            if hasattr(layer, "tags"):
                outputs[layer.tags] = x
        return outputs


class YOLONeck:
    """YOLONeck factory for detection."""

    YOLONECK_CFG: ClassVar[dict[str, Any]] = {
        "yolov9_s": {
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
        "yolov9_m": {
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
        "yolov9_c": {
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
