# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""GELAN implementation for YOLOv9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from torch import Tensor, nn

from otx.algo.detection.layers import ELAN, AConv, ADown, RepNCSPELAN
from otx.algo.detection.utils.utils import auto_pad, set_info_into_instance
from otx.algo.modules import Conv2dModule

logger = logging.getLogger(__name__)


class GELANModule(nn.Module):
    """Generalized Efficient Layer Aggregation Network (GELAN) implementation for YOLOv9.

    Args:
        first_dim (int): The number of input channels.
        block_entry_cfg (dict[str, Any]): The configuration for the entry block.
        csp_channels (list[list[int]]): The configuration for the CSP blocks.
        csp_args (dict[str, Any] | None, optional): The configuration for the CSP blocks. Defaults to None.
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
                        padding=auto_pad(kernel_size=3),
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
                padding=auto_pad(kernel_size=3),
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

    def forward(self, x: Tensor, *args, **kwargs) -> dict[int | str, Tensor]:
        """Forward pass for GELAN."""
        outputs: dict[int | str, Tensor] = {0: x}
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
        "yolov9_s": {
            "first_dim": 32,
            "block_entry_cfg": {"type": "ELAN", "args": {"in_channels": 64, "out_channels": 64, "part_channels": 64}},
            "csp_channels": [[128, 128, 128], [192, 192, 192], [256, 256, 256]],
            "csp_args": {"repeat_num": 3},
        },
        "yolov9_m": {
            "first_dim": 32,
            "block_entry_cfg": {
                "type": "RepNCSPELAN",
                "args": {"in_channels": 64, "out_channels": 128, "part_channels": 128},
            },
            "csp_channels": [[240, 240, 240], [360, 360, 360], [480, 480, 480]],
        },
        "yolov9_c": {
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
