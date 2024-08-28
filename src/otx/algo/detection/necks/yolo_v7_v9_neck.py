# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Neck implementation of YOLOv7 and YOLOv9."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.detection.backbones.yolo_v7_v9_backbone import Conv, Pool, RepNCSPELAN, build_layer, module_list_forward


class SPPELAN(nn.Module):
    """SPPELAN module comprising multiple pooling and convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, neck_channels: int | None = None):
        super(SPPELAN, self).__init__()
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
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class YOLOv9Neck(nn.Module):
    """Neck for YOLOv9.

    TODO (sungchul): change neck name
    """

    def __new__(cls, model_name: str) -> nn.ModuleList:
        nn.ModuleList.forward = module_list_forward
        if model_name == "yolov9-s":
            return nn.ModuleList(
                [
                    build_layer({"module": SPPELAN(256, 256), "tags": "N3"}),
                    build_layer({"module": UpSample(scale_factor=2, mode="nearest")}),
                    build_layer({"module": Concat(), "source": [-1, "B4"]}),
                    build_layer(
                        {"module": RepNCSPELAN(448, 192, part_channels=192, csp_args={"repeat_num": 3}), "tags": "N4"},
                    ),
                ],
            )

        # if model_name == "v9-m":
        #     return nn.ModuleList(
        #         [
        #             SPPELAN(480, 480),
        #             UpSample(scale_factor=2, mode="nearest"),
        #             Concat(),
        #             RepNCSPELAN(480, 360, part_channels=360),
        #             UpSample(scale_factor=2, mode="nearest"),
        #             Concat(),
        #         ]
        #     )

        # if model_name == "v9-c":
        #     return nn.ModuleList(
        #         [
        #             SPPELAN(512, 512),
        #             UpSample(scale_factor=2, mode="nearest"),
        #             Concat(),
        #             RepNCSPELAN(512, 512, part_channels=512),
        #             UpSample(scale_factor=2, mode="nearest"),
        #             Concat(),
        #         ]
        #     )

        msg = f"Unknown model_name: {model_name}"
        raise ValueError(msg)
