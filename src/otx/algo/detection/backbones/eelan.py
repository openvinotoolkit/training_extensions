# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""EELAN implementation for YOLOv7.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

from typing import Any, ClassVar

from torch import nn

from otx.algo.detection.layers import Concat
from otx.algo.detection.utils.utils import auto_pad, set_info_into_instance
from otx.algo.modules import Conv2dModule


class EELANModule(nn.Module):
    """Extended-Efficient Layer Aggregation Network (EELAN) implementation for YOLOv7."""

    def __init__(
        self,
        first_dim: int,
        conv_channels: list[int],
    ) -> None:
        super().__init__()

        self.module = nn.ModuleList()

        # stem
        prev_output_channel = 3
        for idx, (output_channel, stride) in enumerate(zip([first_dim, first_dim * 2, first_dim * 2], [1, 2, 1])):
            module_info = {
                "module": Conv2dModule(
                    prev_output_channel,
                    output_channel,
                    3,
                    stride=stride,
                    padding=auto_pad(kernel_size=3),
                    normalization=nn.BatchNorm2d(output_channel, eps=1e-3, momentum=3e-2),
                    activation=nn.SiLU(inplace=True),
                ),
            }
            if idx == 0:
                module_info["source"] = 0

            self.module.append(set_info_into_instance(module_info))
            prev_output_channel = output_channel

        # block
        for idx, conv_channel in enumerate(conv_channels):
            if idx == 0:
                # conv
                self.module.append(
                    Conv2dModule(
                        prev_output_channel,
                        conv_channel * 2,
                        3,
                        stride=2,
                        padding=auto_pad(kernel_size=3),
                        normalization=nn.BatchNorm2d(conv_channel * 2, eps=1e-3, momentum=3e-2),
                        activation=nn.SiLU(inplace=True),
                    )
                )

            else:
                # maxpool - conv w/ k=1 x 2 - conv w/ k=3 & s=2 - concat([-1, -3])
                self.module.append(nn.MaxPool2d(kernel_size=2, padding=0))
                self.module.append(
                    Conv2dModule(
                        prev_output_channel,
                        conv_channel,
                        1,
                        stride=1,
                        padding=auto_pad(kernel_size=1),
                        normalization=nn.BatchNorm2d(conv_channel, eps=1e-3, momentum=3e-2),
                        activation=nn.SiLU(inplace=True),
                    )
                )
                self.module.append(
                    set_info_into_instance(
                        {
                            "module": Conv2dModule(
                                conv_channel,
                                conv_channel,
                                1,
                                stride=1,
                                padding=auto_pad(kernel_size=1),
                                normalization=nn.BatchNorm2d(conv_channel, eps=1e-3, momentum=3e-2),
                                activation=nn.SiLU(inplace=True),
                            ),
                            "source": -3,
                        }
                    )
                )
                self.module.append(
                    Conv2dModule(
                        conv_channel,
                        conv_channel,
                        3,
                        stride=2,
                        padding=auto_pad(kernel_size=3),
                        normalization=nn.BatchNorm2d(conv_channel, eps=1e-3, momentum=3e-2),
                        activation=nn.SiLU(inplace=True),
                    )
                )
                self.module.append({"module": set_info_into_instance(Concat()), "source": [-1, -3]})

            # conv w/ k=1 x 2 - conv w/ k=3 x 4 - concat([-1, -3, -5, -6]) - conv w/ k=1
            self.module.append(
                Conv2dModule(
                    conv_channel,
                    conv_channel,
                    1,
                    stride=1,
                    padding=auto_pad(kernel_size=1),
                    normalization=nn.BatchNorm2d(conv_channel, eps=1e-3, momentum=3e-2),
                    activation=nn.SiLU(inplace=True),
                )
            )
            self.module.append(
                set_info_into_instance(
                    {
                        "module": Conv2dModule(
                            conv_channel,
                            conv_channel * 2,
                            1,
                            stride=1,
                            padding=auto_pad(kernel_size=1),
                            normalization=nn.BatchNorm2d(conv_channel, eps=1e-3, momentum=3e-2),
                            activation=nn.SiLU(inplace=True),
                        ),
                        "source": -2,  # concat output
                    }
                )
            )

            for _ in range(4):
                self.module.append(
                    Conv2dModule(
                        conv_channel,
                        conv_channel,
                        3,
                        stride=1,
                        padding=auto_pad(kernel_size=3),
                        normalization=nn.BatchNorm2d(output_channel[1], eps=1e-3, momentum=3e-2),
                        activation=nn.SiLU(inplace=True),
                    ),
                )

            module_info = {"module": Concat(), "source": [-1, -3, -5, -6]}
            if idx == 1:
                module_info["tags"] = "B3"
            self.module.append(set_info_into_instance(module_info))

            module_info = {
                "module": Conv2dModule(
                    conv_channel * 4,  # after concatenate 4 outputs
                    conv_channel * 4,
                    1,
                    stride=1,
                    padding=auto_pad(kernel_size=1),
                    normalization=nn.BatchNorm2d(output_channel[1], eps=1e-3, momentum=3e-2),
                    activation=nn.SiLU(inplace=True),
                )
            }
            if idx in [2, 3]:
                module_info["tags"] = f"B{idx+2}"
            self.module.append(set_info_into_instance(module_info))

            output_channel = conv_channel * 4  # after concatenate 4 outputs


class EELAN:
    """EELAN factory for detection."""

    EELAN_CFG: ClassVar[dict[str, Any]] = {
        "yolov7": {
            "first_dim": 32,
            "conv_channels": [64, 128, 256, 256],
        }
    }

    def __new__(cls, model_name: str) -> EELANModule:
        """Constructor for EELAN for YOLOv7."""
        if model_name not in cls.EELAN_CFG:
            msg = f"model type '{model_name}' is not supported"
            raise KeyError(msg)

        return EELANModule(**cls.EELAN_CFG[model_name])
