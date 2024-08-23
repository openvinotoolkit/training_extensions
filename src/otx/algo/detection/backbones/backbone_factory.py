# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone factory for detection."""

from torch import nn


class DetectionBackboneFactory:
    """Factory class for backbone."""

    def __new__(cls, version: str) -> nn.Module:
        """Create backbone instance."""
        if "ssd" in version and "mobilenetv2" in version:
            from otx.algo.common.backbones import build_model_including_pytorchcv

            return build_model_including_pytorchcv(
                cfg={
                    "type": "mobilenetv2_w1",
                    "out_indices": [4, 5],
                    "frozen_stages": -1,
                    "norm_eval": False,
                    "pretrained": True,
                },
            )

        if "atss" in version:
            if "mobilenetv2" in version:
                from otx.algo.common.backbones import build_model_including_pytorchcv

                return build_model_including_pytorchcv(
                    cfg={
                        "type": "mobilenetv2_w1",
                        "out_indices": [2, 3, 4, 5],
                        "frozen_stages": -1,
                        "norm_eval": False,
                        "pretrained": True,
                    },
                )

            if "resnext101" in version:
                from otx.algo.common.backbones import ResNeXt

                return ResNeXt(
                    depth=101,
                    groups=64,
                    frozen_stages=1,
                    init_cfg={"type": "Pretrained", "checkpoint": "open-mmlab://resnext101_64x4d"},
                )

        if "yolox" in version:
            from otx.algo.detection.backbones import CSPDarknet

            return CSPDarknet(version)

        if "rtmdet" in version:
            from otx.algo.common.backbones import CSPNeXt

            return CSPNeXt(version)

        if "rtdetr" in version:
            from otx.algo.detection.backbones import PResNet

            return PResNet(version)

        msg = f"Unknown backbone name: {version}"
        raise ValueError(msg)
