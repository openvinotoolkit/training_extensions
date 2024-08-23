# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Backbone factory for detection."""

from torch import nn


class DetectionBackboneFactory:
    """Factory class for backbone."""

    def __new__(cls, model_name: str) -> nn.Module:
        """Create backbone instance."""
        if "ssd" in model_name and "mobilenetv2" in model_name:
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

        if "atss" in model_name:
            if "mobilenetv2" in model_name:
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

            if "resnext101" in model_name:
                from otx.algo.common.backbones import ResNeXt

                return ResNeXt(
                    depth=101,
                    groups=64,
                    frozen_stages=1,
                    init_cfg={"type": "Pretrained", "checkpoint": "open-mmlab://resnext101_64x4d"},
                )

        if "yolox" in model_name:
            from otx.algo.detection.backbones import CSPDarknet

            return CSPDarknet(model_name)

        if "rtmdet" in model_name:
            from otx.algo.common.backbones import CSPNeXt

            return CSPNeXt(model_name)

        if "rtdetr" in model_name:
            from otx.algo.detection.backbones import PResNet

            return PResNet(model_name)

        msg = f"Unknown backbone name: {model_name}"
        raise ValueError(msg)
