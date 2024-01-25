# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LiteHRNet model implementations."""

from typing import Any, Literal

from torch.onnx import OperatorExportTypes

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.segmentation import MMSegCompatibleModel


class LiteHRNet(MMSegCompatibleModel):
    """LiteHRNet Model."""

    def __init__(self, num_classes: int, variant: Literal["18", 18, "s", "x"]) -> None:
        model_name = f"litehrnet_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def export_params(self) -> dict[str, Any]:
        """Parameters for an  exporter."""
        export_params = super().export_params
        export_params["onnx_export_configuration"] = {
            "operator_export_type": OperatorExportTypes.ONNX_ATEN_FALLBACK,
        }

        return export_params
