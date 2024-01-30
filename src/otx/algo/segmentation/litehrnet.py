# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LiteHRNet model implementations."""

from typing import Any, Literal

from torch.onnx import OperatorExportTypes

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.segmentation import MMSegCompatibleModel


class LiteHRNet(MMSegCompatibleModel):
    """LiteHRNet Model."""

    def __init__(self, num_classes: int, variant: Literal["18", 18, "s", "x"]) -> None:
        model_name = f"litehrnet_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parent_parameters = super()._export_parameters
        parent_parameters.update({"onnx_export_configuration": OperatorExportTypes.ONNX_ATEN_FALLBACK})

        return parent_parameters

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_lite_hrnet_ckpt(state_dict, add_prefix)
