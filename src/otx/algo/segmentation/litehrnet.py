# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LiteHRNet model implementations."""

from __future__ import annotations

from typing import Any, Literal

from torch.onnx import OperatorExportTypes

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.entity.segmentation import MMSegCompatibleModel


class LiteHRNet(MMSegCompatibleModel):
    """LiteHRNet Model."""

    def __init__(self, num_classes: int, variant: Literal["18", 18, "s", "x"]) -> None:
        self.model_name = f"litehrnet_{variant}"
        config = read_mmconfig(model_name=self.model_name)
        super().__init__(num_classes=num_classes, config=config)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parent_parameters = super()._export_parameters
        parent_parameters.update(
            {
                "onnx_export_configuration": {"operator_export_type": OperatorExportTypes.ONNX_ATEN_FALLBACK},
                "via_onnx": True,
            },
        )

        return parent_parameters

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_lite_hrnet_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for LiteHRNet."""
        # TODO(Kirill): check PTQ without adding the whole backbone to ignored_scope #noqa: TD003
        ignored_scope = self._obtain_ignored_scope()
        optim_config = {
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {"statistics_type": "QUANTILE", "aggregator_type": "MIN", "quantile_outlier_prob": 1e-4},
                    "max": {"statistics_type": "QUANTILE", "aggregator_type": "MAX", "quantile_outlier_prob": 1e-4},
                },
            },
        }
        optim_config.update(ignored_scope)
        return optim_config

    def _obtain_ignored_scope(self) -> dict[str, Any]:
        """Returns the ignored scope for the model based on the litehrnet version."""
        if self.model_name == "litehrnet_18":
            ignored_scope_names = [
                "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.0/Add_1",
                "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.1/Add_1",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.0/Add_1",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.0/Add_2",
                "/backbone/stage1/stage1.0/Add_5",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.1/Add_1",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.1/Add_2",
                "/backbone/stage1/stage1.1/Add_5",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.2/Add_1",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.2/Add_2",
                "/backbone/stage1/stage1.2/Add_5",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.3/Add_1",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.3/Add_2",
                "/backbone/stage1/stage1.3/Add_5",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.0/Add_1",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.0/Add_2",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.0/Add_3",
                "/backbone/stage2/stage2.0/Add_6",
                "/backbone/stage2/stage2.0/Add_7",
                "/backbone/stage2/stage2.0/Add_11",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.1/Add_1",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.1/Add_2",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.1/Add_3",
                "/backbone/stage2/stage2.1/Add_6",
                "/backbone/stage2/stage2.1/Add_7",
                "/backbone/stage2/stage2.1/Add_11",
                "/aggregator/Add",
                "/aggregator/Add_1",
                "/aggregator/Add_2",
                "/backbone/stage2/stage2.1/Add",
            ]

            return {
                "ignored_scope": {
                    "patterns": ["/backbone/*"],
                    "names": ignored_scope_names,
                },
                "preset": "mixed",
            }

        if self.model_name == "litehrnet_s":
            ignored_scope_names = [
                "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.0/Add_1",
                "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.1/Add_1",
                "/backbone/stage0/stage0.2/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.2/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.2/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.2/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.2/Add_1",
                "/backbone/stage0/stage0.3/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.3/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.3/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.3/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.3/Add_1",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.0/Add_1",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.0/Add_2",
                "/backbone/stage1/stage1.0/Add_5",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.1/Add_1",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.1/Add_2",
                "/backbone/stage1/stage1.1/Add_5",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.2/Add_1",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.2/Add_2",
                "/backbone/stage1/stage1.2/Add_5",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.3/Add_1",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.3/Add_2",
                "/backbone/stage1/stage1.3/Add_5",
                "/aggregator/Add",
                "/aggregator/Add_1",
            ]

            return {
                "ignored_scope": {
                    "names": ignored_scope_names,
                },
                "preset": "mixed",
            }

        if self.model_name == "litehrnet_x":
            ignored_scope_names = [
                "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.0/Add_1",
                "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage0/stage0.1/Add_1",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.0/Add_1",
                "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.0/Add_2",
                "/backbone/stage1/stage1.0/Add_5",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.1/Add_1",
                "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.1/Add_2",
                "/backbone/stage1/stage1.1/Add_5",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.2/Add_1",
                "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.2/Add_2",
                "/backbone/stage1/stage1.2/Add_5",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage1/stage1.3/Add_1",
                "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage1/stage1.3/Add_2",
                "/backbone/stage1/stage1.3/Add_5",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.0/Add_1",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.0/Add_2",
                "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.0/Add_3",
                "/backbone/stage2/stage2.0/Add_6",
                "/backbone/stage2/stage2.0/Add_7",
                "/backbone/stage2/stage2.0/Add_11",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.1/Add_1",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.1/Add_2",
                "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.1/Add_3",
                "/backbone/stage2/stage2.1/Add_6",
                "/backbone/stage2/stage2.1/Add_7",
                "/backbone/stage2/stage2.1/Add_11",
                "/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.2/Add_1",
                "/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.2/Add_2",
                "/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.2/Add_3",
                "/backbone/stage2/stage2.2/Add_6",
                "/backbone/stage2/stage2.2/Add_7",
                "/backbone/stage2/stage2.2/Add_11",
                "/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage2/stage2.3/Add_1",
                "/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage2/stage2.3/Add_2",
                "/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage2/stage2.3/Add_3",
                "/backbone/stage2/stage2.3/Add_6",
                "/backbone/stage2/stage2.3/Add_7",
                "/backbone/stage2/stage2.3/Add_11",
                "/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_4",
                "/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage3/stage3.0/Add_1",
                "/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage3/stage3.0/Add_2",
                "/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage3/stage3.0/Add_3",
                "/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_4",
                "/backbone/stage3/stage3.0/Add_4",
                "/backbone/stage3/stage3.0/Add_7",
                "/backbone/stage3/stage3.0/Add_8",
                "/backbone/stage3/stage3.0/Add_9",
                "/backbone/stage3/stage3.0/Add_13",
                "/backbone/stage3/stage3.0/Add_14",
                "/backbone/stage3/stage3.0/Add_19",
                "/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul",
                "/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_1",
                "/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_2",
                "/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_3",
                "/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_4",
                "/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul",
                "/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_1",
                "/backbone/stage3/stage3.1/Add_1",
                "/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_2",
                "/backbone/stage3/stage3.1/Add_2",
                "/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_3",
                "/backbone/stage3/stage3.1/Add_3",
                "/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_4",
                "/backbone/stage3/stage3.1/Add_4",
                "/backbone/stage3/stage3.1/Add_7",
                "/backbone/stage3/stage3.1/Add_8",
                "/backbone/stage3/stage3.1/Add_9",
                "/backbone/stage3/stage3.1/Add_13",
                "/backbone/stage3/stage3.1/Add_14",
                "/backbone/stage3/stage3.1/Add_19",
                "/backbone/stage0/stage0.0/Add",
                "/backbone/stage0/stage0.1/Add",
                "/backbone/stage1/stage1.0/Add",
                "/backbone/stage1/stage1.1/Add",
                "/backbone/stage1/stage1.2/Add",
                "/backbone/stage1/stage1.3/Add",
                "/backbone/stage2/stage2.0/Add",
                "/backbone/stage2/stage2.1/Add",
                "/backbone/stage2/stage2.2/Add",
                "/backbone/stage2/stage2.3/Add",
                "/backbone/stage3/stage3.0/Add",
                "/backbone/stage3/stage3.1/Add",
            ]

            return {
                "ignored_scope": {
                    "patterns": ["/aggregator/*"],
                    "names": ignored_scope_names,
                },
                "preset": "performance",
            }

        return {}
