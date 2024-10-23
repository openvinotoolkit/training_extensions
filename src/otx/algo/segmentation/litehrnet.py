# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LiteHRNet model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from torch.onnx import OperatorExportTypes

from otx.algo.segmentation.backbones import LiteHRNetBackbone
from otx.algo.segmentation.heads import FCNHead
from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore
from otx.algo.segmentation.segmentors import BaseSegmModel
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.segmentation import OTXSegmentationModel

if TYPE_CHECKING:
    from torch import nn


class LiteHRNet(OTXSegmentationModel):
    """LiteHRNet Model."""

    AVAILABLE_MODEL_VERSIONS: ClassVar[list[str]] = [
        "lite_hrnet_s",
        "lite_hrnet_18",
        "lite_hrnet_x",
    ]

    def _build_model(self) -> nn.Module:
        if self.model_version not in self.AVAILABLE_MODEL_VERSIONS:
            msg = f"Model version {self.model_version} is not supported."
            raise ValueError(msg)

        backbone = LiteHRNetBackbone(self.model_version)
        decode_head = FCNHead(self.model_version, num_classes=self.num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]
        return BaseSegmModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_lite_hrnet_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for LiteHRNet."""
        ignored_scope = self.ignore_scope
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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.scale,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={"operator_export_type": OperatorExportTypes.ONNX_ATEN_FALLBACK},
            output_names=["preds", "feature_vector"] if self.explain_mode else None,
        )

    @property
    def ignore_scope(self) -> dict[str, Any]:
        """Get the ignored scope for LiteHRNet."""
        if self.model_version == "lite_hrnet_x":
            return {
                "ignored_scope": {
                    "patterns": ["__module.model.decode_head.aggregator/*"],
                    "names": [
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0/aten::add_/Add_1",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_5",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_5",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2/aten::add_/Add_1",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_5",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_5",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0/aten::add_/Add_1",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0/aten::add_/Add_2",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_6",
                        "__module.model.backbone.stage2.0/aten::add_/Add_7",
                        "__module.model.backbone.stage2.0/aten::add_/Add_11",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1/aten::add_/Add_1",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add_2",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_6",
                        "__module.model.backbone.stage2.1/aten::add_/Add_7",
                        "__module.model.backbone.stage2.1/aten::add_/Add_11",
                        "__module.model.decode_head.aggregator/aten::add/Add",
                        "__module.model.decode_head.aggregator/aten::add/Add_1",
                        "__module.model.decode_head.aggregator/aten::add/Add_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add",
                    ],
                },
                "preset": "performance",
            }

        if self.model_version == "lite_hrnet_18":
            return {
                "ignored_scope": {
                    "patterns": ["__module.model.backbone/*"],
                    "names": [
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0/aten::add_/Add_1",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_5",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_5",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2/aten::add_/Add_1",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_5",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_5",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0/aten::add_/Add_1",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0/aten::add_/Add_2",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_6",
                        "__module.model.backbone.stage2.0/aten::add_/Add_7",
                        "__module.model.backbone.stage2.0/aten::add_/Add_11",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1/aten::add_/Add_1",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add_2",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_6",
                        "__module.model.backbone.stage2.1/aten::add_/Add_7",
                        "__module.model.backbone.stage2.1/aten::add_/Add_11",
                        "__module.model.decode_head.aggregator/aten::add/Add",
                        "__module.model.decode_head.aggregator/aten::add/Add_1",
                        "__module.model.decode_head.aggregator/aten::add/Add_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add",
                    ],
                },
                "preset": "mixed",
            }

        if self.model_version == "lite_hrnet_s":
            return {
                "ignored_scope": {
                    "names": [
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0/aten::add_/Add_1",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1/aten::add_/Add_1",
                        "__module.model.backbone.stage0.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.2/aten::add_/Add_1",
                        "__module.model.backbone.stage0.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_5",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_5",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2/aten::add_/Add_1",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_5",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_5",
                        "__module.model.decode_head.aggregator/aten::add/Add",
                        "__module.model.decode_head.aggregator/aten::add/Add_1",
                    ],
                },
                "preset": "mixed",
            }

        return {}
