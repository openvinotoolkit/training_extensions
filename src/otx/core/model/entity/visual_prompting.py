# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model entity used in OTX."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Literal
import torch

from otx.core.data.entity.base import T_OTXBatchPredEntityWithXAI
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.types.precision import OTXPrecisionType
import openvino

if TYPE_CHECKING:
    from pathlib import Path
    from otx.core.types.export import OTXExportFormatType


class OTXVisualPromptingModel(
    OTXModel[
        VisualPromptingBatchDataEntity,
        VisualPromptingBatchPredEntity,
        T_OTXBatchPredEntityWithXAI,
        T_OTXTileBatchDataEntity,
    ],
):
    """Base class for the visual prompting models used in OTX."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__(num_classes=num_classes)
        self.parameters_for_export = {
            "image_encoder": {
                "input_size": (1, 3, self.model.image_size, self.model.image_size),
                "mean": (123.675, 116.28, 103.53),
                "std": (58.395, 57.12, 57.375),
                "resize_mode": "fit_to_window",
            },
            "decoder": {
                "input_size": (1, self.model.embed_dim, self.model.image_embedding_size, self.model.image_embedding_size),
            }
        }
        
    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        model = {
            "image_encoder": self.model.image_encoder,
            "decoder": self.model,
        }
        dummy_inputs = {
            "image_encoder": {
                "images": torch.randn(1, 3, self.model.image_size, self.model.image_size, dtype=torch.float32)
            },
            "decoder": {
                "image_embeddings": torch.zeros(1, self.model.embed_dim, self.model.image_embedding_size, self.model.image_embedding_size, dtype=torch.float32),
                "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float32),
                "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float32),
                "mask_input": torch.randn(1, 1, 4 * self.model.image_embedding_size, 4 * self.model.image_embedding_size, dtype=torch.float32),
                "has_mask_input": torch.tensor([[1]], dtype=torch.float32),
                "orig_size": torch.randint(low=256, high=2048, size=(1, 2), dtype=torch.int64),
            }
        }
        output_names = {
            "image_encoder": ["image_embeddings"],
            "decoder": ["upscaled_masks", "iou_predictions", "low_res_masks"]
        }
        dynamic_axes = {
            "image_encoder": None,
            "decoder": {
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            }
        }
        
        export_paths: dict[str, str] = {}
        for module in ["image_encoder", "decoder"]:
            self._export_parameters = module
            self._exporter = module
            export_paths[module] = self._exporter.export(
                model=model[module],
                output_dir=output_dir,
                base_model_name=f"visual_prompting_{module}",
                export_format=export_format,
                precision=precision,
                export_args={
                    "args": tuple(dummy_inputs[module].values()),
                    "input_names": list(dummy_inputs[module].keys()),
                    "output_names": output_names[module],
                    "dynamic_axes": dynamic_axes[module],
                }
            )
            
        return export_paths
        
    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return self.__exporter
    
    @_exporter.setter
    def _exporter(self, module: Literal["image_encoder", "decoder"]) -> None:
        self.__exporter = OTXNativeModelExporter(via_onnx=True, **self._export_parameters)
    
    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        return self.__export_parameters
    
    @_export_parameters.setter
    def _export_parameters(self, module: Literal["image_encoder", "decoder"]) -> None:
        self.__export_parameters = super()._export_parameters
        self.__export_parameters.update(**self.parameters_for_export.get(module, {}))
        # self.__export_parameters["metadata"].update(
        #     {
        #         ("model_info", "model_type"): "segment_anything",
        #         ("model_info", "task_type"): "visual_prompting",
        #     }
        # )


class OTXZeroShotVisualPromptingModel(
    OTXModel[
        ZeroShotVisualPromptingBatchDataEntity,
        ZeroShotVisualPromptingBatchPredEntity,
        T_OTXBatchPredEntityWithXAI,
        T_OTXTileBatchDataEntity,
    ],
):
    """Base class for the zero-shot visual prompting models used in OTX."""

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__(num_classes=num_classes)

        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)

    def state_dict(
        self,
        *args,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any] | None:
        """Return state dictionary of model entity with reference features, masks, and used indices."""
        super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        if isinstance(destination, dict):
            # to save reference_info instead of reference_feats only
            destination.pop(prefix + "model.reference_info.reference_feats")
            destination.update({prefix + "model.reference_info": self.model.reference_info})
        return destination

    def load_state_dict_pre_hook(self, state_dict: dict[str, Any], prefix: str = "", *args, **kwargs) -> None:
        """Load reference info manually."""
        self.model.reference_info = state_dict.get(prefix + "model.reference_info", self.model.reference_info)
        state_dict[prefix + "model.reference_info.reference_feats"] = self.model.reference_info["reference_feats"]
