# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model entity used in OTX."""

from __future__ import annotations

import logging as log
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from openvino.model_api.models import Model
from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity, T_OTXBatchPredEntityWithXAI
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    import numpy as np

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
        self._parameters_for_export: dict[str, dict[str, Any]] = {
            "image_encoder": {
                "input_size": (1, 3, self.model.image_size, self.model.image_size),
                "mean": (123.675, 116.28, 103.53),
                "std": (58.395, 57.12, 57.375),
                "resize_mode": "fit_to_window",
            },
            "decoder": {
                "input_size": (
                    1,
                    self.model.embed_dim,
                    self.model.image_embedding_size,
                    self.model.image_embedding_size,
                ),
            },
        }

    def export(  # type: ignore[override]
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> dict[str, Path]:
        """Export the model to the specified format."""
        model = {
            "image_encoder": self.model.image_encoder,
            "decoder": self.model,
        }
        dummy_inputs = {
            "image_encoder": {
                "images": torch.randn(1, 3, self.model.image_size, self.model.image_size, dtype=torch.float32),
            },
            "decoder": {
                "image_embeddings": torch.zeros(
                    1,
                    self.model.embed_dim,
                    self.model.image_embedding_size,
                    self.model.image_embedding_size,
                    dtype=torch.float32,
                ),
                "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float32),
                "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float32),
                "mask_input": torch.randn(
                    1,
                    1,
                    4 * self.model.image_embedding_size,
                    4 * self.model.image_embedding_size,
                    dtype=torch.float32,
                ),
                "has_mask_input": torch.tensor([[1]], dtype=torch.float32),
                "orig_size": torch.randint(low=256, high=2048, size=(1, 2), dtype=torch.int64),
            },
        }
        output_names = {
            "image_encoder": ["image_embeddings"],
            "decoder": ["upscaled_masks", "iou_predictions", "low_res_masks"],
        }
        dynamic_axes = {
            "image_encoder": None,
            "decoder": {
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            },
        }

        export_paths: dict[str, Path] = {}
        for module in ["image_encoder", "decoder"]:
            self._export_parameters = module  # type: ignore[assignment]
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
                },
            )

        return export_paths

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(via_onnx=True, **self._export_parameters)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        return self.__export_parameters

    @_export_parameters.setter
    def _export_parameters(self, module: Literal["image_encoder", "decoder"]) -> None:
        self.__export_parameters = super()._export_parameters
        self.__export_parameters.update(**self._parameters_for_export.get(module, {}))
        self.__export_parameters["metadata"].update(
            {
                ("model_info", "model_type"): "segment_anything",
                ("model_info", "task_type"): "visual_prompting",
            },
        )


class OVVisualPromptingModel(
    OVModel[VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity, T_OTXBatchPredEntityWithXAI],
):
    """Visual prompting model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX visual prompting model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_type: str = "Visual_Prompting",
        async_inference: bool = False,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
    ) -> None:
        if async_inference:
            log.warning(
                "Async inference is not supported for visual prompting models. Setting async_inference to False.",
            )
            async_inference = False

        basename: str = Path(model_name).name
        _model_name: dict[str, str] = {
            module: model_name.replace(basename, f"visual_prompting_{module}.xml")
            for module in ["image_encoder", "decoder"]
        }
        super().__init__(
            num_classes,
            _model_name,
            model_type,
            async_inference,
            max_num_requests,
            use_throughput_mode,
            model_api_configuration,
        )

    def _create_model(self) -> dict[str, Model]:
        """Create a OV model with help of Model API."""
        from openvino.model_api.adapters import OpenvinoAdapter, create_core, get_user_config

        self.model_name: dict[str, str]
        ov_models: dict[str, Model] = {}

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_parameters = {"decoder": {"input_layouts": "image_embeddings:NCHW"}}
        for module in ["image_encoder", "decoder"]:
            model_adapter = OpenvinoAdapter(
                core=create_core(),
                model=self.model_name.get(module),
                model_parameters=model_parameters.get(module, {}),
                max_num_requests=self.num_requests,
                plugin_config=plugin_config,
            )
            ov_models[module] = Model.create_model(model_adapter, module, configuration=self.model_api_configuration)
        return ov_models

    def forward(
        self,
        inputs: VisualPromptingBatchDataEntity,
    ) -> VisualPromptingBatchPredEntity | T_OTXBatchPredEntityWithXAI | OTXBatchLossEntity:
        """Model forward function."""
        if self.async_inference:
            log.warning(
                (
                    "Async inference is not supported for visual prompting models yet. "
                    "Running synchronous inference instead.",
                ),
            )

        images, metas, batch_prompts = self._customize_inputs(inputs)
        outputs: list[dict[str, Any]] = []
        for image, meta, prompts in zip(images, metas, batch_prompts):
            # forward image encoder
            image_embeddings = self.model["image_encoder"].infer_sync(image)

            # forward decoder
            for prompt in prompts:
                label = prompt.pop("label")
                prompt.update(**image_embeddings)

                # forward decoder to get predicted mask
                prediction = self.model["decoder"].infer_sync(prompt)
                prediction["scores"] = prediction["iou_predictions"]
                prediction["labels"] = label
                processed_prediction = self.model["decoder"].postprocess(prediction, meta)
                outputs.append(processed_prediction)

        return self._customize_outputs(outputs, inputs)

    def _customize_inputs(  # type: ignore[override]
        self,
        entity: VisualPromptingBatchDataEntity,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """Customize OTX input batch data entity."""
        images: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        prompts: list[list[dict[str, Any]]] = []
        for image, bbox, point, label, imgs_info in zip(
            entity.images,
            entity.bboxes,
            entity.points,
            entity.labels,
            entity.imgs_info,
        ):
            # preprocess image encoder inputs
            numpy_image = image.cpu().numpy().transpose(1, 2, 0)
            processed_image, meta = self.model["image_encoder"].preprocess(numpy_image)
            images.append(processed_image)
            metas.append(meta)

            # preprocess decoder inputs
            processed_prompts = self.model["decoder"].preprocess(
                {
                    "bboxes": bbox.cpu().numpy() if bbox is not None else bbox,
                    "points": point.cpu().numpy() if point is not None else point,
                    "labels": label.cpu().numpy(),
                    "orig_size": imgs_info.ori_shape,
                },
            )
            prompts.append(processed_prompts)

        return images, metas, prompts

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: VisualPromptingBatchDataEntity,
    ) -> VisualPromptingBatchPredEntity | T_OTXBatchPredEntityWithXAI | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for model."""
        masks: list[tv_tensors.Mask] = []
        scores: list[torch.Tensor] = []
        labels: list[torch.LongTensor] = []
        for output in outputs:
            masks.append(torch.as_tensor(output["hard_prediction"]))
            scores.append(torch.as_tensor(output["scores"]))
            labels.append(torch.tensor([output["labels"]]))

        return VisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[torch.cat(scores, dim=0)],
            masks=[tv_tensors.Mask(torch.cat(masks, dim=0))],
            polygons=[],
            points=[],
            bboxes=[],
            labels=[torch.cat(labels)],
        )


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
