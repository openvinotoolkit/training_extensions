# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting models entity used in OTX."""

from __future__ import annotations

import logging as log
import pickle
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import tv_tensors

from otx.core.data.entity.base import Points
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.visual_prompting import OTXVisualPromptingModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.visual_prompting import VisualPromptingMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfo, LabelInfoTypes, NullLabelInfo
from otx.core.utils.mask_util import polygon_to_bitmap

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models import Model
    from torchmetrics import MetricCollection

    from otx.core.data.module import OTXDataModule
    from otx.core.metrics import MetricCallable


def _convert_pred_entity_to_compute_metric(
    preds: VisualPromptingBatchPredEntity | ZeroShotVisualPromptingBatchPredEntity,
    inputs: VisualPromptingBatchDataEntity | ZeroShotVisualPromptingBatchDataEntity,
) -> MetricInput:
    """Convert the prediction entity to the format required by the compute metric function."""
    pred_info = []
    target_info = []

    for masks, scores, labels in zip(
        preds.masks,
        preds.scores,
        preds.labels,
    ):
        pred_info.append(
            {
                "masks": masks.data,
                "scores": scores,
                "labels": labels,
            },
        )

    for imgs_info, masks, polygons, labels in zip(
        inputs.imgs_info,
        inputs.masks,
        inputs.polygons,
        inputs.labels,
    ):
        bit_masks = masks if len(masks) else polygon_to_bitmap(polygons, *imgs_info.ori_shape)
        target_info.append(
            {
                "masks": tv_tensors.Mask(bit_masks, dtype=torch.bool).data,
                "labels": torch.cat(list(labels.values())) if isinstance(labels, dict) else labels,
            },
        )

    return {"preds": pred_info, "target": target_info}


def _inference_step(
    model: OTXVisualPromptingModel | OVVisualPromptingModel,
    metric: MetricCollection,
    inputs: VisualPromptingBatchDataEntity,
) -> None:
    """Perform a single inference step on a batch of data from the inference set."""
    preds = model.forward(inputs)

    if not isinstance(preds, VisualPromptingBatchPredEntity):
        raise TypeError(preds)

    converted_entities: dict[str, list[dict[str, Tensor]]] = _convert_pred_entity_to_compute_metric(preds, inputs)  # type: ignore[assignment]

    for _name, _metric in metric.items():
        if _name == "mAP":
            # MeanAveragePrecision
            _preds = [
                {k: v > 0.5 if k == "masks" else v.squeeze(1) if k == "scores" else v for k, v in ett.items()}
                for ett in converted_entities["preds"]
            ]
            _target = converted_entities["target"]
            _metric.update(preds=_preds, target=_target)
        elif _name in ["iou", "f1-score", "dice"]:
            # BinaryJaccardIndex, BinaryF1Score, Dice
            for cvt_preds, cvt_target in zip(converted_entities["preds"], converted_entities["target"]):
                _metric.update(cvt_preds["masks"], cvt_target["masks"])


def _inference_step_for_zero_shot(
    model: OTXZeroShotVisualPromptingModel | OVZeroShotVisualPromptingModel,
    metric: MetricCollection,
    inputs: ZeroShotVisualPromptingBatchDataEntity,
) -> None:
    """Perform a single inference step on a batch of data from the inference set."""
    preds = model.forward(inputs)

    if not isinstance(preds, ZeroShotVisualPromptingBatchPredEntity):
        raise TypeError(preds)

    converted_entities: dict[str, list[dict[str, Tensor]]] = _convert_pred_entity_to_compute_metric(preds, inputs)  # type: ignore[assignment]

    for _name, _metric in metric.items():
        if _name == "mAP":
            # MeanAveragePrecision
            _preds = [
                {
                    k: v > 0.5 if k == "masks" else v.squeeze(1).to(model.device) if k == "labels" else v
                    for k, v in ett.items()
                }
                for ett in converted_entities["preds"]
            ]
            _target = converted_entities["target"]

            # match #_preds and #_target
            if len(_preds) > len(_target):
                # interpolate _target
                num_diff = len(_preds) - len(_target)
                for idx in range(num_diff):
                    _target.append(_target[idx])
            elif len(_preds) < len(_target):
                num_diff = len(_target) - len(_preds)
                pad_prediction = {
                    "masks": torch.zeros_like(_target[0]["masks"], dtype=_target[0]["masks"].dtype),
                    "labels": torch.zeros_like(_target[0]["labels"], dtype=_target[0]["labels"].dtype),
                    "scores": torch.zeros(len(_target[0]["labels"]), dtype=torch.float32),
                }  # for empty prediction
                for idx in range(num_diff):
                    _preds.append(_preds[idx] if idx < len(_preds) else pad_prediction)

            _metric.update(preds=_preds, target=_target)
        elif _name in ["iou", "f1-score", "dice"]:
            # BinaryJaccardIndex, BinaryF1Score, Dice
            for cvt_preds, cvt_target in zip(converted_entities["preds"], converted_entities["target"]):
                _metric.update(
                    cvt_preds["masks"].sum(dim=0).clamp(0, 1),
                    cvt_target["masks"].sum(dim=0).clamp(0, 1),
                )


class OTXVisualPromptingModel(OTXModel[VisualPromptingBatchDataEntity, VisualPromptingBatchPredEntity]):
    """Base class for the visual prompting models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes = NullLabelInfo(),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        msg = f"Given label_info={label_info} has no effect."
        log.debug(msg)
        super().__init__(
            label_info=NullLabelInfo(),
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXVisualPromptingModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, self.model.image_size, self.model.image_size),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="fit_to_window",
            via_onnx=True,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Visual_Prompting",
            task_type="visual_prompting",
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for visual prompting models."""
        return {
            "model_type": "transformer",
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MIN",
                        "quantile_outlier_prob": "1e-4",
                    },
                    "max": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MAX",
                        "quantile_outlier_prob": "1e-4",
                    },
                },
            },
        }

    def _reset_prediction_layer(self, num_classes: int) -> None:
        return

    def validation_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            inputs (VisualPromptingBatchDataEntity): The input data for the validation step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type VisualPromptingBatchPredEntity.

        Returns:
            None
        """
        _inference_step(model=self, metric=self.metric, inputs=inputs)

    def test_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            inputs (VisualPromptingBatchDataEntity): The input data for the test step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type VisualPromptingBatchPredEntity.
        """
        _inference_step(model=self, metric=self.metric, inputs=inputs)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: VisualPromptingBatchPredEntity,
        inputs: VisualPromptingBatchDataEntity,
    ) -> MetricInput:
        """Convert the prediction entity to the format required by the compute metric function."""
        return _convert_pred_entity_to_compute_metric(preds=preds, inputs=inputs)

    def _set_label_info(self, _: LabelInfoTypes) -> None:
        msg = f"Reconfiguring label_info has no effect on {self.__class__.__name__}."
        log.warning(msg)


class OTXZeroShotVisualPromptingModel(
    OTXModel[ZeroShotVisualPromptingBatchDataEntity, ZeroShotVisualPromptingBatchPredEntity],
):
    """Base class for the zero-shot visual prompting models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes = NullLabelInfo(),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        msg = f"Given label_info={label_info} has no effect."
        log.debug(msg)
        super().__init__(
            label_info=NullLabelInfo(),
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXVisualPromptingModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, self.model.image_size, self.model.image_size),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="fit_to_window",
            via_onnx=True,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Visual_Prompting",
            task_type="visual_prompting",
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for zero-shot visual prompting models."""
        return {
            "model_type": "transformer",
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MIN",
                        "quantile_outlier_prob": "1e-4",
                    },
                    "max": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MAX",
                        "quantile_outlier_prob": "1e-4",
                    },
                },
            },
        }

    def on_train_start(self) -> None:
        """Initialize reference infos before learn."""
        self.initialize_reference_info()

    def on_test_start(self) -> None:
        """Load previously saved reference info."""
        super().on_test_start()
        if not self.load_reference_info(self.trainer.default_root_dir, self.device):
            log.warning("No reference info found. `Learn` will be automatically executed first.")
            self.trainer.lightning_module.automatic_optimization = False
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(self.trainer.default_root_dir, self.device)

    def on_predict_start(self) -> None:
        """Load previously saved reference info."""
        if not self.load_reference_info(self.trainer.default_root_dir, self.device):
            log.warning("No reference info found. `Learn` will be automatically executed first.")
            self.trainer.lightning_module.automatic_optimization = False
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(self.trainer.default_root_dir, self.device)

    def on_train_epoch_start(self) -> None:
        """Skip on_train_epoch_start unused in zero-shot visual prompting."""

    def on_train_epoch_end(self) -> None:
        """Skip on_train_epoch_end unused in zero-shot visual prompting."""
        if self.save_outputs:
            self.save_reference_info(self.trainer.default_root_dir)

    def on_validation_epoch_start(self) -> None:
        """Skip on_validation_epoch_start unused in zero-shot visual prompting."""

    def on_validation_epoch_end(self) -> None:
        """Skip on_validation_epoch_end unused in zero-shot visual prompting."""

    def configure_optimizers(self) -> None:  # type: ignore[override]
        """Skip configure_optimizers unused in zero-shot visual prompting."""

    def training_step(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
        batch_idx: int,
    ) -> Tensor:
        """Skip training_step unused in zero-shot visual prompting."""
        self.forward(inputs)

    def validation_step(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Skip validation_step unused in zero-shot visual prompting."""

    def test_step(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            inputs (ZeroShotVisualPromptingBatchDataEntity): The input data for the test step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type ZeroShotVisualPromptingBatchDataEntity.
        """
        _inference_step_for_zero_shot(model=self, metric=self.metric, inputs=inputs)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: ZeroShotVisualPromptingBatchPredEntity,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
    ) -> MetricInput:
        """Convert the prediction entity to the format required by the compute metric function."""
        return _convert_pred_entity_to_compute_metric(preds=preds, inputs=inputs)

    def _set_label_info(self, _: LabelInfoTypes) -> None:
        msg = f"Reconfiguring label_info has no effect on {self.__class__.__name__}."
        log.warning(msg)


class OVVisualPromptingModel(
    OVModel[
        VisualPromptingBatchDataEntity,
        VisualPromptingBatchPredEntity,
    ],
):
    """Visual prompting model compatible for OpenVINO IR inference.

    It can only consume OpenVINO IR model path and create the OTX visual prompting model compatible
        for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Visual_Prompting",
        async_inference: bool = False,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = VisualPromptingMetricCallable,
        **kwargs,
    ) -> None:
        if async_inference:
            log.warning(
                "Async inference is not supported for visual prompting models. Setting async_inference to False.",
            )
            async_inference = False

        basename: str = Path(model_name).name
        model_type_name: str = "_".join(basename.split("_")[:2])
        self.model_names: dict[str, str] = {
            module: model_name.replace(basename, f"{model_type_name}_{module}.xml")
            for module in ["image_encoder", "decoder"]
        }
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _create_model(self) -> dict[str, Model]:
        """Create a OV model with help of Model API."""
        from model_api.adapters import OpenvinoAdapter, create_core, get_user_config
        from model_api.models import Model

        ov_models: dict[str, Model] = {}

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_parameters = {"decoder": {"input_layouts": "image_embeddings:NCHW"}}
        for module in ["image_encoder", "decoder"]:
            model_adapter = OpenvinoAdapter(
                core=create_core(),
                model=self.model_names.get(module),
                model_parameters=model_parameters.get(module, {}),
                max_num_requests=self.num_requests,
                plugin_config=plugin_config,
            )
            ov_models[module] = Model.create_model(model_adapter, module, configuration=self.model_api_configuration)
        return ov_models

    def forward(
        self,
        inputs: VisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> VisualPromptingBatchPredEntity:
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
                    "labels": {k: v.cpu().numpy() for k, v in label.items()},
                    "orig_size": imgs_info.ori_shape,
                },
            )
            prompts.append(processed_prompts)

        return images, metas, prompts

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: VisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> VisualPromptingBatchPredEntity:
        """Customize OTX output batch data entity if needed for model."""
        masks: list[tv_tensors.Mask] = []
        scores: list[torch.Tensor] = []
        for output in outputs:
            masks.append(torch.as_tensor(output["hard_prediction"]))
            scores.append(torch.as_tensor(output["scores"]))

        return VisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[torch.cat(scores, dim=0)],
            masks=[tv_tensors.Mask(torch.cat(masks, dim=0))],
            polygons=[],
            points=[],
            bboxes=[],
            labels=[torch.cat(list(labels.values())) for labels in inputs.labels],
        )

    def optimize(  # type: ignore[override]
        self,
        output_dir: Path,
        data_module: OTXDataModule,
        ptq_config: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Runs NNCF quantization."""
        import nncf
        import openvino

        def check_if_quantized(model: openvino.Model) -> bool:
            """Checks if OpenVINO model is already quantized."""
            nodes = model.get_ops()
            return any(op.get_type_name() == "FakeQuantize" for op in nodes)

        def transform_fn(
            data_batch: VisualPromptingBatchDataEntity,
            module: Literal["image_encoder", "decoder"],
        ) -> np.ndarray | dict[str, Any]:
            images, _, prompts = self._customize_inputs(data_batch)  # type: ignore[arg-type]

            image = images[0]["images"]  # use only the first image
            if module == "image_encoder":
                # resize
                resized_image = self.model["image_encoder"].resize(
                    image[0],
                    (self.model["image_encoder"].w, self.model["image_encoder"].h),
                )

                # pad image if necessary because `fit_to_window` resize for python in modelapi doesn't support pad
                pad_w = max(0, self.model["image_encoder"].w - resized_image.shape[1])
                pad_h = max(0, self.model["image_encoder"].h - resized_image.shape[0])
                resized_image = np.pad(
                    resized_image,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                # normalization
                resized_image = self.model["image_encoder"].input_transform(resized_image)

                # change layout from HWC to NCHW
                return self.model["image_encoder"]._change_layout(resized_image)  # noqa: SLF001

            # obtain image embeddings from image encoder
            image_embeddings = self.model["image_encoder"].infer_sync(image)
            # use only the first prompt
            prompt_for_optim = next(iter(prompts[0].values()))[0] if isinstance(prompts[0], dict) else prompts[0][0]  # type: ignore[attr-defined]
            prompt_for_optim.pop("label")
            prompt_for_optim.update(**image_embeddings)
            return prompt_for_optim

        output_model_paths: dict[str, Path] = {}
        for module in ["image_encoder", "decoder"]:
            output_model_path = output_dir / (self._OPTIMIZED_MODEL_BASE_NAME + f"_{module}.xml")

            ov_model = openvino.Core().read_model(self.model_names[module])
            if check_if_quantized(ov_model):
                msg = "Model is already optimized by PTQ"
                raise RuntimeError(msg)

            train_dataset = data_module.train_dataloader()

            ptq_config_from_ir = self._read_ptq_config_from_ir(ov_model)
            if ptq_config is not None:
                ptq_config_from_ir.update(ptq_config)
                ptq_config = ptq_config_from_ir
            else:
                ptq_config = ptq_config_from_ir

            quantization_dataset = nncf.Dataset(train_dataset, partial(transform_fn, module=module))  # type: ignore[attr-defined]

            compressed_model = nncf.quantize(  # type: ignore[attr-defined]
                ov_model,
                quantization_dataset,
                **ptq_config,
            )

            openvino.save_model(compressed_model, output_model_path)
            output_model_paths[module] = output_model_path

        return output_model_paths

    def validation_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            inputs (VisualPromptingBatchDataEntity): The input data for the validation step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type VisualPromptingBatchPredEntity.

        Returns:
            None
        """
        _inference_step(model=self, metric=self.metric, inputs=inputs)

    def test_step(self, inputs: VisualPromptingBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            inputs (VisualPromptingBatchDataEntity): The input data for the test step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type VisualPromptingBatchPredEntity.
        """
        _inference_step(model=self, metric=self.metric, inputs=inputs)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: VisualPromptingBatchPredEntity,
        inputs: VisualPromptingBatchDataEntity,
    ) -> MetricInput:
        """Convert the prediction entity to the format required by the compute metric function."""
        return _convert_pred_entity_to_compute_metric(preds=preds, inputs=inputs)

    def _create_label_info_from_ov_ir(self) -> LabelInfo:
        """Create NullLabelInfo since Visual Prompting tasks has no use of label information."""
        return NullLabelInfo()

    def _set_label_info(self, _: LabelInfoTypes) -> None:
        msg = f"Reconfiguring label_info has no effect on {self.__class__.__name__}."
        log.warning(msg)


class OVZeroShotVisualPromptingModel(
    OVModel[
        ZeroShotVisualPromptingBatchDataEntity,
        ZeroShotVisualPromptingBatchPredEntity,
    ],
):
    """Zero-shot visual prompting model compatible for OpenVINO IR inference.

    It can only consume OpenVINO IR model path and create the OTX zero-shot visual prompting model compatible
        for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Zero_Shot_Visual_Prompting",
        async_inference: bool = False,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = VisualPromptingMetricCallable,
        reference_info_dir: Path | str = "reference_infos",
        infer_reference_info_root: Path | str = "../.latest/train",
        save_outputs: bool = True,
        **kwargs,
    ) -> None:
        if async_inference:
            log.warning(
                (
                    "Async inference is not supported for zero-shot visual prompting models. "
                    "Setting async_inference to False.",
                ),
            )
            async_inference = False

        basename: str = Path(model_name).name
        model_type_name: str = "_".join(basename.split("_")[:2])
        self.model_names: dict[str, str] = {
            module: model_name.replace(basename, f"{model_type_name}_{module}.xml")
            for module in ["image_encoder", "decoder"]
        }
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )
        self.reference_info_dir: Path = Path(reference_info_dir)
        self.infer_reference_info_root: Path = Path(infer_reference_info_root)
        self.save_outputs: bool = save_outputs

        self.point_labels_box = np.array([[2, 3]], dtype=np.float32)
        self.has_mask_inputs = [np.array([[0.0]]), np.array([[1.0]])]

        self.initialize_reference_info()

    def _create_model(self) -> dict[str, Model]:
        """Create a OV model with help of Model API."""
        from openvino.model_api.adapters import OpenvinoAdapter, create_core, get_user_config
        from openvino.model_api.models import Model

        ov_models: dict[str, Model] = {}

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_parameters = {"decoder": {"input_layouts": "image_embeddings:NCHW"}}
        for module in ["image_encoder", "decoder"]:
            model_adapter = OpenvinoAdapter(
                core=create_core(),
                model=self.model_names.get(module),
                model_parameters=model_parameters.get(module, {}),
                max_num_requests=self.num_requests,
                plugin_config=plugin_config,
            )
            ov_models[module] = Model.create_model(model_adapter, module, configuration=self.model_api_configuration)
        return ov_models

    def learn(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reset_feat: bool = False,
        default_threshold_reference: float = 0.3,
        is_cascade: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        """`Learn` for reference features."""
        if reset_feat or self.reference_feats is None:
            self.initialize_reference_info()

        images, metas, processed_prompts = self._customize_inputs(inputs)
        largest_label: int = max(sum([[int(p) for p in prompt] for prompt in processed_prompts], []))
        self.expand_reference_info(largest_label)

        reference_masks: list[np.ndarray] = []
        for image, meta, prompts in zip(images, metas, processed_prompts):
            original_shape = np.array(meta["original_shape"][:2])

            # forward image encoder
            image_embeddings = self.model["image_encoder"].infer_sync(image)
            processed_embedding = image_embeddings["image_embeddings"].squeeze().transpose(1, 2, 0)

            # get reference masks
            ref_masks: np.ndarray = np.zeros((largest_label + 1, *original_shape), dtype=np.uint8)
            for label, input_prompts in prompts.items():
                ref_mask: np.ndarray = np.zeros(original_shape, dtype=np.uint8)
                for inputs_decoder in input_prompts:
                    label = inputs_decoder.pop("label")  # noqa: PLW2901
                    if "point_coords" in inputs_decoder:
                        # bboxes and points
                        inputs_decoder.update(image_embeddings)
                        prediction = self._predict_masks(inputs_decoder, original_shape, is_cascade=is_cascade)
                        masks = prediction["upscaled_masks"]
                    else:
                        log.warning("annotation and polygon will be supported.")
                        continue
                    ref_mask[masks] += 1
                ref_mask = np.clip(ref_mask, 0, 1)

                ref_feat: np.ndarray | None = None
                cur_default_threshold_reference = deepcopy(default_threshold_reference)
                while ref_feat is None:
                    log.info(f"[*] default_threshold_reference : {cur_default_threshold_reference:.4f}")
                    ref_feat = self._generate_masked_features(
                        feats=processed_embedding,
                        masks=ref_mask,
                        threshold_mask=cur_default_threshold_reference,
                        image_size=self.model["image_encoder"].image_size,
                    )
                    cur_default_threshold_reference -= 0.05

                self.reference_feats[label] = ref_feat
                self.used_indices: np.ndarray = np.concatenate((self.used_indices, label))
                ref_masks[label] = ref_mask
            reference_masks.append(ref_masks)
        self.used_indices = np.unique(self.used_indices)
        return {"reference_feats": self.reference_feats, "used_indices": self.used_indices}, reference_masks

    def infer(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: np.ndarray,
        used_indices: np.ndarray,
        is_cascade: bool = True,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        default_threshold_target: float = 0.65,
        image_size: int = 1024,
        downsizing: int = 64,
    ) -> list[list[defaultdict[int, list]]]:
        """`Infer` for target predictions."""
        images, metas, _ = self._customize_inputs(inputs)
        total_results: list[list[defaultdict[int, list]]] = []
        for image, meta in zip(images, metas):
            original_shape = np.array(meta["original_shape"][:2])

            # forward image encoder
            image_embeddings = self.model["image_encoder"].infer_sync(image)

            # get point candidates
            total_points_scores, total_bg_coords = self._get_prompt_candidates(
                image_embeddings=image_embeddings["image_embeddings"],
                reference_feats=reference_feats,
                used_indices=used_indices,
                original_shape=original_shape,
                threshold=threshold,
                num_bg_points=num_bg_points,
                default_threshold_target=default_threshold_target,
                image_size=image_size,
                downsizing=downsizing,
            )

            predicted_masks: defaultdict[int, list] = defaultdict(list)
            used_points: defaultdict[int, list] = defaultdict(list)
            for label in total_points_scores:
                points_scores = total_points_scores[label]
                bg_coords = total_bg_coords[label]
                for points_score in points_scores:
                    if points_score[-1] in [-1.0, 0.0]:
                        continue

                    x, y = points_score[:2]
                    is_done = False
                    for pm in predicted_masks.get(label, []):
                        # check if that point is already assigned
                        if pm[int(y), int(x)] > 0:
                            is_done = True
                            break
                    if is_done:
                        continue

                    point_coords = np.concatenate((np.array([[x, y]]), bg_coords), axis=0, dtype=np.float32)
                    point_coords = self.model["decoder"].apply_coords(point_coords, original_shape)
                    point_labels = np.array([1] + [0] * len(bg_coords), dtype=np.float32)
                    inputs_decoder = {
                        "point_coords": point_coords[None],
                        "point_labels": point_labels[None],
                        "orig_size": original_shape[None],
                    }
                    inputs_decoder.update(image_embeddings)

                    prediction = self._predict_masks(inputs_decoder, original_shape, is_cascade)
                    prediction.update({"scores": points_score[-1]})

                    predicted_masks[label].append(prediction[self.model["decoder"].output_blob_name])
                    used_points[label].append(points_score)

            # check overlapping area between different label masks
            self._inspect_overlapping_areas(predicted_masks, used_points)
            total_results.append([predicted_masks, used_points])
        return total_results

    def forward(  # type: ignore[override]
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity:
        """Model forward function."""
        kwargs: dict[str, Any] = {}
        fn = self.learn if self.training else self.infer
        if not self.training:
            kwargs.update(
                {
                    "reference_feats": self.reference_feats,
                    "used_indices": self.used_indices,
                },
            )

        if self.async_inference:
            log.warning(
                (
                    "Async inference is not supported for zero-shot visual prompting models yet. "
                    "Running synchronous inference instead.",
                ),
            )

        return self._customize_outputs(fn(inputs, **kwargs), inputs)  # type: ignore[operator]

    def _customize_inputs(  # type: ignore[override]
        self,
        entity: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[int, list[Any]]]]:
        """Customize OTX input batch data entity."""
        images: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        processed_prompts: list[list[dict[str, Any]]] = []
        for image, prompts, labels, imgs_info in zip(
            entity.images,
            entity.prompts,
            entity.labels,
            entity.imgs_info,
        ):
            # preprocess image encoder inputs
            numpy_image = image.cpu().numpy().transpose(1, 2, 0)
            processed_image, meta = self.model["image_encoder"].preprocess(numpy_image)
            images.append(processed_image)
            metas.append(meta)
            if self.training:
                points: list[np.ndarray] = []
                bboxes: list[np.ndarray] = []
                _labels: dict[str, list[int]] = defaultdict(list)
                for prompt, label in zip(prompts, labels):  # type: ignore[arg-type]
                    if isinstance(prompt, tv_tensors.BoundingBoxes):
                        bboxes.append(prompt.cpu().numpy())
                        _labels["bboxes"].append(label.cpu().numpy())
                    elif isinstance(prompt, Points):
                        points.append(prompt.cpu().numpy())
                        _labels["points"].append(label.cpu().numpy())

                # preprocess decoder inputs
                processed_prompts.append(
                    self.model["decoder"].preprocess(
                        {
                            "bboxes": bboxes,
                            "points": points,
                            "labels": _labels,
                            "orig_size": imgs_info.ori_shape,
                        },
                    ),
                )
        processed_prompts_w_labels = self._gather_prompts_with_labels(processed_prompts)
        return images, metas, processed_prompts_w_labels

    def _customize_outputs(  # type: ignore[override]
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity:
        """Customize OTX output batch data entity if needed for model."""
        if self.training:
            return outputs

        masks: list[tv_tensors.Mask] = []
        prompts: list[Points] = []
        scores: list[torch.Tensor] = []
        labels: list[torch.LongTensor] = []
        for output in outputs:
            predicted_masks, used_points = output
            for label, predicted_mask in predicted_masks.items():
                if len(predicted_mask) == 0:
                    continue
                masks.append(
                    tv_tensors.Mask(
                        torch.stack([torch.as_tensor(m) for m in predicted_mask], dim=0),
                        dtype=torch.float32,
                    ),
                )
                prompts.append(
                    Points(
                        torch.stack([torch.as_tensor(p[:2]) for p in used_points[label]], dim=0),
                        canvas_size=inputs.imgs_info[0].ori_shape,
                        dtype=torch.float32,
                    ),
                )
                scores.append(torch.stack([torch.as_tensor(p[2]) for p in used_points[label]], dim=0))
                labels.append(torch.stack([torch.LongTensor([label]) for _ in range(len(scores[-1]))], dim=0))

        return ZeroShotVisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            prompts=prompts,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    def optimize(  # type: ignore[override]
        self,
        output_dir: Path,
        data_module: OTXDataModule,
        ptq_config: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Runs NNCF quantization."""
        import nncf
        import openvino

        def check_if_quantized(model: openvino.Model) -> bool:
            """Checks if OpenVINO model is already quantized."""
            nodes = model.get_ops()
            return any(op.get_type_name() == "FakeQuantize" for op in nodes)

        def transform_fn(
            data_batch: ZeroShotVisualPromptingBatchDataEntity,
            module: Literal["image_encoder", "decoder"],
        ) -> np.ndarray | dict[str, Any]:
            images, _, prompts = self._customize_inputs(data_batch)  # type: ignore[arg-type]

            image = images[0]["images"]  # use only the first image
            if module == "image_encoder":
                # resize
                resized_image = self.model["image_encoder"].resize(
                    image[0],
                    (self.model["image_encoder"].w, self.model["image_encoder"].h),
                )

                # pad image if necessary because `fit_to_window` resize for python in modelapi doesn't support pad
                pad_w = max(0, self.model["image_encoder"].w - resized_image.shape[1])
                pad_h = max(0, self.model["image_encoder"].h - resized_image.shape[0])
                resized_image = np.pad(
                    resized_image,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                # normalization
                resized_image = self.model["image_encoder"].input_transform(resized_image)

                # change layout from HWC to NCHW
                return self.model["image_encoder"]._change_layout(resized_image)  # noqa: SLF001

            # obtain image embeddings from image encoder
            image_embeddings = self.model["image_encoder"].infer_sync(image)
            # use only the first prompt
            prompt_for_optim = next(iter(prompts[0].values()))[0] if isinstance(prompts[0], dict) else prompts[0][0]  # type: ignore[attr-defined]
            prompt_for_optim.pop("label")
            prompt_for_optim.update(**image_embeddings)
            return prompt_for_optim

        output_model_paths: dict[str, Path] = {}
        for module in ["image_encoder", "decoder"]:
            output_model_path = output_dir / (self._OPTIMIZED_MODEL_BASE_NAME + f"_{module}.xml")

            ov_model = openvino.Core().read_model(self.model_names[module])
            if check_if_quantized(ov_model):
                msg = "Model is already optimized by PTQ"
                raise RuntimeError(msg)

            train_dataset = data_module.train_dataloader()

            ptq_config_from_ir = self._read_ptq_config_from_ir(ov_model)
            if ptq_config is not None:
                ptq_config_from_ir.update(ptq_config)
                ptq_config = ptq_config_from_ir
            else:
                ptq_config = ptq_config_from_ir

            quantization_dataset = nncf.Dataset(train_dataset, partial(transform_fn, module=module))  # type: ignore[attr-defined]

            compressed_model = nncf.quantize(  # type: ignore[attr-defined]
                ov_model,
                quantization_dataset,
                **ptq_config,
            )

            openvino.save_model(compressed_model, output_model_path)
            output_model_paths[module] = output_model_path

        return output_model_paths

    ######################################
    #             Preprocess             #
    ######################################
    def _gather_prompts_with_labels(
        self,
        batch_prompts: list[list[dict[str, Any]]],
    ) -> list[dict[int, list[np.ndarray]]]:
        """Gather prompts according to labels."""
        total_processed_prompts: list[dict[int, list[np.ndarray]]] = []
        for prompts in batch_prompts:
            processed_prompts: defaultdict[int, list[np.ndarray]] = defaultdict(list)
            for prompt in prompts:
                processed_prompts[int(prompt["label"])].append(prompt)
            total_processed_prompts.append(dict(sorted(processed_prompts.items(), key=lambda x: x)))
        return total_processed_prompts

    ######################################
    #               Common               #
    ######################################
    def _predict_masks(
        self,
        inputs: dict[str, np.ndarray],
        original_size: np.ndarray,
        is_cascade: bool = False,
    ) -> dict[str, np.ndarray]:
        """Process function of OpenVINO Visual Prompting Inferencer."""
        masks: np.ndarray
        logits: np.ndarray
        scores: np.ndarray
        num_iter = 3 if is_cascade else 1
        for i in range(num_iter):
            if i == 0:
                # First-step prediction
                mask_input = np.zeros(
                    (1, 1, *(x * 4 for x in inputs["image_embeddings"].shape[2:])),
                    dtype=np.float32,
                )
                has_mask_input = self.has_mask_inputs[0]

            elif i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks = self._decide_masks(masks, logits, scores, is_single=True)  # noqa: F821
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self.has_mask_inputs[1]

            elif i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks = self._decide_masks(masks, logits, scores)  # noqa: F821
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self.has_mask_inputs[1]
                y, x = np.nonzero(masks)
                box_coords = self.model["decoder"].apply_coords(
                    np.array([[x.min(), y.min()], [x.max(), y.max()]], dtype=np.float32),
                    original_size,
                )
                box_coords = np.expand_dims(box_coords, axis=0)
                inputs.update(
                    {
                        "point_coords": np.concatenate((inputs["point_coords"], box_coords), axis=1),
                        "point_labels": np.concatenate((inputs["point_labels"], self.point_labels_box), axis=1),
                    },
                )

            inputs.update({"mask_input": mask_input, "has_mask_input": has_mask_input})
            prediction = self.model["decoder"].infer_sync(inputs)
            upscaled_masks, scores, logits = (
                prediction["upscaled_masks"],
                prediction["iou_predictions"],
                prediction["low_res_masks"],
            )
            masks = upscaled_masks > self.model["decoder"].mask_threshold

        _, masks = self._decide_masks(masks, logits, scores)
        return {"upscaled_masks": masks}

    def _decide_masks(
        self,
        masks: np.ndarray,
        logits: np.ndarray,
        scores: np.ndarray,
        is_single: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Post-process logits for resized masks according to best index based on scores."""
        if is_single:
            best_idx = 0
        else:
            # skip the first index components
            scores, masks, logits = (x[:, 1:] for x in (scores, masks, logits))

            # filter zero masks
            while len(scores[0]) > 0 and masks[0, (best_idx := np.argmax(scores[0]))].sum() == 0:
                scores, masks, logits = (
                    np.concatenate((x[:, :best_idx], x[:, best_idx + 1 :]), axis=1) for x in (scores, masks, logits)
                )

            if len(scores[0]) == 0:
                # all predicted masks were zero masks, ignore them.
                return None, np.zeros(masks.shape[-2:])

            best_idx = np.argmax(scores[0])
        return logits[:, [best_idx]], masks[0, best_idx]

    ######################################
    #               Learn                #
    ######################################
    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.reference_feats = np.zeros((0, 1, self.model["decoder"].embed_dim), dtype=np.float32)
        self.used_indices = np.array([], dtype=np.int64)

    def expand_reference_info(self, new_largest_label: int) -> None:
        """Expand reference info dimensions if newly given processed prompts have more lables."""
        if new_largest_label > (cur_largest_label := len(self.reference_feats) - 1):
            diff = new_largest_label - cur_largest_label
            self.reference_feats = np.pad(self.reference_feats, ((0, diff), (0, 0), (0, 0)), constant_values=0.0)

    def save_reference_info(self, default_root_dir: Path | str) -> None:
        """Save reference info."""
        reference_info = {
            "reference_feats": self.reference_feats,
            "used_indices": self.used_indices,
        }
        # save reference info
        path_reference_info: Path = Path(default_root_dir) / self.reference_info_dir / "reference_info.pt"
        path_reference_info.parent.mkdir(parents=True, exist_ok=True)
        # TODO (sungchul): ticket no. 139210
        torch.save({k: torch.as_tensor(v) for k, v in reference_info.items()}, path_reference_info)
        pickle.dump(reference_info, path_reference_info.with_suffix(".pickle").open("wb"))
        log.info(f"Saved reference info at {path_reference_info}.")

    def _generate_masked_features(
        self,
        feats: np.ndarray,
        masks: np.ndarray,
        threshold_mask: float,
        image_size: int = 1024,
    ) -> tuple[np.ndarray, ...] | None:
        """Generate masked features.

        Args:
            feats (np.ndarray): Raw reference features. It will be filtered with masks.
            masks (np.ndarray): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.
            image_size (int): Input image size.

        Returns:
            (np.ndarray): Masked features.
        """
        target_shape = image_size / max(masks.shape) * np.array(masks.shape)
        target_shape = target_shape[::-1].astype(np.int32)

        # Post-process masks
        masks = cv2.resize(masks, target_shape, interpolation=cv2.INTER_LINEAR)
        masks = self._pad_to_square(masks, image_size)
        masks = cv2.resize(masks, feats.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0)[None]
        return masked_feat / np.linalg.norm(masked_feat, axis=-1, keepdims=True)

    def _pad_to_square(self, x: np.ndarray, image_size: int = 1024) -> np.ndarray:
        """Pad to a square input.

        Args:
            x (np.ndarray): Mask to be padded.

        Returns:
            (np.ndarray): Padded mask.
        """
        h, w = x.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        return np.pad(x, ((0, padh), (0, padw)), constant_values=0.0)

    ######################################
    #               Infer                #
    ######################################
    def load_reference_info(self, default_root_dir: Path | str, *args, **kwargs) -> bool:
        """Load latest reference info to be used."""
        _infer_reference_info_root: Path = (
            self.infer_reference_info_root
            if self.infer_reference_info_root == self.infer_reference_info_root.absolute()
            else Path(default_root_dir) / self.infer_reference_info_root
        )

        if (
            path_reference_info := _infer_reference_info_root / self.reference_info_dir / "reference_info.pickle"
        ).is_file():
            reference_info: dict[str, np.ndarray] = pickle.load(path_reference_info.open("rb"))  # noqa: S301
            self.reference_feats = reference_info.get(
                "reference_feats",
                np.zeros((0, 1, self.model["decoder"].embed_dim), dtype=np.float32),
            )
            self.used_indices = reference_info.get("used_indices", np.array([], dtype=np.int64))
            log.info(f"reference info saved at {path_reference_info} was successfully loaded.")
            return True
        return False

    def _get_prompt_candidates(
        self,
        image_embeddings: np.ndarray,
        reference_feats: np.ndarray,
        used_indices: np.ndarray,
        original_shape: np.ndarray,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        default_threshold_target: float = 0.65,
        image_size: int = 1024,
        downsizing: int = 64,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Get prompt candidates."""
        target_feat = image_embeddings.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / np.linalg.norm(target_feat, axis=0, keepdims=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        total_points_scores: dict[int, np.ndarray] = {}
        total_bg_coords: dict[int, np.ndarray] = {}
        for label in used_indices:
            sim = reference_feats[label] @ target_feat
            sim = sim.reshape(h_feat, w_feat)
            sim = self._resize_to_original_shape(sim, image_size, original_shape)

            threshold = (threshold == 0) * default_threshold_target + threshold
            points_scores, bg_coords = self._point_selection(
                mask_sim=sim,
                original_shape=original_shape,
                threshold=threshold,
                num_bg_points=num_bg_points,
                image_size=image_size,
                downsizing=downsizing,
            )

            if points_scores is not None:
                total_points_scores[label] = points_scores
                total_bg_coords[label] = bg_coords
        return total_points_scores, total_bg_coords

    def _point_selection(
        self,
        mask_sim: np.ndarray,
        original_shape: np.ndarray,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        image_size: int = 1024,
        downsizing: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-first point selection
        point_coords = np.where(mask_sim > threshold)
        fg_coords_scores = np.stack(point_coords[::-1] + (mask_sim[point_coords],), axis=0).T

        ## skip if there is no point coords
        if len(fg_coords_scores) == 0:
            return None, None

        ratio = image_size / original_shape.max()
        width = (original_shape[1] * ratio).astype(np.int64)
        n_w = width // downsizing

        ## get grid numbers
        idx_grid = fg_coords_scores[:, 1] * ratio // downsizing * n_w + fg_coords_scores[:, 0] * ratio // downsizing
        idx_grid_unique = np.unique(idx_grid.astype(np.int64))

        ## get matched indices
        matched_matrix = np.expand_dims(idx_grid, axis=-1) == idx_grid_unique  # (totalN, uniqueN)

        ## sample fg_coords_scores matched by matched_matrix
        matched_grid = np.expand_dims(fg_coords_scores, axis=1) * np.expand_dims(matched_matrix, axis=-1)

        ## sample the highest score one of the samples that are in the same grid
        matched_indices = self._topk_numpy(matched_grid[..., -1], k=1, axis=0, largest=True)[1][0].astype(np.int64)
        points_scores = matched_grid[matched_indices].diagonal().T

        ## sort by the highest score
        sorted_points_scores_indices = np.flip(np.argsort(points_scores[:, -1]), axis=-1).astype(np.int64)
        points_scores = points_scores[sorted_points_scores_indices]

        # Top-last point selection
        bg_indices = self._topk_numpy(mask_sim.flatten(), num_bg_points, largest=False)[1]
        bg_x = np.expand_dims(bg_indices // w_sim, axis=0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = np.concatenate((bg_y, bg_x), axis=0).transpose(1, 0)
        bg_coords = bg_coords.astype(np.float32)

        return points_scores, bg_coords

    def _resize_to_original_shape(self, masks: np.ndarray, image_size: int, original_shape: np.ndarray) -> np.ndarray:
        """Resize feature size to original shape."""
        # resize feature size to input size
        masks = cv2.resize(masks, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        # remove pad
        prepadded_size = self._get_prepadded_size(original_shape, image_size)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

        # resize unpadded one to original shape
        original_shape = original_shape.astype(np.int64)
        h, w = original_shape[0], original_shape[1]
        return cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)

    def _get_prepadded_size(self, original_shape: int, image_size: int) -> np.ndarray:
        """Get pre-padded size."""
        scale = image_size / np.max(original_shape)
        transformed_size = scale * original_shape
        return np.floor(transformed_size + 0.5).astype(np.int64)

    def _inspect_overlapping_areas(
        self,
        predicted_masks: dict[int, list[np.ndarray]],
        used_points: dict[int, list[np.ndarray]],
        threshold_iou: float = 0.8,
    ) -> None:
        def _calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> tuple[float, np.ndarray | None]:
            assert mask1.ndim == 2  # noqa: S101
            assert mask2.ndim == 2  # noqa: S101
            # Avoid division by zero
            if (union := np.logical_or(mask1, mask2).sum().item()) == 0:
                return 0.0, None
            intersection = np.logical_and(mask1, mask2)
            return intersection.sum().item() / union, intersection

        for (label, masks), (other_label, other_masks) in product(predicted_masks.items(), predicted_masks.items()):
            if other_label <= label:
                continue

            overlapped_label = []
            overlapped_other_label = []
            for (im, mask), (jm, other_mask) in product(enumerate(masks), enumerate(other_masks)):
                _mask_iou, _intersection = _calculate_mask_iou(mask, other_mask)
                if _mask_iou > threshold_iou:
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)
                elif _mask_iou > 0:
                    # refine the slightly overlapping region
                    overlapped_coords = np.where(_intersection)
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        other_mask[overlapped_coords] = 0.0
                    else:
                        mask[overlapped_coords] = 0.0

            for im in sorted(set(overlapped_label), reverse=True):
                masks.pop(im)
                used_points[label].pop(im)

            for jm in sorted(set(overlapped_other_label), reverse=True):
                other_masks.pop(jm)
                used_points[other_label].pop(jm)

    def _topk_numpy(self, x: np.ndarray, k: int, axis: int = -1, largest: bool = True) -> np.ndarray:
        """Top-k function for numpy same with torch.topk."""
        if largest:
            k = -k
            indices = range(k, 0)
        else:
            indices = range(k)
        partitioned_ind = np.argpartition(x, k, axis=axis).take(indices=indices, axis=axis)
        partitioned_scores = np.take_along_axis(x, partitioned_ind, axis=axis)
        sorted_trunc_ind = np.argsort(partitioned_scores, axis=axis)
        if largest:
            sorted_trunc_ind = np.flip(sorted_trunc_ind, axis=axis)
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
        return scores, ind

    def _reset_prediction_layer(self, num_classes: int) -> None:
        return

    ######################################
    #            Lit Module              #
    ######################################
    def on_train_start(self) -> None:
        """Initialize reference infos before learn."""
        self.initialize_reference_info()

    def on_test_start(self) -> None:
        """Load previously saved reference info."""
        super().on_test_start()
        if not self.load_reference_info(self.trainer.default_root_dir, self.device):
            log.warning("No reference info found. `Learn` will be automatically executed first.")
            self.trainer.lightning_module.automatic_optimization = False
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(self.trainer.default_root_dir, self.device)

    def on_predict_start(self) -> None:
        """Load previously saved reference info."""
        if not self.load_reference_info(self.trainer.default_root_dir, self.device):
            log.warning("No reference info found. `Learn` will be automatically executed first.")
            self.trainer.lightning_module.automatic_optimization = False
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(self.trainer.default_root_dir, self.device)

    def on_train_epoch_start(self) -> None:
        """Skip on_train_epoch_start unused in zero-shot visual prompting."""

    def on_train_epoch_end(self) -> None:
        """Skip on_train_epoch_end unused in zero-shot visual prompting."""
        if self.save_outputs:
            self.save_reference_info(self.trainer.default_root_dir)

    def on_validation_epoch_start(self) -> None:
        """Skip on_validation_epoch_start unused in zero-shot visual prompting."""

    def on_validation_epoch_end(self) -> None:
        """Skip on_validation_epoch_end unused in zero-shot visual prompting."""

    def configure_optimizers(self) -> None:  # type: ignore[override]
        """Skip configure_optimizers unused in zero-shot visual prompting."""

    def training_step(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
        batch_idx: int,
    ) -> Tensor:
        """Skip training_step unused in zero-shot visual prompting."""
        self.forward(inputs)

    def validation_step(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Skip validation_step unused in zero-shot visual prompting."""

    def test_step(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            inputs (ZeroShotVisualPromptingBatchDataEntity): The input data for the test step.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the predictions are not of type ZeroShotVisualPromptingBatchPredEntity.
        """
        _inference_step_for_zero_shot(model=self, metric=self.metric, inputs=inputs)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: ZeroShotVisualPromptingBatchPredEntity,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
    ) -> MetricInput:
        """Convert the prediction entity to the format required by the compute metric function."""
        return _convert_pred_entity_to_compute_metric(preds=preds, inputs=inputs)

    def _create_label_info_from_ov_ir(self) -> LabelInfo:
        """Create NullLabelInfo since Visual Prompting tasks has no use of label information."""
        return NullLabelInfo()

    def _set_label_info(self, _: LabelInfoTypes) -> None:
        msg = f"Reconfiguring label_info has no effect on {self.__class__.__name__}."
        log.warning(msg)
