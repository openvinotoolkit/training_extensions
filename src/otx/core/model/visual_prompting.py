# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for visual prompting models entity used in OTX."""

from __future__ import annotations

import logging as log
import pickle  # nosec: B403   used pickle dump and load only to share inference results
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from datumaro import Polygon as dmPolygon
from model_api.models import Model
from model_api.models.visual_prompting import (
    Prompt,
    SAMLearnableVisualPrompter,
    SAMVisualPrompter,
    VisualPromptingFeatures,
)
from torch import Tensor, nn
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity, Points
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
    ZeroShotVisualPromptingLabel,
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
    from model_api.models.utils import PredictedMask, VisualPromptingResult
    from torchmetrics import MetricCollection

    from otx.core.data.module import OTXDataModule
    from otx.core.metrics import MetricCallable


# ruff: noqa: F401


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

    # filter labels using corresponding ground truth
    inputs.labels = [
        label.masks if inputs.masks and label.masks is not None else label.polygons for label in inputs.labels
    ]

    converted_entities: dict[str, list[dict[str, Tensor]]] = _convert_pred_entity_to_compute_metric(preds, inputs)  # type: ignore[assignment]

    for _name, _metric in metric.items():
        if _name == "mAP":
            # MeanAveragePrecision
            _preds = [
                {k: v > 0.5 if k == "masks" else v.to(model.device) if k == "labels" else v for k, v in ett.items()}
                for ett in converted_entities["preds"]
            ]
            _target = converted_entities["target"]
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
        input_size: tuple[int, int] = (1024, 1024),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        msg = f"Given label_info={label_info} has no effect."
        log.debug(msg)
        super().__init__(
            label_info=NullLabelInfo(),
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.input_size: tuple[int, int]

    @abstractmethod
    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    def _create_model(self) -> nn.Module:
        return self._build_model()

    def _customize_inputs(self, inputs: VisualPromptingBatchDataEntity) -> dict[str, Any]:  # type: ignore[override]
        """Customize the inputs for the model."""
        images = tv_tensors.wrap(torch.stack(inputs.images, dim=0).to(dtype=torch.float32), like=inputs.images[0])
        return {
            "images": images,
            "ori_shapes": [torch.tensor(info.ori_shape) for info in inputs.imgs_info],
            "gt_masks": inputs.masks,
            "bboxes": self._inspect_prompts(inputs.bboxes),
            "points": [
                (
                    (tv_tensors.wrap(point.unsqueeze(1), like=point), torch.ones(len(point), 1, device=point.device))
                    if point is not None
                    else None
                )
                for point in self._inspect_prompts(inputs.points)
            ],
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: VisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> VisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for model."""
        if self.training:
            return outputs

        masks: list[tv_tensors.Mask] = []
        scores: list[torch.Tensor] = []
        for mask, score in zip(*outputs):
            masks.append(tv_tensors.Mask(mask, dtype=torch.float32))
            scores.append(score)

        return VisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            masks=masks,
            polygons=[],
            points=[],
            bboxes=[],
            labels=[torch.cat(list(labels.values())) for labels in inputs.labels],
        )

    def _inspect_prompts(self, prompts: list[tv_tensors.TVTensor]) -> list[tv_tensors.TVTensor | None]:
        """Inspect if given prompts are empty.

        If there are empty prompts (shape=0), they will be converted to None.
        """
        return [None if p is None or p.shape[0] == 0 else p for p in prompts]

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXVisualPromptingModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
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

    def get_dummy_input(self, batch_size: int = 1) -> VisualPromptingBatchDataEntity:
        """Returns a dummy input for VPT model."""
        images = [torch.rand(3, *self.input_size) for _ in range(batch_size)]
        labels = [{"points": torch.LongTensor([0] * batch_size)}] * batch_size
        prompts = [torch.zeros((1, 2))] * batch_size
        return VisualPromptingBatchDataEntity(
            batch_size,
            images,
            imgs_info=[],
            labels=labels,
            points=prompts,
            masks=[None] * batch_size,
            polygons=[[None]] * batch_size,
            bboxes=[None] * batch_size,
        )


class OTXZeroShotVisualPromptingModel(
    OTXModel[ZeroShotVisualPromptingBatchDataEntity, ZeroShotVisualPromptingBatchPredEntity],
):
    """Base class for the zero-shot visual prompting models used in OTX."""

    reference_feats: Tensor
    used_indices: Tensor

    def __init__(
        self,
        input_size: tuple[int, int],
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
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.input_size: tuple[int, int]

    @abstractmethod
    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    def _create_model(self) -> nn.Module:
        return self._build_model()

    def _customize_inputs(  # type: ignore[override]
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: Tensor | None = None,
        used_indices: Tensor | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        """Customize the inputs for the model."""
        inputs = self.transforms(inputs)
        forward_inputs = {
            "images": [tv_tensors.wrap(image.unsqueeze(0), like=image) for image in inputs.images],
            "reference_feats": reference_feats if reference_feats is not None else self.reference_feats,
            "used_indices": used_indices if used_indices is not None else self.used_indices,
            "ori_shapes": [torch.tensor(info.ori_shape) for info in inputs.imgs_info],
        }
        if self.training:
            # learn
            forward_inputs.update({"processed_prompts": self._gather_prompts_with_labels(inputs)})

        return forward_inputs

    def _customize_outputs(  # type: ignore[override]
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        if self.training:
            self.reference_feats = outputs[0].get("reference_feats")
            self.used_indices = outputs[0].get("used_indices")
            return outputs

        masks: list[tv_tensors.Mask] = []
        prompts: list[Points] = []
        scores: list[Tensor] = []
        labels: list[Tensor] = []
        for idx, (predicted_masks, used_points) in enumerate(outputs):
            _masks: list[Tensor] = []
            _prompts: list[Tensor] = []
            _scores: list[Tensor] = []
            _labels: list[Tensor] = []
            for label, predicted_mask in predicted_masks.items():
                if len(predicted_mask) == 0:
                    continue
                _masks.append(torch.stack(predicted_mask, dim=0))
                _used_points_scores = torch.stack(used_points[label], dim=0)
                _prompts.append(_used_points_scores[:, :2])
                _scores.append(_used_points_scores[:, 2])
                _labels.append(torch.tensor([label] * len(_used_points_scores), dtype=torch.int64, device=self.device))

            if len(_masks) == 0:
                masks.append(
                    tv_tensors.Mask(
                        torch.zeros((1, *inputs.imgs_info[idx].ori_shape), dtype=torch.float32, device=self.device),
                    ),
                )
                prompts.append(
                    Points([], canvas_size=inputs.imgs_info[idx].ori_shape, dtype=torch.float32, device=self.device),
                )
                scores.append(torch.tensor([-1.0], dtype=torch.float32, device=self.device))
                labels.append(torch.tensor([-1], dtype=torch.int64, device=self.device))
                continue

            masks.append(tv_tensors.Mask(torch.cat(_masks, dim=0)))
            prompts.append(Points(torch.cat(_prompts, dim=0), canvas_size=inputs.imgs_info[idx].ori_shape))
            scores.append(torch.cat(_scores, dim=0))
            labels.append(torch.cat(_labels, dim=0))

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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXVisualPromptingModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
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
            self.training = True
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(
                self.trainer.default_root_dir,
                self.device,
                path_to_directly_load=self.saved_reference_info_path,
            )

    def on_predict_start(self) -> None:
        """Load previously saved reference info."""
        if not self.load_reference_info(self.trainer.default_root_dir, self.device):
            log.warning("No reference info found. `Learn` will be automatically executed first.")
            self.trainer.lightning_module.automatic_optimization = False
            self.training = True
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(
                self.trainer.default_root_dir,
                self.device,
                path_to_directly_load=self.saved_reference_info_path,
            )

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

    def get_dummy_input(self, batch_size: int = 1) -> ZeroShotVisualPromptingBatchDataEntity:
        """Returns a dummy input for ZSL VPT model."""
        images = [torch.rand(3, *self.input_size) for _ in range(batch_size)]
        labels = [ZeroShotVisualPromptingLabel(prompts=torch.LongTensor([0]))] * batch_size
        prompts = [torch.zeros((1, 2))] * batch_size
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return ZeroShotVisualPromptingBatchDataEntity(
            batch_size,
            images,
            imgs_info=infos,
            labels=labels,
            prompts=prompts,
            masks=[],
            polygons=[],
        )


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
        use_throughput_mode: bool = False,
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

    def _create_model(self) -> SAMVisualPrompter:
        """Create a OV model with help of Model API."""
        from model_api.adapters import OpenvinoAdapter, create_core

        ov_device = "CPU"
        ie = create_core()
        if not self.force_cpu:
            devices = ie.available_devices
            for device in devices:
                device_name = ie.get_property(device_name=device, property="FULL_DEVICE_NAME")
                if "dGPU" in device_name and "Intel" in device_name:
                    ov_device = device
                    break

        plugin_config = {}
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"
        model_parameters = {"decoder": {"input_layouts": "image_embeddings:NCHW"}}

        ov_models: dict[str, Model] = {}
        for module in ["image_encoder", "decoder"]:
            model_adapter = OpenvinoAdapter(
                core=create_core(),
                device=ov_device,
                model=self.model_names.get(module),
                model_parameters=model_parameters.get(module, {}),
                max_num_requests=self.num_requests,
                plugin_config=plugin_config,
            )
            ov_models[module] = Model.create_model(
                model_adapter,
                model_type=f"sam_{module}",
                configuration=self.model_api_configuration,
            )
        return SAMVisualPrompter(ov_models["image_encoder"], ov_models["decoder"])

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

        images, batch_prompts = self._customize_inputs(inputs)
        outputs: list[VisualPromptingResult] = []
        for image, prompt in zip(images, batch_prompts):
            outputs.append(self.model(image, **prompt))

        return self._customize_outputs(outputs, inputs)

    def _customize_inputs(  # type: ignore[override]
        self,
        entity: VisualPromptingBatchDataEntity,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """Customize OTX input batch data entity."""
        images: list[np.ndarray] = []
        prompts: list[dict[str, Any]] = []

        for image, bbox, point, label in zip(
            entity.images,
            entity.bboxes,
            entity.points,
            entity.labels,
        ):
            processed_image = image.cpu().numpy().transpose(1, 2, 0)
            images.append(processed_image)

            all_labels = {k: v.cpu().numpy() for k, v in label.items()}
            boxes_prompts = []
            points_prompts = []

            if bbox is not None:
                for i, box in enumerate(bbox.cpu().numpy()):
                    boxes_prompts.append(Prompt(box, all_labels["bboxes"][i]))

            if point is not None:
                for i, p in enumerate(point.cpu().numpy()):
                    points_prompts.append(Prompt(p, all_labels["points"][i]))

            processed_prompt = {
                "boxes": boxes_prompts if boxes_prompts else None,
                "points": points_prompts if points_prompts else None,
            }

            prompts.append(processed_prompt)

        return images, prompts

    def _customize_outputs(
        self,
        outputs: list[VisualPromptingResult],
        inputs: VisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> VisualPromptingBatchPredEntity:
        """Customize OTX output batch data entity if needed for model."""
        masks: list[tv_tensors.Mask] = []
        scores: list[Tensor] = []
        labels: list[Tensor] = []
        for image_output in outputs:
            masks.append(tv_tensors.Mask(np.concatenate(image_output.hard_predictions), device=self.device))
            scores.append(torch.as_tensor(np.concatenate(image_output.scores), device=self.device))
            labels.append(torch.as_tensor(image_output.labels, device=self.device))

        return VisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            masks=masks,
            polygons=[],
            points=[],
            bboxes=[],
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
            data_batch: VisualPromptingBatchDataEntity,
            module: Literal["image_encoder", "decoder"],
        ) -> np.ndarray | dict[str, Any]:
            images: list[np.ndarray] = []
            prompts: list[list[dict[str, Any]]] = []

            for image, bbox, point, label, imgs_info in zip(
                data_batch.images,
                data_batch.bboxes,
                data_batch.points,
                data_batch.labels,
                data_batch.imgs_info,
            ):
                # preprocess image encoder inputs
                numpy_image = image.cpu().numpy().transpose(1, 2, 0)
                processed_image, meta = self.model.encoder.preprocess(numpy_image)
                images.append(processed_image)

                # preprocess decoder inputs
                processed_prompts = self.model.decoder.preprocess(
                    {
                        "bboxes": bbox.cpu().numpy() if bbox is not None else bbox,
                        "points": point.cpu().numpy() if point is not None else point,
                        "labels": {k: v.cpu().numpy() for k, v in label.items()},
                        "orig_size": imgs_info.ori_shape,
                    },
                )
                prompts.append(processed_prompts)

            image = images[0]["images"]  # use only the first image

            if module == "image_encoder":
                # resize
                resized_image = self.model.encoder.resize(
                    image[0],
                    (self.model.encoder.w, self.model.encoder.h),
                )

                # pad image if necessary because `fit_to_window` resize for python in modelapi doesn't support pad
                pad_w = max(0, self.model.encoder.w - resized_image.shape[1])
                pad_h = max(0, self.model.encoder.h - resized_image.shape[0])
                resized_image = np.pad(
                    resized_image,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                # normalization
                resized_image = self.model.encoder.input_transform(resized_image)

                # change layout from HWC to NCHW
                return self.model.encoder._change_layout(resized_image)  # noqa: SLF001

            # obtain image embeddings from image encoder
            image_embeddings = self.model.encoder.infer_sync(image)
            # use only the first prompt
            prompt_for_optim = next(iter(prompts[0].values()))[0] if isinstance(prompts[0], dict) else prompts[0][0]  # type: ignore[attr-defined]
            prompt_for_optim.pop("label")
            prompt_for_optim.update(**image_embeddings)
            return prompt_for_optim

        # ticket no. : CVS-135462
        # There is segmentation fault issue when using num_workers > 0 during releasing memory.
        # To avoid this issue, force num_workers to 0.
        data_module.train_subset.num_workers = 0

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

    def get_dummy_input(self, batch_size: int = 1) -> VisualPromptingBatchDataEntity:
        """Returns a dummy input for classification OV model."""
        # Resize is embedded to the OV model, which means we don't need to know the actual size
        images = [torch.rand(3, 224, 224) for _ in range(batch_size)]
        labels = [{"points": torch.LongTensor([0] * batch_size)}] * batch_size
        prompts = [torch.zeros((1, 2))] * batch_size
        return VisualPromptingBatchDataEntity(
            batch_size,
            images,
            imgs_info=[],
            labels=labels,
            points=prompts,
            masks=[None] * batch_size,
            polygons=[[None]] * batch_size,
            bboxes=[None] * batch_size,
        )


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
        use_throughput_mode: bool = False,
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

        self.reference_feats: np.ndarray | None = None
        self.used_indices: np.ndarray | None = None

        self.initialize_reference_info()

    def _create_model(self) -> SAMLearnableVisualPrompter:
        """Create a OV model with help of Model API."""
        from model_api.adapters import OpenvinoAdapter, create_core

        ov_device = "CPU"
        ie = create_core()
        if not self.force_cpu:
            devices = ie.available_devices
            for device in devices:
                device_name = ie.get_property(device_name=device, property="FULL_DEVICE_NAME")
                if "dGPU" in device_name and "Intel" in device_name:
                    ov_device = device
                    break

        plugin_config = {}
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"
        model_parameters = {"decoder": {"input_layouts": "image_embeddings:NCHW"}}

        ov_models: dict[str, Model] = {}
        for module in ["image_encoder", "decoder"]:
            model_adapter = OpenvinoAdapter(
                core=create_core(),
                device=ov_device,
                model=self.model_names.get(module),
                model_parameters=model_parameters.get(module, {}),
                max_num_requests=self.num_requests,
                plugin_config=plugin_config,
            )
            ov_models[module] = Model.create_model(
                model_adapter,
                model_type=f"sam_{module}",
                configuration=self.model_api_configuration,
            )

        return SAMLearnableVisualPrompter(ov_models["image_encoder"], ov_models["decoder"])

    def learn(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reset_feat: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        """`Learn` for reference features."""
        images, processed_prompts = self._customize_inputs(inputs)
        reference_masks: list[np.ndarray] = []
        reference_features = VisualPromptingFeatures(np.array(0), np.array(0))

        for image, prompts in zip(images, processed_prompts):
            reference_features, masks = self.model.learn(image, reset_features=reset_feat, **prompts)
            reference_masks.append(masks)

        self.reference_feats = reference_features.feature_vectors
        self.used_indices = reference_features.used_indices

        return {"reference_feats": self.reference_feats, "used_indices": self.used_indices}, reference_masks

    def infer(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: np.ndarray,
        used_indices: np.ndarray,
    ) -> list[dict[int, PredictedMask]]:
        """`Infer` for target predictions."""
        images, _ = self._customize_inputs(inputs)

        total_results: list[dict[int, PredictedMask]] = []
        for image in images:
            result = self.model(image, VisualPromptingFeatures(reference_feats, used_indices))
            total_results.append(result.data)

        return total_results

    def forward(  # type: ignore[override]
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]] | ZeroShotVisualPromptingBatchPredEntity:
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
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """Customize OTX input batch data entity."""
        images: list[np.ndarray] = []
        processed_prompts: list[dict[str, Any]] = []

        for image, prompts, polygons, labels in zip(
            entity.images,
            entity.prompts,
            entity.polygons,
            entity.labels,
        ):
            # preprocess image encoder inputs
            numpy_image = image.cpu().numpy().transpose(1, 2, 0)
            images.append(numpy_image)

            if self.training:
                _bboxes: list[Prompt] = []
                _points: list[Prompt] = []
                _polygons: list[Prompt] = []
                for prompt, label in zip(prompts, labels.prompts):  # type: ignore[arg-type]
                    if isinstance(prompt, tv_tensors.BoundingBoxes):
                        _bboxes.append(Prompt(prompt.cpu().numpy(), label.cpu().numpy()))
                    elif isinstance(prompt, Points):
                        _points.append(Prompt(prompt.cpu().numpy(), label.cpu().numpy()))

                if polygons and labels.polygons is not None:
                    for polygon, label in zip(polygons, labels.polygons):
                        _polygons.append(Prompt(np.array(polygon.points, dtype=np.int32), label.cpu().numpy()))

                # TODO (sungchul, sovrasov): support mask?

                # preprocess decoder inputs
                processed_prompts.append(
                    {
                        "boxes": _bboxes,
                        "points": _points,
                        "polygons": _polygons,
                    },
                )

        return images, processed_prompts

    def _customize_outputs(  # type: ignore[override]
        self,
        outputs: tuple[dict[str, np.ndarray], list[np.ndarray]] | list[dict[int, PredictedMask]],
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]] | ZeroShotVisualPromptingBatchPredEntity:
        """Customize OTX output batch data entity if needed for model."""
        if self.training and isinstance(outputs, tuple):
            return outputs

        masks: list[tv_tensors.Mask] = []
        prompts: list[Points] = []
        scores: list[Tensor] = []
        labels: list[Tensor] = []
        for idx, output in enumerate(outputs):
            if not isinstance(output, dict):
                continue
            _masks: list[np.ndarray] = []
            _prompts: list[np.ndarray] = []
            _scores: list[np.ndarray] = []
            _labels: list[np.ndarray] = []
            for label, predicted_mask in output.items():
                if len(predicted_mask.mask) == 0:
                    continue
                _masks.append(np.stack(predicted_mask.mask, axis=0))
                _used_points_scores = np.stack(predicted_mask.scores, axis=0)
                _prompts.append(np.stack(predicted_mask.points, axis=0))
                _scores.append(_used_points_scores)
                _labels.append(np.array([label] * len(_used_points_scores)))

            if len(_masks) == 0:
                masks.append(
                    tv_tensors.Mask(
                        torch.zeros((1, *inputs.imgs_info[idx].ori_shape), dtype=torch.float32, device=self.device),
                    ),
                )
                prompts.append(
                    Points([], canvas_size=inputs.imgs_info[idx].ori_shape, dtype=torch.float32, device=self.device),
                )
                scores.append(torch.tensor([-1.0], dtype=torch.float32, device=self.device))
                labels.append(torch.tensor([-1], dtype=torch.int64, device=self.device))
                continue

            masks.append(tv_tensors.Mask(np.concatenate(_masks, axis=0), dtype=torch.float32, device=self.device))
            prompts.append(
                Points(
                    np.concatenate(_prompts, axis=0),
                    canvas_size=inputs.imgs_info[idx].ori_shape,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            scores.append(torch.as_tensor(np.concatenate(_scores, axis=0), dtype=torch.float32, device=self.device))
            labels.append(torch.as_tensor(np.concatenate(_labels, axis=0), dtype=torch.int64, device=self.device))

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
            numpy_image = data_batch.images[0].cpu().numpy().transpose(1, 2, 0)  # use only the first image
            image, meta = self.model.encoder.preprocess(numpy_image)
            if module == "image_encoder":
                # resize
                resized_image = self.model.encoder.resize(
                    image["images"][0],
                    (self.model.encoder.w, self.model.encoder.h),
                )

                # normalization
                resized_image = self.model.encoder.input_transform(resized_image)

                # change layout from HWC to NCHW
                return self.model.encoder._change_layout(resized_image)  # noqa: SLF001

            # obtain image embeddings from image encoder
            image_embeddings = self.model.encoder.infer_sync(image)

            points: list[np.ndarray] = []
            bboxes: list[np.ndarray] = []
            _labels: dict[str, list[int]] = defaultdict(list)

            # use only the first prompt
            for prompt, label in zip(data_batch.prompts[0], data_batch.labels[0].prompts):  # type: ignore[arg-type]
                if isinstance(prompt, tv_tensors.BoundingBoxes):
                    bboxes.append(prompt.cpu().numpy())
                    _labels["bboxes"].append(label.cpu().numpy())
                elif isinstance(prompt, Points):
                    points.append(prompt.cpu().numpy())
                    _labels["points"].append(label.cpu().numpy())

            # preprocess decoder inputs
            processed_prompts = self.model.decoder.preprocess(
                {
                    "bboxes": bboxes,
                    "points": points,
                    "labels": _labels,
                    "orig_size": meta["original_shape"][:2],
                },
            )

            processed_prompts_w_labels = self._gather_prompts_with_labels(processed_prompts)
            prompt_for_optim = (
                next(iter(processed_prompts_w_labels.values()))[0]
                if isinstance(processed_prompts_w_labels, dict)
                else processed_prompts_w_labels[0][0]
            )  # type: ignore[attr-defined]
            prompt_for_optim.pop("label")
            prompt_for_optim.update(**image_embeddings)
            return prompt_for_optim

        # ticket no. : CVS-135462
        # There is segmentation fault issue when using num_workers > 0 during releasing memory.
        # To avoid this issue, force num_workers to 0.
        data_module.val_subset.num_workers = 0

        output_model_paths: dict[str, Path] = {}
        for module in ["image_encoder", "decoder"]:
            output_model_path = output_dir / (self._OPTIMIZED_MODEL_BASE_NAME + f"_{module}.xml")

            ov_model = openvino.Core().read_model(self.model_names[module])
            if check_if_quantized(ov_model):
                msg = "Model is already optimized by PTQ"
                raise RuntimeError(msg)

            val_dataset = data_module.val_dataloader()

            ptq_config_from_ir = self._read_ptq_config_from_ir(ov_model)
            if ptq_config is not None:
                ptq_config_from_ir.update(ptq_config)
                ptq_config = ptq_config_from_ir
            else:
                ptq_config = ptq_config_from_ir

            quantization_dataset = nncf.Dataset(val_dataset, partial(transform_fn, module=module))  # type: ignore[attr-defined]

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
        image_prompts: list[dict[str, np.ndarray]],
    ) -> dict[int, list[dict[str, np.ndarray]]]:
        """Gather prompts according to labels."""
        processed_prompts: defaultdict[int, list[dict[str, np.ndarray]]] = defaultdict(
            list,
        )
        for prompt in image_prompts:
            processed_prompts[int(prompt["label"])].append(prompt)

        return dict(sorted(processed_prompts.items(), key=lambda x: x))

    ######################################
    #               Learn                #
    ######################################
    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.reference_feats = np.zeros((0, 1, self.model.decoder.embed_dim), dtype=np.float32)
        self.used_indices = np.array([], dtype=np.int64)

    def save_reference_info(self, default_root_dir: Path | str) -> None:
        """Save reference info."""
        reference_info = {
            "reference_feats": self.reference_feats,
            "used_indices": self.used_indices,
        }
        # save reference info
        self.saved_reference_info_path: Path = (
            Path(default_root_dir) / self.reference_info_dir / "reference_info.pickle"
        )
        self.saved_reference_info_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO (sungchul): ticket no. 139210
        torch.save(
            {k: torch.as_tensor(v) for k, v in reference_info.items()},
            self.saved_reference_info_path.with_suffix(".pt"),
        )
        pickle.dump(reference_info, self.saved_reference_info_path.open("wb"))
        log.info(f"Saved reference info at {self.saved_reference_info_path}.")

    ######################################
    #               Infer                #
    ######################################
    def load_reference_info(
        self,
        default_root_dir: Path | str,
        *args,
        path_to_directly_load: Path | None = None,
        **kwargs,
    ) -> bool:
        """Load latest reference info to be used.

        Args:
            default_root_dir (Path | str): Default root directory to be used
                when inappropriate infer_reference_info_root is given.
            path_to_directly_load (Path | None): Reference info path to directly be loaded.
                Normally, it is obtained after `learn` which is executed when trying to do `infer`
                without reference features in `on_test_start` or `on_predict_start`.

        Returns:
            (bool): Whether normally loading checkpoint or not.
        """

        def _load_and_assign_reference_info(path: Path) -> bool:
            reference_info: dict[str, np.ndarray] = pickle.load(path.open("rb"))  # noqa: S301   # nosec: B301
            self.reference_feats = reference_info.get(
                "reference_feats",
                np.zeros((0, 1, self.model.decoder.embed_dim), dtype=np.float32),
            )
            self.used_indices = reference_info.get("used_indices", np.array([], dtype=np.int64))
            log.info(f"reference info saved at {path} was successfully loaded.")
            return True

        if path_to_directly_load is not None:
            # if `path_to_directly_load` is given, forcely load
            return _load_and_assign_reference_info(path_to_directly_load)

        if str(self.infer_reference_info_root) == "../.latest/train":
            # for default setting
            path_reference_info = (
                Path(default_root_dir)
                / self.infer_reference_info_root
                / self.reference_info_dir
                / "reference_info.pickle"
            )
        else:
            # for user input
            path_reference_info = self.infer_reference_info_root / self.reference_info_dir / "reference_info.pickle"

        if path_reference_info.is_file():
            return _load_and_assign_reference_info(path_reference_info)

        return False

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
        if not self.load_reference_info(self.trainer.default_root_dir):
            log.warning("No reference info found. `Learn` will be automatically executed first.")
            self.trainer.lightning_module.automatic_optimization = False
            self.training = True
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(
                self.trainer.default_root_dir,
                path_to_directly_load=self.saved_reference_info_path,
            )

    def on_predict_start(self) -> None:
        """Load previously saved reference info."""
        if not self.load_reference_info(self.trainer.default_root_dir):
            log.warning("No reference info found. `Learn` will be automatically executed first.")
            self.trainer.lightning_module.automatic_optimization = False
            self.training = True
            self.trainer.fit_loop.run()
            # to use infer logic
            self.training = False
            # to set _combined_loader
            self.trainer._evaluation_loop.setup_data()  # noqa: SLF001
            self.trainer._evaluation_loop.reset()  # noqa: SLF001
            self.load_reference_info(
                self.trainer.default_root_dir,
                path_to_directly_load=self.saved_reference_info_path,
            )

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

    def get_dummy_input(self, batch_size: int = 1) -> ZeroShotVisualPromptingBatchDataEntity:
        """Returns a dummy input for classification OV model."""
        # Resize is embedded to the OV model, which means we don't need to know the actual size
        images = [torch.rand(3, 224, 224) for _ in range(batch_size)]
        labels = [ZeroShotVisualPromptingLabel(prompts=torch.LongTensor([0]))] * batch_size
        prompts = [torch.zeros((1, 2))] * batch_size
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return ZeroShotVisualPromptingBatchDataEntity(
            batch_size,
            images,
            imgs_info=infos,
            labels=labels,
            prompts=prompts,
            masks=[None] * batch_size,
            polygons=[[None]] * batch_size,
        )
