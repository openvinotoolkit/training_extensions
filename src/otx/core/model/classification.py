# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

import json
import types
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torchmetrics import Accuracy

from otx.algo.hooks.recording_forward_hook import feature_vector_fn
from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
    T_OTXBatchPredEntityWithXAI,
)
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    HlabelClsBatchPredEntityWithXAI,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MulticlassClsBatchPredEntityWithXAI,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
    MultilabelClsBatchPredEntityWithXAI,
)
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import (
    HLabelClsMetricCallble,
    MultiClassClsMetricCallable,
    MultiLabelClsMetricCallable,
)
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.types.label import HLabelInfo
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from mmpretrain.models import ImageClassifier
    from mmpretrain.models.utils import ClsDataPreprocessor
    from mmpretrain.structures import DataSample
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import ClassificationResult
    from torch import nn

    from otx.core.metrics import MetricCallable


class ExplainableOTXClsModel(
    OTXModel[T_OTXBatchDataEntity, T_OTXBatchPredEntity, T_OTXBatchPredEntityWithXAI, T_OTXTileBatchDataEntity],
):
    """OTX classification model which can attach a XAI hook."""

    @property
    def has_gap(self) -> bool:
        """Defines if GAP is used right after backbone. Can be redefined at the model's level."""
        return True

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params = super()._export_parameters
        export_params["output_names"] = ["logits", "feature_vector", "saliency_map"] if self.explain_mode else None
        return export_params

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward. Can be redefined at the model's level."""
        if (neck := getattr(self.model, "neck", None)) is None:
            raise ValueError
        if (head := getattr(self.model, "head", None)) is None:
            raise ValueError

        output = neck(x)
        return head([output])

    def forward_explain(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntityWithXAI:
        """Model forward function."""
        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()

        # If customize_inputs is overridden
        outputs = (
            self._forward_explain_image_classifier(self.model, **self._customize_inputs(inputs))
            if self._customize_inputs != ExplainableOTXClsModel._customize_inputs
            else self._forward_explain_image_classifier(self.model, inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != ExplainableOTXClsModel._customize_outputs
            else outputs["predictions"]
        )

    @staticmethod
    def _forward_explain_image_classifier(
        self: ImageClassifier,
        inputs: torch.Tensor,
        data_samples: list[DataSample] | None = None,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor]:
        """Forward func of the ImageClassifier instance, which located in ExplainableOTXClsModel().model."""
        x = self.backbone(inputs)
        backbone_feat = x

        feature_vector = self.feature_vector_fn(backbone_feat)
        saliency_map = self.explain_fn(backbone_feat)

        if self.with_neck:
            x = self.neck(x)

        if mode == "tensor":
            logits = self.head(x) if self.with_head else x
        elif mode == "predict":
            logits = self.head.predict(x, data_samples)
        else:
            msg = f'Invalid mode "{mode}".'
            raise RuntimeError(msg)

        return {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        from otx.algo.hooks.recording_forward_hook import ReciproCAMHook

        explainer = ReciproCAMHook(
            self.head_forward_fn,
            num_classes=self.num_classes,
            optimize_gap=self.has_gap,
        )
        return explainer.func

    def _reset_model_forward(self) -> None:
        if not self.explain_mode:
            return

        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()
        forward_with_explain = self._forward_explain_image_classifier

        self.original_model_forward = self.model.forward

        func_type = types.MethodType
        self.model.forward = func_type(forward_with_explain, self.model)

    def _restore_model_forward(self) -> None:
        if not self.explain_mode:
            return

        if not self.original_model_forward:
            msg = "Original model forward was not saved."
            raise RuntimeError(msg)

        func_type = types.MethodType
        self.model.forward = func_type(self.original_model_forward, self.model)
        self.original_model_forward = None

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)


class OTXMulticlassClsModel(
    ExplainableOTXClsModel[
        MulticlassClsBatchDataEntity,
        MulticlassClsBatchPredEntity,
        MulticlassClsBatchPredEntityWithXAI,
        T_OTXTileBatchDataEntity,
    ],
):
    """Base class for the classification models used in OTX."""

    def __init__(
        self,
        num_classes: int,
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        parameters["metadata"].update(
            {
                ("model_info", "model_type"): "Classification",
                ("model_info", "task_type"): "classification",
                ("model_info", "multilabel"): str(False),
                ("model_info", "hierarchical"): str(False),
            },
        )
        return parameters

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity | MulticlassClsBatchPredEntityWithXAI,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }


class MMPretrainMulticlassClsModel(OTXMulticlassClsModel):
    """Multi-class Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(
        self,
        num_classes: int,
        config: DictConfig,
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size = (1, 3, 224, 224)
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        from .utils.mmpretrain import create_model

        model, self.classification_layers = create_model(self.config, self.load_from)
        return model

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: dict[str, Any],
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | MulticlassClsBatchPredEntityWithXAI | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        predictions = outputs["logits"] if isinstance(outputs, dict) else outputs
        scores = []
        labels = []

        for output in predictions:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            feature_vectors = outputs["feature_vector"].detach().cpu().numpy()
            saliency_maps = outputs["saliency_map"].detach().cpu().numpy()

            return MulticlassClsBatchPredEntityWithXAI(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                labels=labels,
                feature_vectors=list(feature_vectors),
                saliency_maps=list(saliency_maps),
            )

        return MulticlassClsBatchPredEntity(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params = super()._export_parameters
        export_params.update(get_mean_std_from_data_processing(self.config))
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = self.image_size
        export_params["onnx_export_configuration"] = None

        return export_params


### NOTE, currently, although we've made the separate Multi-cls, Multi-label classes
### It'll be integrated after H-label classification integration with more advanced design.


class OTXMultilabelClsModel(
    ExplainableOTXClsModel[
        MultilabelClsBatchDataEntity,
        MultilabelClsBatchPredEntity,
        MultilabelClsBatchPredEntityWithXAI,
        T_OTXTileBatchDataEntity,
    ],
):
    """Multi-label classification models used in OTX."""

    def __init__(
        self,
        num_classes: int,
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        parameters["metadata"].update(
            {
                ("model_info", "model_type"): "Classification",
                ("model_info", "task_type"): "classification",
                ("model_info", "multilabel"): str(True),
                ("model_info", "hierarchical"): str(False),
                ("model_info", "confidence_threshold"): str(0.5),
            },
        )
        return parameters

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MultilabelClsBatchPredEntity | MultilabelClsBatchPredEntityWithXAI,
        inputs: MultilabelClsBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }


class MMPretrainMultilabelClsModel(OTXMultilabelClsModel):
    """Multi-label Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(
        self,
        num_classes: int,
        config: DictConfig,
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = lambda num_labels: Accuracy(task="multilabel", num_labels=num_labels),
        torch_compile: bool = False,
    ) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size = (1, 3, 224, 224)
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        from .utils.mmpretrain import create_model

        model, classification_layers = create_model(self.config, self.load_from)
        self.classification_layers = classification_layers
        return model

    def _customize_inputs(self, entity: MultilabelClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                    "ignored_labels": img_info.ignored_labels,
                },
                gt_score=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity | MultilabelClsBatchPredEntityWithXAI | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        predictions = outputs["logits"] if isinstance(outputs, dict) else outputs
        scores = []
        labels = []

        for output in predictions:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            feature_vectors = outputs["feature_vector"].detach().cpu().numpy()
            saliency_maps = outputs["saliency_map"].detach().cpu().numpy()

            return MultilabelClsBatchPredEntityWithXAI(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                labels=labels,
                feature_vectors=list(feature_vectors),
                saliency_maps=list(saliency_maps),
            )

        return MultilabelClsBatchPredEntity(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params = super()._export_parameters
        export_params.update(get_mean_std_from_data_processing(self.config))
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = self.image_size
        export_params["onnx_export_configuration"] = None

        return export_params


class OTXHlabelClsModel(
    ExplainableOTXClsModel[
        HlabelClsBatchDataEntity,
        HlabelClsBatchPredEntity,
        HlabelClsBatchPredEntityWithXAI,
        T_OTXTileBatchDataEntity,
    ],
):
    """H-label classification models used in OTX."""

    def __init__(
        self,
        hlabel_info: HLabelInfo,
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallble,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            num_classes=hlabel_info.num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        # NOTE: This should be behind of super().__init__() to avoid overwriting
        self._label_info = hlabel_info

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        hierarchical_config: dict = {}

        label_info: HLabelInfo = self.label_info  # type: ignore[assignment]
        hierarchical_config["cls_heads_info"] = label_info.as_dict()
        hierarchical_config["label_tree_edges"] = label_info.label_tree_edges

        parameters["metadata"].update(
            {
                ("model_info", "model_type"): "Classification",
                ("model_info", "task_type"): "classification",
                ("model_info", "multilabel"): str(False),
                ("model_info", "hierarchical"): str(True),
                ("model_info", "confidence_threshold"): str(0.5),
                ("model_info", "hierarchical_config"): json.dumps(hierarchical_config),
                # NOTE: There is currently too many channels for label related metadata.
                # This should be clean up afterwards in ModelAPI side.
                ("model_info", "label_info"): json.dumps(label_info.as_dict()),
            },
        )
        return parameters

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity | HlabelClsBatchPredEntityWithXAI,
        inputs: HlabelClsBatchDataEntity,
    ) -> MetricInput:
        hlabel_info: HLabelInfo = self.label_info  # type: ignore[assignment]

        if hlabel_info.num_multilabel_classes > 0:
            preds_multiclass = torch.stack(preds.labels)[:, : hlabel_info.num_multiclass_heads]
            preds_multilabel = torch.stack(preds.scores)[:, hlabel_info.num_multiclass_heads :]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = torch.stack(preds.labels)
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }


class MMPretrainHlabelClsModel(OTXHlabelClsModel):
    """H-label Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(
        self,
        hlabel_info: HLabelInfo,
        config: DictConfig,
        optimizer: list[OptimizerCallable] | OptimizerCallable = DefaultOptimizerCallable,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallble,
        torch_compile: bool = False,
    ) -> None:
        config = inplace_num_classes(cfg=config, num_classes=hlabel_info.num_classes)

        if (head_config := getattr(config, "head", None)) is None:
            msg = 'Config should have "head" section'
            raise ValueError(msg)

        head_config.update(**hlabel_info.as_head_config_dict())

        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size = (1, 3, 224, 224)
        super().__init__(
            hlabel_info=hlabel_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        from .utils.mmpretrain import create_model

        model, classification_layers = create_model(self.config, self.load_from)
        self.classification_layers = classification_layers
        return model

    def _customize_inputs(self, entity: HlabelClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                    "ignored_labels": img_info.ignored_labels,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity | HlabelClsBatchPredEntityWithXAI | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        predictions = outputs["logits"] if isinstance(outputs, dict) else outputs
        scores = []
        labels = []

        for output in predictions:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            feature_vectors = outputs["feature_vector"].detach().cpu().numpy()
            saliency_maps = outputs["saliency_map"].detach().cpu().numpy()

            return HlabelClsBatchPredEntityWithXAI(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                labels=labels,
                feature_vectors=list(feature_vectors),
                saliency_maps=list(saliency_maps),
            )

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params = super()._export_parameters
        export_params.update(get_mean_std_from_data_processing(self.config))
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = self.image_size
        export_params["onnx_export_configuration"] = None

        return export_params


class OVMulticlassClassificationModel(
    OVModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity, MulticlassClsBatchPredEntityWithXAI],
):
    """Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = False,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiClassClsMetricCallable,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | MulticlassClsBatchPredEntityWithXAI:
        pred_labels = [torch.tensor(out.top_labels[0][0], dtype=torch.long) for out in outputs]
        pred_scores = [torch.tensor(out.top_labels[0][2]) for out in outputs]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return MulticlassClsBatchPredEntityWithXAI(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=pred_labels,
                saliency_maps=predicted_s_maps,
                feature_vectors=predicted_f_vectors,
            )

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=pred_labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity | MulticlassClsBatchPredEntityWithXAI,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }


class OVMultilabelClassificationModel(
    OVModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity, MultilabelClsBatchPredEntityWithXAI],
):
    """Multilabel classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        **kwargs,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"multilabel": True, "confidence_threshold": 0.0})
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity | MultilabelClsBatchPredEntityWithXAI:
        pred_scores = [torch.tensor([top_label[2] for top_label in out.top_labels]) for out in outputs]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return MultilabelClsBatchPredEntityWithXAI(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=[],
                saliency_maps=predicted_s_maps,
                feature_vectors=predicted_f_vectors,
            )

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=[],
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MultilabelClsBatchPredEntity | MultilabelClsBatchPredEntityWithXAI,
        inputs: MultilabelClsBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }


class OVHlabelClassificationModel(
    OVModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity, HlabelClsBatchPredEntityWithXAI],
):
    """Hierarchical classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = HLabelClsMetricCallble,
        **kwargs,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"hierarchical": True, "output_raw_scores": True})
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity | HlabelClsBatchPredEntityWithXAI:
        all_pred_labels = []
        all_pred_scores = []
        for output in outputs:
            logits = output.raw_scores
            predicted_labels = []
            predicted_scores = []
            cls_heads_info = self.model.hierarchical_info["cls_heads_info"]
            for i in range(cls_heads_info["num_multiclass_heads"]):
                logits_begin, logits_end = cls_heads_info["head_idx_to_logits_range"][str(i)]
                head_logits = logits[logits_begin:logits_end]
                j = np.argmax(head_logits)
                predicted_labels.append(j)
                predicted_scores.append(head_logits[j])

            if cls_heads_info["num_multilabel_classes"]:
                logits_begin = cls_heads_info["num_single_label_classes"]
                head_logits = logits[logits_begin:]

                for i in range(head_logits.shape[0]):
                    predicted_scores.append(head_logits[i])
                    if head_logits[i] > self.model.confidence_threshold:
                        predicted_labels.append(1)
                    else:
                        predicted_labels.append(0)

            all_pred_labels.append(torch.tensor(predicted_labels, dtype=torch.long))
            all_pred_scores.append(torch.tensor(predicted_scores))

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return HlabelClsBatchPredEntityWithXAI(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=all_pred_scores,
                labels=all_pred_labels,
                saliency_maps=predicted_s_maps,
                feature_vectors=predicted_f_vectors,
            )

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=all_pred_scores,
            labels=all_pred_labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity | HlabelClsBatchPredEntityWithXAI,
        inputs: HlabelClsBatchDataEntity,
    ) -> MetricInput:
        cls_heads_info = self.model.hierarchical_info["cls_heads_info"]
        num_multilabel_classes = cls_heads_info["num_multilabel_classes"]
        num_multiclass_heads = cls_heads_info["num_multiclass_heads"]
        if num_multilabel_classes > 0:
            preds_multiclass = torch.stack(preds.labels)[:, :num_multiclass_heads]
            preds_multilabel = torch.stack(preds.scores)[:, num_multiclass_heads:]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = torch.stack(preds.labels)
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }

    def _create_label_info_from_ov_ir(self) -> HLabelInfo:
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            return HLabelInfo.from_json(serialized)

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)
