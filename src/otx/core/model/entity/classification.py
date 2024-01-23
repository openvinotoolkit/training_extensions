# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch

from otx.algo.hooks.recording_forward_hook import ReciproCAMHook
from otx.core.data.entity.base import OTXBatchLossEntity, T_OTXBatchDataEntity, T_OTXBatchPredEntity
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.utils.utils import get_mean_std_from_data_processing
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from mmpretrain.models.utils import ClsDataPreprocessor
    from omegaconf import DictConfig
    from openvino.model_api.models import Model
    from openvino.model_api.models.utils import ClassificationResult
    from torch import device, nn


class ExplainableOTXClsModel(OTXModel[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """OTX classification model which can attach a XAI hook."""

    @property
    def has_gap(self) -> bool:
        """Defines if GAP is used right after backbone. Can be redefined at the model's level."""
        return True

    @property
    def backbone(self) -> nn.Module:
        """Returns model's backbone. Can be redefined at the model's level."""
        if backbone := getattr(self.model, "backbone", None):
            return backbone
        raise ValueError

    def register_explain_hook(self) -> None:
        """Register explain hook at the model backbone output."""
        self.explain_hook = ReciproCAMHook.create_and_register_hook(
            self.backbone,
            self.head_forward_fn,
            num_classes=self.num_classes,
            optimize_gap=self.has_gap,
        )

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward. Can be redefined at the model's level."""
        if (neck := getattr(self.model, "neck", None)) is None:
            raise ValueError
        if (head := getattr(self.model, "head", None)) is None:
            raise ValueError

        output = neck(x)
        return head(output)

    def remove_explain_hook_handle(self) -> None:
        """Removes explain hook from the model."""
        if self.explain_hook.handle is not None:
            self.explain_hook.handle.remove()

    def reset_explain_hook(self) -> None:
        """Clear all history of explain records."""
        self.explain_hook.reset()


class OTXMulticlassClsModel(ExplainableOTXClsModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity]):
    """Base class for the classification models used in OTX."""

    def _generate_model_metadata(
        self,
    ) -> dict[tuple[str, str], Any]:
        metadata = super()._generate_model_metadata()
        metadata[("model_info", "model_type")] = "Classification"
        metadata[("model_info", "task_type")] = "classification"
        metadata[("model_info", "multilabel")] = str(False)
        metadata[("model_info", "hierarchical")] = str(False)
        return metadata


def _create_mmpretrain_model(config: DictConfig, load_from: str) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    from mmpretrain.models.utils import ClsDataPreprocessor as _ClsDataPreprocessor
    from mmpretrain.registry import MODELS

    # NOTE: For the history of this monkey patching, please see
    # https://github.com/openvinotoolkit/training_extensions/issues/2743
    @MODELS.register_module(force=True)
    class ClsDataPreprocessor(_ClsDataPreprocessor):
        @property
        def device(self) -> device:
            try:
                buf = next(self.buffers())
            except StopIteration:
                return super().device
            else:
                return buf.device

    classification_layers = get_classification_layers(config, MODELS, "model.")
    return build_mm_model(config, MODELS, load_from), classification_layers


class MMPretrainMulticlassClsModel(OTXMulticlassClsModel):
    """Multi-class Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size = (224, 224)
        super().__init__(num_classes=num_classes)

    @property
    def export_params(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = (1, 3, *self.image_size)
        export_params["onnx_export_configuration"] = None

        return export_params

    def _create_model(self) -> nn.Module:
        model, classification_layers = _create_mmpretrain_model(self.config, self.load_from)
        self.classification_layers = classification_layers
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
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    def _create_exporter(
        self,
        test_pipeline: list[dict] | None = None,
    ) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self.export_params)


### NOTE, currently, although we've made the separate Multi-cls, Multi-label classes
### It'll be integrated after H-label classification integration with more advanced design.


class OTXMultilabelClsModel(ExplainableOTXClsModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity]):
    """Multi-label classification models used in OTX."""

    def _generate_model_metadata(
        self,
    ) -> dict[tuple[str, str], Any]:
        metadata = super()._generate_model_metadata()
        metadata[("model_info", "model_type")] = "Classification"
        metadata[("model_info", "task_type")] = "classification"
        metadata[("model_info", "multilabel")] = str(True)
        metadata[("model_info", "hierarchical")] = str(False)
        metadata[("model_info", "confidence_threshold")] = str(0.5)
        return metadata


class MMPretrainMultilabelClsModel(OTXMultilabelClsModel):
    """Multi-label Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size = (224, 224)
        super().__init__(num_classes=num_classes)

    @property
    def export_params(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = (1, 3, *self.image_size)
        export_params["onnx_export_configuration"] = None

        return export_params

    def _create_model(self) -> nn.Module:
        model, classification_layers = _create_mmpretrain_model(self.config, self.load_from)
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
    ) -> MultilabelClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    def _create_exporter(
        self,
        test_pipeline: list[dict] | None = None,
    ) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self.export_params)


class OTXHlabelClsModel(
    ExplainableOTXClsModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity],
):
    """H-label classification models used in OTX."""

    def _generate_model_metadata(
        self,
    ) -> dict[tuple[str, str], Any]:
        metadata = super()._generate_model_metadata()
        metadata[("model_info", "model_type")] = "Classification"
        metadata[("model_info", "task_type")] = "classification"
        metadata[("model_info", "multilabel")] = str(False)
        metadata[("model_info", "hierarchical")] = str(True)
        metadata[("model_info", "confidence_threshold")] = str(0.5)
        hierarchical_config: dict = {}
        hierarchical_config["cls_heads_info"] = {}
        hierarchical_config["label_tree_edges"] = []

        metadata[("model_info", "hierarchical_config")] = json.dumps(hierarchical_config)

        return metadata


class MMPretrainHlabelClsModel(OTXHlabelClsModel):
    """H-label Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size = (224, 224)
        super().__init__(num_classes=num_classes)

    @property
    def export_params(self) -> dict[str, Any]:
        """Parameters for an exporter."""
        export_params = get_mean_std_from_data_processing(self.config)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = (1, 3, *self.image_size)
        export_params["onnx_export_configuration"] = None

        return export_params

    def _create_model(self) -> nn.Module:
        model, classification_layers = _create_mmpretrain_model(self.config, self.load_from)
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
    ) -> HlabelClsBatchPredEntity | OTXBatchLossEntity:
        from mmpretrain.structures import DataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    def _create_exporter(
        self,
        test_pipeline: list[dict] | None = None,
    ) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self.export_params)


class OVMulticlassClassificationModel(OVModel):
    """Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity:
        pred_labels = [torch.tensor(out.top_labels[0][0], dtype=torch.long) for out in outputs]
        pred_scores = [torch.tensor(out.top_labels[0][2]) for out in outputs]

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=pred_labels,
        )


class OVMultilabelClassificationModel(OVModel):
    """Multilabel classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def _create_model(self, *args) -> Model:
        # confidence_threshold is 0.0 to return scores for all classes
        configuration = {"multilabel": True, "confidence_threshold": 0.0}
        return super()._create_model(configuration)

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity:
        pred_scores = [torch.tensor([top_label[2] for top_label in out.top_labels]) for out in outputs]

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=[],
        )
