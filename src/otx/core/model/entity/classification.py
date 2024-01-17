# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.types.export import OTXExportFormatType, OTXExportPrecisionType
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from mmpretrain.models.utils import ClsDataPreprocessor
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import ClassificationResult
    from torch import device, nn


class OTXMulticlassClsModel(
    OTXModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity],
):
    """Base class for the classification models used in OTX."""

    def _generate_model_metadata(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        resize_mode: str,
        pad_value: int,
        swap_rgb: bool,
    ) -> dict[tuple[str, str], Any]:
        metadata = super()._generate_model_metadata(mean, std, resize_mode, pad_value, swap_rgb)
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


def _get_export_params_from_cls_mmconfig(config: DictConfig):
    return {
            "mean": config["data_preprocessor"]["mean"],
            "std": config["data_preprocessor"]["std"],
            }


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
        super().__init__(num_classes=num_classes)

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

    def export(
        self,
        output_dir: Path,
        export_format: OTXExportFormatType,
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
    ) -> None:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory path to save exported binary files.
            export_format: Format in which this `OTXModel` is exported.
            precision: Precision of the exported model.
        """
        export_params = _get_export_params_from_cls_mmconfig(self.config)
        self._export(output_dir, export_format, precision=precision,
                     onnx_export_configuration=None,
                     pad_value = 0,
                     resize_mode = "standard",
                     swap_rgb = False,
                     via_onnx = False,
                     input_size=(1,3,224,224),
                    **export_params)


### NOTE, currently, although we've made the separate Multi-cls, Multi-label classes
### It'll be integrated after H-label classification integration with more advanced design.


class OTXMultilabelClsModel(
    OTXModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity],
):
    """Multi-label classification models used in OTX."""

    def _generate_model_metadata(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        resize_mode: str,
        pad_value: int,
        swap_rgb: bool,
    ) -> dict[tuple[str, str], Any]:
        metadata = super()._generate_model_metadata(mean, std, resize_mode, pad_value, swap_rgb)
        metadata[("model_info", "model_type")] = "Classification"
        metadata[("model_info", "task_type")] = "classification"
        metadata[("model_info", "multilabel")] = str(True)
        metadata[("model_info", "hierarchical")] = str(False)
        metadata[("model_info", "multilabel")] = str(0.5)
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
        super().__init__(num_classes=num_classes)

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

    def export(
        self,
        output_dir: Path,
        export_format: OTXExportFormatType,
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
    ) -> None:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory path to save exported binary files.
            export_format: Format in which this `OTXModel` is exported.
            precision: Precision of the exported model.
        """
        export_params = _get_export_params_from_cls_mmconfig(self.config)
        self._export(output_dir, export_format, precision=precision,
                     onnx_export_configuration=None,
                     pad_value = 0,
                     resize_mode = "standard",
                     swap_rgb = False,
                     via_onnx = False,
                     input_size=(1,3,224,224),
                    **export_params)


class OTXHlabelClsModel(OTXModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity]):
    """H-label classification models used in OTX."""

    def _generate_model_metadata(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        resize_mode: str,
        pad_value: int,
        swap_rgb: bool,
    ) -> dict[tuple[str, str], Any]:
        metadata = super()._generate_model_metadata(mean, std, resize_mode, pad_value, swap_rgb)
        metadata[("model_info", "model_type")] = "Classification"
        metadata[("model_info", "task_type")] = "classification"
        metadata[("model_info", "multilabel")] = str(False)
        metadata[("model_info", "hierarchical")] = str(True)
        metadata[("model_info", "multilabel")] = str(0.5)
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
        super().__init__(num_classes=num_classes)

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

    def export(
        self,
        output_dir: Path,
        export_format: OTXExportFormatType,
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
    ) -> None:
        """Export this model to the specified output directory.

        Args:
            output_dir: Directory path to save exported binary files.
            export_format: Format in which this `OTXModel` is exported.
            precision: Precision of the exported model.
        """
        export_params = _get_export_params_from_cls_mmconfig(self.config)
        self._export(output_dir, export_format, precision=precision,
                     onnx_export_configuration=None,
                     pad_value = 0,
                     resize_mode = "standard",
                     swap_rgb = False,
                     via_onnx = False,
                     input_size=(1,3,224,224),
                    **export_params)


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
