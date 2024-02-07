# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch

from otx.core.data.dataset.classification import HLabelMetaInfo
from otx.core.data.entity.base import OTXBatchLossEntity, T_OTXBatchDataEntity, T_OTXBatchPredEntity
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from mmpretrain.models.utils import ClsDataPreprocessor
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import ClassificationResult
    from torch import nn

    from otx.core.data.entity.classification import HLabelInfo


class ExplainableOTXClsModel(OTXModel[T_OTXBatchDataEntity, T_OTXBatchPredEntity, T_OTXTileBatchDataEntity]):
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
        from otx.algo.hooks.recording_forward_hook import ReciproCAMHook

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
        return head([output])

    def remove_explain_hook_handle(self) -> None:
        """Removes explain hook from the model."""
        if self.explain_hook.handle is not None:
            self.explain_hook.handle.remove()

    def reset_explain_hook(self) -> None:
        """Clear all history of explain records."""
        self.explain_hook.reset()


class OTXMulticlassClsModel(
    ExplainableOTXClsModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity, T_OTXTileBatchDataEntity],
):
    """Base class for the classification models used in OTX."""

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
        self.image_size = (1, 3, 224, 224)
        super().__init__(num_classes=num_classes)

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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)


### NOTE, currently, although we've made the separate Multi-cls, Multi-label classes
### It'll be integrated after H-label classification integration with more advanced design.


class OTXMultilabelClsModel(
    ExplainableOTXClsModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity, T_OTXTileBatchDataEntity],
):
    """Multi-label classification models used in OTX."""

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
        self.image_size = (1, 3, 224, 224)
        super().__init__(num_classes=num_classes)

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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)


class OTXHlabelClsModel(
    ExplainableOTXClsModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity, T_OTXTileBatchDataEntity],
):
    """H-label classification models used in OTX."""

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        hierarchical_config: dict = {}

        label_info: HLabelMetaInfo = self.label_info  # type: ignore[assignment]
        hierarchical_config["cls_heads_info"] = {
            "num_multiclass_heads": label_info.hlabel_info.num_multiclass_heads,
            "num_multilabel_classes": label_info.hlabel_info.num_multilabel_classes,
            "head_idx_to_logits_range": label_info.hlabel_info.head_idx_to_logits_range,
            "num_single_label_classes": label_info.hlabel_info.num_single_label_classes,
            "class_to_group_idx": label_info.hlabel_info.class_to_group_idx,
            "all_groups": label_info.hlabel_info.all_groups,
            "label_to_idx": label_info.hlabel_info.label_to_idx,
            "empty_multiclass_head_indices": label_info.hlabel_info.empty_multiclass_head_indices,
        }
        hierarchical_config["label_tree_edges"] = label_info.hlabel_info.label_tree_edges

        parameters["metadata"].update(
            {
                ("model_info", "model_type"): "Classification",
                ("model_info", "task_type"): "classification",
                ("model_info", "multilabel"): str(False),
                ("model_info", "hierarchical"): str(True),
                ("model_info", "confidence_threshold"): str(0.5),
                ("model_info", "hierarchical_config"): json.dumps(hierarchical_config),
            },
        )
        return parameters


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
        self.image_size = (1, 3, 224, 224)
        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        from .utils.mmpretrain import create_model

        model, classification_layers = create_model(self.config, self.load_from)
        self.classification_layers = classification_layers
        return model

    def set_hlabel_info(self, hierarchical_info: HLabelInfo) -> None:
        """Set hierarchical information in model head.

        Args:
            hierarchical_info: the label information represents the hierarchy.
        """
        self.model.head.set_hlabel_info(hierarchical_info)

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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)


class OVMulticlassClassificationModel(
    OVModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity],
):
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


class OVHlabelClassificationModel(
    OVModel[HlabelClsBatchDataEntity, HlabelClsBatchPredEntity],
):
    """Hierarchical classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_type: str,
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        num_multiclass_heads: int = 1,
        num_multilabel_classes: int = 0,
    ) -> None:
        self.num_multiclass_heads = num_multiclass_heads
        self.num_multilabel_classes = num_multilabel_classes
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"hierarchical": True, "confidence_threshold": 0.0})
        super().__init__(
            num_classes,
            model_name,
            model_type,
            async_inference,
            max_num_requests,
            use_throughput_mode,
            model_api_configuration,
        )

    def set_hlabel_info(self, hierarchical_info: HLabelInfo) -> None:
        """Set hierarchical information in model head.

        Since OV IR model consist of all required hierarchy information,
        this method serves as placehloder
        """
        if not hasattr(self.model, "hierarchical_info") or not self.model.hierarchical_info:
            msg = "OpenVINO IR model should have hierarchical config embedded in rt_info of the model"
            raise ValueError(msg)

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity:
        pred_labels = [torch.tensor([label[0] for label in out.top_labels], dtype=torch.long) for out in outputs]
        pred_scores = [torch.tensor([label[2] for label in out.top_labels]) for out in outputs]

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=pred_labels,
        )


class OVMultilabelClassificationModel(
    OVModel[MultilabelClsBatchDataEntity, MultilabelClsBatchPredEntity],
):
    """Multilabel classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_type: str,
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"multilabel": True, "confidence_threshold": 0.0})
        super().__init__(
            num_classes,
            model_name,
            model_type,
            async_inference,
            max_num_requests,
            use_throughput_mode,
            model_api_configuration,
        )

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
