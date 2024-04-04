# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MobileNetV3 model implementation."""


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
from torch import nn

from otx.algo.classification.backbones import OTXMobileNetV3
from otx.algo.classification.classifier.base_classifier import ImageClassifier
from otx.algo.classification.heads import HierarchicalNonLinearClsHead, LinearClsHead, MultiLabelNonLinearClsHead
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import OTXBatchLossEntity
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
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import HLabelClsMetricCallble, MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import OTXHlabelClsModel, OTXMulticlassClsModel, OTXMultilabelClsModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class MobileNetV3ForMulticlassCls(OTXMulticlassClsModel):
    """MobileNetV3ForMulticlassCls is a class that represents a MobileNetV3 model for multiclass classification.

    Args:
        mode (Literal["large", "small"]): The mode of the MobileNetV3 model, either "large" or "small".
        num_classes (int): The number of classes for classification.
        loss_callable (Callable[[], nn.Module], optional): The loss function callable. Defaults to nn.CrossEntropyLoss.
        optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        freeze_backbone (bool, optional): Whether to freeze the backbone layers during training. Defaults to False.
    """

    def __init__(
        self,
        mode: Literal["large", "small"],
        num_classes: int,
        loss_callable: Callable[[], nn.Module] = nn.CrossEntropyLoss,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.mode = mode
        self.head_config = {"loss_callable": loss_callable}

        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        loss = self.head_config["loss_callable"]
        return ImageClassifier(
            backbone=OTXMobileNetV3(mode=self.mode),
            neck=GlobalAveragePooling(dim=2),
            head=LinearClsHead(
                num_classes=self.num_classes,
                in_channels=960,
                topk=(1, 5) if self.num_classes >= 5 else (1,),
                loss=loss if isinstance(loss, nn.Module) else loss(),
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "multiclass", add_prefix)

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.cat(inputs.labels, dim=0),
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)
        preds = logits.argmax(-1, keepdim=True).unbind(0)

        return MulticlassClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params: dict[str, Any] = {}
        export_params["input_size"] = (1, 3, 224, 224)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = True
        export_params["onnx_export_configuration"] = None
        export_params["mean"] = [123.675, 116.28, 103.53]
        export_params["std"] = [58.395, 57.12, 57.375]

        parameters = super()._export_parameters
        parameters.update(export_params)

        return parameters

    def forward_explain(self, inputs: MulticlassClsBatchDataEntity) -> MulticlassClsBatchPredEntity:
        """Model forward explain function."""
        outputs = self.model(images=inputs.stacked_images, mode="explain")

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["preds"],
            scores=outputs["scores"],
            saliency_map=outputs["saliency_map"],
            feature_vector=outputs["feature_vector"],
        )


class MobileNetV3ForMultilabelCls(OTXMultilabelClsModel):
    """MobileNetV3 Model for multi-class classification task."""

    def __init__(
        self,
        mode: Literal["large", "small"],
        num_classes: int,
        loss_callable: Callable[[], nn.Module] = nn.CrossEntropyLoss,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.mode = mode
        self.head_config = {"loss_callable": loss_callable}

        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        loss = self.head_config["loss_callable"]
        return ImageClassifier(
            backbone=OTXMobileNetV3(mode=self.mode),
            neck=GlobalAveragePooling(dim=2),
            head=MultiLabelNonLinearClsHead(
                num_classes=self.num_classes,
                in_channels=960,
                hid_channels=1280,
                normalized=True,
                scale=7.0,
                activation_callable=nn.PReLU(),
                loss=loss if isinstance(loss, nn.Module) else loss(),
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "multilabel", add_prefix)

    def _customize_inputs(self, inputs: MultilabelClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.stack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MultilabelClsBatchDataEntity,
    ) -> MultilabelClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)
        preds = logits.argmax(-1, keepdim=True).unbind(0)

        return MultilabelClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params: dict[str, Any] = {}
        export_params["input_size"] = (1, 3, 224, 224)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = True
        export_params["onnx_export_configuration"] = None
        export_params["mean"] = [123.675, 116.28, 103.53]
        export_params["std"] = [58.395, 57.12, 57.375]

        parameters = super()._export_parameters
        parameters.update(export_params)

        return parameters


class MobileNetV3ForHLabelCls(OTXHlabelClsModel):
    """MobileNetV3 Model for hierarchical label classification task."""

    def __init__(
        self,
        mode: Literal["large", "small"],
        hlabel_info: HLabelInfo,
        multiclass_loss_callable: Callable[[], nn.Module] = nn.CrossEntropyLoss,
        multilabel_loss_callable: Callable[[], nn.Module] = nn.CrossEntropyLoss,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallble,
        torch_compile: bool = False,
    ) -> None:
        self.mode = mode
        self.head_config = {
            "multiclass_loss_callable": multiclass_loss_callable,
            "multilabel_loss_callable": multilabel_loss_callable,
        }
        self.hlabel_info = hlabel_info

        super().__init__(
            hlabel_info=hlabel_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        multiclass_loss = self.head_config["multiclass_loss_callable"]
        multilabel_loss = self.head_config["multilabel_loss_callable"]
        return ImageClassifier(
            backbone=OTXMobileNetV3(mode=self.mode),
            neck=GlobalAveragePooling(dim=2),
            head=HierarchicalNonLinearClsHead(
                in_channels=960,
                multiclass_loss=multiclass_loss if isinstance(multiclass_loss, nn.Module) else multiclass_loss(),
                multilabel_loss=multilabel_loss if isinstance(multilabel_loss, nn.Module) else multilabel_loss(),
                **self.hlabel_info.as_head_config_dict(),
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "hlabel", add_prefix)

    def _customize_inputs(self, inputs: HlabelClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.stack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        # To list, batch-wise
        scores = outputs["pred_scores"]
        labels = outputs["pred_labels"]

        return HlabelClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity,
        inputs: HlabelClsBatchDataEntity,
    ) -> MetricInput:
        hlabel_info: HLabelInfo = self.label_info  # type: ignore[assignment]

        _labels = torch.stack(preds.labels) if isinstance(preds.labels, list) else preds.labels
        _scores = torch.stack(preds.scores) if isinstance(preds.scores, list) else preds.scores
        if hlabel_info.num_multilabel_classes > 0:
            preds_multiclass = _labels[:, : hlabel_info.num_multiclass_heads]
            preds_multilabel = _scores[:, hlabel_info.num_multiclass_heads :]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = _labels
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params: dict[str, Any] = {}
        export_params["input_size"] = (1, 3, 224, 224)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = True
        export_params["onnx_export_configuration"] = None
        export_params["mean"] = [123.675, 116.28, 103.53]
        export_params["std"] = [58.395, 57.12, 57.375]

        parameters = super()._export_parameters
        parameters.update(export_params)

        return parameters
