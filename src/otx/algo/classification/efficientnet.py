# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNet-B0 model implementation."""


from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from otx.algo.classification.backbones.efficientnet import EFFICIENTNET_VERSION, OTXEfficientNet
from otx.algo.classification.classifier.base_classifier import ImageClassifier
from otx.algo.classification.classifier.semi_sl_classifier import SemiSLClassifier
from otx.algo.classification.heads import (
    HierarchicalLinearClsHead,
    LinearClsHead,
    MultiLabelLinearClsHead,
    OTXSemiSLLinearClsHead,
)
from otx.algo.classification.losses.asymmetric_angular_loss_with_ignore import AsymmetricAngularLossWithIgnore
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.classification.utils import get_classification_layers
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
from otx.core.types.label import HLabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class EfficientNetForMulticlassCls(OTXMulticlassClsModel):
    """EfficientNet Model for multi-class classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        version: EFFICIENTNET_VERSION = "b0",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.version = version

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(num_classes=self.num_classes)
        model.init_weights()
        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        return ImageClassifier(
            backbone=OTXEfficientNet(version=self.version, pretrained=True),
            neck=GlobalAveragePooling(dim=2),
            head=LinearClsHead(
                num_classes=num_classes,
                in_channels=1280,
                topk=(1, 5) if num_classes >= 5 else (1,),
                loss=nn.CrossEntropyLoss(reduction="none"),
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)

    def _reset_prediction_layer(self, num_classes: int) -> None:
        return

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
            "imgs_info": inputs.imgs_info,
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
            images=inputs.stacked_images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

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

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")


class EfficientNetForMulticlassClsSemiSL(EfficientNetForMulticlassCls):
    """EfficientNet model for multiclass classification with semi-supervised learning.

    This class extends the `EfficientNetForMulticlassCls` class and adds support for semi-supervised learning.
    It overrides the `_build_model` and `_customize_inputs` methods to incorporate the semi-supervised learning.

    Args:
        EfficientNetForMulticlassCls (class): The base class for EfficientNet multiclass classification.

    Attributes:
        version (str): The version of the EfficientNet model.
    """

    def _build_model(self, num_classes: int) -> nn.Module:
        return SemiSLClassifier(
            backbone=OTXEfficientNet(version=self.version, pretrained=True),
            neck=GlobalAveragePooling(dim=2),
            head=OTXSemiSLLinearClsHead(
                num_classes=num_classes,
                in_channels=1280,
                loss=nn.CrossEntropyLoss(reduction="none"),
            ),
        )

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        """Customizes the input data for the model based on the current mode.

        Args:
            inputs (MulticlassClsBatchDataEntity): The input batch of data.

        Returns:
            dict[str, Any]: The customized input data.
        """
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        if isinstance(inputs, dict):
            # When used with an unlabeled dataset, it comes in as a dict.
            images = {key: inputs[key].images for key in inputs}
            labels = {key: torch.cat(inputs[key].labels, dim=0) for key in inputs}
            imgs_info = {key: inputs[key].imgs_info for key in inputs}
            return {
                "images": images,
                "labels": labels,
                "imgs_info": imgs_info,
                "mode": mode,
            }
        return {
            "images": inputs.stacked_images,
            "labels": torch.cat(inputs.labels, dim=0),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def training_step(self, batch: MulticlassClsBatchDataEntity, batch_idx: int) -> Tensor:
        """Performs a single training step on a batch of data.

        Args:
            batch (MulticlassClsBatchDataEntity): The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            Tensor: The computed loss for the training step.
        """
        loss = super().training_step(batch, batch_idx)
        # Collect metrics related to Semi-SL Training.
        self.log(
            "train/unlabeled_coef",
            self.model.head.unlabeled_coef,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/num_pseudo_label",
            self.model.head.num_pseudo_label,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss


class EfficientNetForMultilabelCls(OTXMultilabelClsModel):
    """EfficientNet Model for multi-label classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        version: EFFICIENTNET_VERSION = "b0",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.version = version

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(num_classes=self.num_classes)
        model.init_weights()
        return model

    def _build_model(self, num_classes: int) -> nn.Module:
        return ImageClassifier(
            backbone=OTXEfficientNet(version=self.version, pretrained=True),
            neck=GlobalAveragePooling(dim=2),
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=1280,
                scale=7.0,
                normalized=True,
                loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multilabel", add_prefix)

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

        return MultilabelClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=logits.argmax(-1, keepdim=True).unbind(0),
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, 224, 224),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_explain(self, inputs: MultilabelClsBatchDataEntity) -> MultilabelClsBatchPredEntity:
        """Model forward explain function."""
        outputs = self.model(images=inputs.stacked_images, mode="explain")

        return MultilabelClsBatchPredEntity(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["preds"],
            scores=outputs["scores"],
            saliency_map=outputs["saliency_map"],
            feature_vector=outputs["feature_vector"],
        )

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")


class EfficientNetForHLabelCls(OTXHlabelClsModel):
    """EfficientNetB0 Model for hierarchical label classification task."""

    label_info: HLabelInfo

    def __init__(
        self,
        label_info: HLabelInfo,
        version: EFFICIENTNET_VERSION = "b0",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallble,
        torch_compile: bool = False,
    ) -> None:
        self.version = version

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        # Get classification_layers for class-incr learning
        sample_config = deepcopy(self.label_info.as_head_config_dict())
        sample_config["num_classes"] = 5
        sample_model_dict = self._build_model(head_config=sample_config).state_dict()
        sample_config["num_classes"] = 6
        incremental_model_dict = self._build_model(head_config=sample_config).state_dict()
        self.classification_layers = get_classification_layers(
            sample_model_dict,
            incremental_model_dict,
            prefix="model.",
        )

        model = self._build_model(head_config=self.label_info.as_head_config_dict())
        model.init_weights()
        return model

    def _build_model(self, head_config: dict) -> nn.Module:
        if not isinstance(self.label_info, HLabelInfo):
            raise TypeError(self.label_info)

        return ImageClassifier(
            backbone=OTXEfficientNet(version=self.version, pretrained=True),
            neck=GlobalAveragePooling(dim=2),
            head=HierarchicalLinearClsHead(
                in_channels=1280,
                multiclass_loss=nn.CrossEntropyLoss(),
                multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
                **head_config,
            ),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "hlabel", add_prefix)

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
        if isinstance(outputs, dict):
            scores = outputs["scores"]
            labels = outputs["labels"]
        else:
            scores = outputs
            labels = outputs.argmax(-1, keepdim=True)

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
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, 224, 224),
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_explain(self, inputs: HlabelClsBatchDataEntity) -> HlabelClsBatchPredEntity:
        """Model forward explain function."""
        outputs = self.model(images=inputs.stacked_images, mode="explain")

        return HlabelClsBatchPredEntity(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["preds"],
            scores=outputs["scores"],
            saliency_map=outputs["saliency_map"],
            feature_vector=outputs["feature_vector"],
        )

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
