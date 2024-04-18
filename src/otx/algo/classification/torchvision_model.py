# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
from torch import Tensor, nn
from torchvision.models import get_model, get_model_weights

from otx.algo.explain.explain_algo import ReciproCAM
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.classification import OTXMulticlassClsModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


TVModelType = Literal[
    "alexnet",
    "convnext_base",
    "convnext_large",
    "convnext_small",
    "convnext_tiny",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_l",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
    "googlenet",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "regnet_x_16gf",
    "regnet_x_1_6gf",
    "regnet_x_32gf",
    "regnet_x_3_2gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_8gf",
    "regnet_y_128gf",
    "regnet_y_16gf",
    "regnet_y_1_6gf",
    "regnet_y_32gf",
    "regnet_y_3_2gf",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_8gf",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "resnext50_32x4d",
    "swin_b",
    "swin_s",
    "swin_t",
    "swin_v2_b",
    "swin_v2_s",
    "swin_v2_t",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "wide_resnet101_2",
    "wide_resnet50_2",
]


class TVModelWithLossComputation(nn.Module):
    """TorchVision Model with Loss Computation.

    This class represents a TorchVision model with loss computation for classification tasks.
    It takes a backbone model, number of classes, and an optional loss function as input.

    Args:
        backbone (TVModelType): The backbone model to use for feature extraction.
        num_classes (int): The number of classes for the classification task.
        loss (Callable | None, optional): The loss function to use.
        freeze_backbone (bool, optional): Whether to freeze the backbone model. Defaults to False.

    Methods:
        forward(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            Performs forward pass of the model.

    """

    def __init__(
        self,
        backbone: TVModelType,
        num_classes: int,
        loss: nn.Module,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        net = get_model(name=backbone, weights=get_model_weights(backbone))

        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.use_layer_norm_2d = False

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        last_layer = list(net.children())[-1]
        classifier_len = len(list(last_layer.children()))
        if classifier_len >= 1:
            feature_channel = list(last_layer.children())[-1].in_features
            layers = list(last_layer.children())[:-1]
            self.use_layer_norm_2d = layers[0].__class__.__name__ == "LayerNorm2d"
            self.head = nn.Sequential(*layers, nn.Linear(feature_channel, num_classes))
        else:
            feature_channel = last_layer.in_features
            self.head = nn.Linear(feature_channel, num_classes)

        self.softmax = nn.Softmax(dim=-1)
        self.loss = loss

        self.explainer = ReciproCAM(
            self._head_forward_fn,
            num_classes=num_classes,
            optimize_gap=True,
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        mode: str = "tensor",
    ) -> torch.Tensor:
        """Performs forward pass of the model.

        Args:
            images (torch.Tensor): The input images.
            labels (torch.Tensor): The ground truth labels.
            mode (str, optional): The mode of the forward pass. Defaults to "tensor".

        Returns:
            torch.Tensor: The output logits or loss, depending on the training mode.
        """
        feats = self.backbone(images)
        if len(feats.shape) == 4 and not self.use_layer_norm_2d:  # If feats is a 4D tensor: (b, c, h, w)
            feats = feats.view(feats.size(0), -1)  # Flatten the output of the backbone: (b, f)
        logits = self.head(feats)
        if mode == "tensor":
            return logits
        if mode == "loss":
            return self.loss(logits, labels)
        if mode == "explain":
            return self._forward_explain(images)

        return self.softmax(logits)

    def _forward_explain(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        x = self.backbone(images)
        backbone_feat = x

        saliency_map = self.explainer.func(backbone_feat)

        if len(x.shape) == 4 and not self.use_layer_norm_2d:
            x = x.view(x.size(0), -1)

        feature_vector = x

        logits = self.head(x)

        return {
            "logits": logits,
            "preds": logits.argmax(-1, keepdim=False),
            "scores": self.softmax(logits),
            "saliency_map": saliency_map,
            "feature_vector": feature_vector,
        }

    @torch.no_grad()
    def _head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward."""
        if len(x.shape) == 4 and not self.use_layer_norm_2d:
            x = x.view(x.size(0), -1)
        return self.head(x)


class OTXTVModel(OTXMulticlassClsModel):
    """OTXTVModel is that represents a TorchVision model for multiclass classification.

    Args:
        backbone (TVModelType): The backbone architecture of the model.
        num_classes (int): The number of classes for classification.
        loss (Callable | None, optional): The loss function to be used. Defaults to None.
        freeze_backbone (bool, optional): Whether to freeze the backbone model. Defaults to False.
    """

    model: TVModelWithLossComputation

    def __init__(
        self,
        backbone: TVModelType,
        label_info: LabelInfoTypes,
        loss_callable: Callable[[], nn.Module] = nn.CrossEntropyLoss,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        self.backbone = backbone
        self.loss_callable = loss_callable
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        return TVModelWithLossComputation(
            backbone=self.backbone,
            num_classes=self.num_classes,
            loss=self.loss_callable(),
            freeze_backbone=self.freeze_backbone,
        )

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
