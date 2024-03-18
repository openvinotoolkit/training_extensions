# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the OTX classification."""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.models import get_model, get_model_weights

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MulticlassClsBatchPredEntityWithXAI,
)
from otx.core.model.entity.classification import OTXMulticlassClsModel

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
        loss: Callable | None = None,
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
        self.loss = nn.CrossEntropyLoss() if loss is None else loss

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
        return self.softmax(logits)


class OTXTVModel(OTXMulticlassClsModel):
    """OTXTVModel is that represents a TorchVision model for multiclass classification.

    Args:
        backbone (TVModelType): The backbone architecture of the model.
        num_classes (int): The number of classes for classification.
        loss (Callable | None, optional): The loss function to be used. Defaults to None.
        freeze_backbone (bool, optional): Whether to freeze the backbone model. Defaults to False.
    """

    def __init__(
        self,
        backbone: TVModelType,
        num_classes: int,
        loss: Callable | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        self.backbone = backbone
        self.loss = loss
        self.freeze_backbone = freeze_backbone

        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        return TVModelWithLossComputation(
            backbone=self.backbone,
            num_classes=self.num_classes,
            loss=self.loss,
            freeze_backbone=self.freeze_backbone,
        )

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        if isinstance(inputs.images, list):
            images = tv_tensors.wrap(torch.stack(inputs.images, dim=0), like=inputs.images[0])
        else:
            images = inputs.images
        return {
            "images": images,
            "labels": torch.cat(inputs.labels, dim=0),
            "mode": "loss" if self.training else "predict",
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | MulticlassClsBatchPredEntityWithXAI | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)
        preds = logits.argmax(-1, keepdim=True).unbind(0)

        if self.explain_mode:
            if not isinstance(outputs, dict) or "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_maps = outputs["saliency_map"].detach().cpu().numpy()

            return MulticlassClsBatchPredEntityWithXAI(
                batch_size=len(preds),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                labels=preds,
                saliency_maps=list(saliency_maps),
                feature_vectors=[],
            )

        return MulticlassClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params: dict[str, Any] = {}
        export_params["input_size"] = (1, 3, 224, 224)
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["onnx_export_configuration"] = None
        export_params["mean"] = [123.675, 116.28, 103.53]
        export_params["std"] = [58.395, 57.12, 57.375]

        parameters = super()._export_parameters
        parameters.update(export_params)
        return parameters

    @staticmethod
    def _forward_explain_image_classifier(
        self: TVModelWithLossComputation,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,  # noqa: ARG004
        mode: str = "tensor",
    ) -> dict:
        """Forward func of the TVModelWithLossComputation instance."""
        x = self.backbone(images)
        backbone_feat = x

        saliency_map = self.explain_fn(backbone_feat)

        if len(x.shape) == 4 and not self.use_layer_norm_2d:
            x = x.view(x.size(0), -1)

        feature_vector = x
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.unsqueeze(0)

        logits = self.head(x)
        if mode == "predict":
            logits = self.softmax(logits)

        return {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward. Can be redefined at the model's level."""
        if (head := getattr(self.model, "head", None)) is None:
            raise ValueError

        if len(x.shape) == 4 and not self.model.use_layer_norm_2d:
            x = x.view(x.size(0), -1)
        return head(x)
