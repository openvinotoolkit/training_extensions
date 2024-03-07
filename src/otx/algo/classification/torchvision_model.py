# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the OTX classification."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
from torchvision import models, tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MulticlassClsBatchPredEntityWithXAI,
)
from otx.core.model.entity.classification import OTXMulticlassClsModel

TV_WEIGHTS = {
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "efficientnet_b1": models.EfficientNet_B1_Weights.IMAGENET1K_V2,
    "efficientnet_b3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,  # Balanced
    "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
    "efficientnet_v2_l": models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,  # Accuracy
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,  # Speed
}


class TVModelWithLossComputation(nn.Module):
    """TorchVision Model with Loss Computation.

    This class represents a TorchVision model with loss computation for classification tasks.
    It takes a backbone model, number of classes, and an optional loss function as input.

    Args:
        backbone (
            Literal["resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b3",
            "efficientnet_b4", "efficientnet_v2_l", "mobilenet_v3_small"]):
            The backbone model to use for feature extraction.
        num_classes (int): The number of classes for the classification task.
        loss (nn.Module | None, optional): The loss function to use.

    Methods:
        forward(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            Performs forward pass of the model.

    """

    def __init__(
        self,
        backbone: Literal[
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_v2_l",
            "mobilenet_v3_small",
        ],
        num_classes: int,
        loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        net = getattr(models, backbone)(weights=TV_WEIGHTS[backbone])

        self.backbone = nn.Sequential(*list(net.children())[:-1])

        last_layer = list(net.children())[-1]
        classifier_len = len(list(last_layer.children()))
        if classifier_len >= 1:
            feature_channel = list(last_layer.children())[-1].in_features
            layers = list(last_layer.children())[:-1]
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
        if len(feats.shape) == 4:  # If feats is a 4D tensor: (batch_size, channels, height, width)
            feats = feats.view(feats.size(0), -1)  # Flatten the output of the backbone: (batch_size, features)
        logits = self.head(feats)
        if mode == "tensor":
            return logits
        if mode == "loss":
            return self.loss(logits, labels)
        return self.softmax(logits)


class OTXTVModel(OTXMulticlassClsModel):
    """OTXTVModel is that represents a TorchVision model for multiclass classification.

    Args:
        backbone (
            Literal["resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b3", "efficientnet_b4",
            "efficientnet_v2_l", "mobilenet_v3_small"]):
            The backbone architecture of the model.
        num_classes (int): The number of classes for classification.
        loss (nn.Module | None, optional): The loss function to be used. Defaults to None.
    """

    def __init__(
        self,
        backbone: Literal[
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_v2_l",
            "mobilenet_v3_small",
        ],
        num_classes: int,
        loss: nn.Module | None = None,
    ) -> None:
        self.backbone = backbone
        self.loss = loss

        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        return TVModelWithLossComputation(
            backbone=self.backbone,
            num_classes=self.num_classes,
            loss=self.loss,
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
        export_params["output_names"] = ["logits", "saliency_map"] if self.explain_mode else None
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["onnx_export_configuration"] = None
        export_params["mean"] = [0.485, 0.456, 0.406]
        export_params["std"] = [0.229, 0.224, 0.225]

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
        """Forward func of the ImageClassifier instance, which located in is in OTXModel().model."""
        x = self.backbone(images)
        backbone_feat = x

        saliency_map = self.explain_fn(backbone_feat)

        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        logits = self.head(x)
        if mode == "predict":
            logits = self.softmax(logits)

        return {
            "logits": logits,
            "saliency_map": saliency_map,
        }

    @torch.no_grad()
    def head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward. Can be redefined at the model's level."""
        if (head := getattr(self.model, "head", None)) is None:
            raise ValueError

        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        return head(x)
