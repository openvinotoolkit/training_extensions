# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torchvision.models import get_model, get_model_weights

from otx.algo.classification.heads import OTXSemiSLLinearClsHead
from otx.algo.explain.explain_algo import ReciproCAM, feature_vector_fn
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


class TVClassificationModel(nn.Module):
    """TorchVision Model with Loss Computation.

    This class represents a TorchVision model with loss computation for classification tasks.
    It takes a backbone model, number of classes, and an optional loss function as input.

    Args:
        backbone (TVModelType): The backbone model to use for feature extraction.
        num_classes (int): The number of classes for the classification task.
        loss (Callable | None, optional): The loss function to use.
        freeze_backbone (bool, optional): Whether to freeze the backbone model. Defaults to False.
        task (Literal["multiclass", "multilabel", "hlabel"], optional): The type of classification task.
        train_type (Literal["supervised", "semi_supervised"], optional): The type of training.

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
        task: Literal["multiclass", "multilabel", "hlabel"] = "multiclass",
        train_type: Literal["supervised", "semi_supervised"] = "supervised",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task = task
        self.train_type = train_type

        net = get_model(name=backbone, weights=get_model_weights(backbone))

        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.use_layer_norm_2d = False

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.softmax = nn.Softmax(dim=-1)
        self.loss_module = loss
        self.neck: nn.Module | None = None
        self.head = self._get_head(net)

        avgpool_index = 0
        for i, layer in enumerate(self.backbone.children()):
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                avgpool_index = i
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:avgpool_index])
        self.avgpool = nn.Sequential(
            *list(self.backbone.children())[avgpool_index:],
        )  # Avgpool and Dropout (if the model has it)

        self.explainer = ReciproCAM(
            self._head_forward_fn,
            num_classes=num_classes,
            optimize_gap=True,
        )

    def _get_head(self, net: nn.Module) -> nn.Module:
        """Returns the head of the model."""
        head: nn.Module
        last_layer = list(net.children())[-1]
        layers = []
        if self.task == "multiclass":
            classifier_len = len(list(last_layer.children()))
            if classifier_len >= 1:
                feature_channel = list(last_layer.children())[-1].in_features
                layers = list(last_layer.children())[:-1]
                self.use_layer_norm_2d = layers[0].__class__.__name__ == "LayerNorm2d"

                head = nn.Sequential(*layers, nn.Linear(feature_channel, self.num_classes))
            else:
                feature_channel = last_layer.in_features
                head = nn.Linear(feature_channel, self.num_classes)

            if self.train_type == "semi_supervised":
                self.neck = nn.Sequential(*layers) if layers else None
                head = OTXSemiSLLinearClsHead(
                    num_classes=self.num_classes,
                    in_channels=feature_channel,
                    loss=self.loss_module,
                )

        return head

    def forward(
        self,
        images: torch.Tensor | dict[str, torch.Tensor],
        labels: torch.Tensor | dict[str, torch.Tensor] | None = None,
        mode: str = "tensor",
        **kwargs,
    ) -> dict[str, tuple | torch.Tensor] | tuple | torch.Tensor:
        """Performs forward pass of the model.

        Args:
            images (torch.Tensor | dict[str, torch.Tensor]): The input images.
            labels (torch.Tensor | dict[str, torch.Tensor] | None, optional): The ground truth labels.
            mode (str, optional): The mode of the forward pass. Defaults to "tensor".

        Returns:
            torch.Tensor: The output logits or loss, depending on the training mode.
        """
        if mode == "tensor":
            return self.extract_feat(images, stage="head")
        if mode == "loss":
            feats = self.extract_feat(images, stage="neck")
            return self.loss(feats, labels)
        if mode == "explain":
            return self._forward_explain(images)
        logits = self.extract_feat(images, stage="head")
        return self.softmax(logits)

    def extract_feat(
        self,
        inputs: dict[str, torch.Tensor] | torch.Tensor,
        stage: str = "neck",
    ) -> dict[str, tuple | torch.Tensor] | tuple | torch.Tensor:
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (dict[str, torch.Tensor] | torch.Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.
                - "pre_logits": The feature before the final classification
                  linear layer. Usually returns a tensor.

                Defaults to "neck".

        Returns:
            dict[str, tuple | torch.Tensor] | tuple | torch.Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple.
        """
        if isinstance(inputs, dict):
            return self._extract_feat_with_unlabeled(inputs, stage)

        x = self._flatten_outputs(self.backbone(inputs))

        if stage == "backbone":
            return x

        if self.neck is not None:
            x = self.neck(x)

        if stage == "neck":
            return x

        return self.head(x)

    def _extract_feat_with_unlabeled(
        self,
        images: dict[str, torch.Tensor],
        stage: str = "neck",
    ) -> dict[str, torch.Tensor]:
        labeled_inputs = images["labeled"]
        unlabeled_weak_inputs = images["weak_transforms"]
        unlabeled_strong_inputs = images["strong_transforms"]

        x = {}
        x["labeled"] = self.extract_feat(labeled_inputs, stage)
        # For weak augmentation inputs, use no_grad to use as a pseudo-label.
        with torch.no_grad():
            x["unlabeled_weak"] = self.extract_feat(unlabeled_weak_inputs, stage)
        x["unlabeled_strong"] = self.extract_feat(unlabeled_strong_inputs, stage)
        return x

    def loss(
        self,
        inputs: dict[str, torch.Tensor] | tuple | torch.Tensor,
        labels: dict[str, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the loss of the model.

        Args:
            inputs (dict[str, torch.Tensor] | tuple | torch.Tensor): The outputs of the model backbone.
            labels (dict[str, torch.Tensor] | torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        if hasattr(self.head, "loss"):
            return self.head.loss(inputs, labels)
        logits = self.head(inputs)
        return self.loss_module(logits, labels)

    @torch.no_grad()
    def _forward_explain(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        backbone_feat = self.feature_extractor(images)

        saliency_map = self.explainer.func(backbone_feat)
        feature_vector = feature_vector_fn(backbone_feat)

        x = self._flatten_outputs(self.avgpool(backbone_feat))
        logits = self.head(x)

        outputs = {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

        if not torch.jit.is_tracing():
            outputs["scores"] = self.softmax(logits)
            outputs["preds"] = logits.argmax(-1, keepdim=False)

        return outputs

    @torch.no_grad()
    def _head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward."""
        x = self._flatten_outputs(self.avgpool(x))
        return self.head(x)

    def _flatten_outputs(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the output
        if len(x.shape) == 4 and not self.use_layer_norm_2d:  # If feats is a 4D tensor: (b, c, h, w)
            x = x.view(x.size(0), -1)  # Flatten the output of the backbone: (b, f)
        return x


class OTXTVModel(OTXMulticlassClsModel):
    """OTXTVModel is that represents a TorchVision model for multiclass classification.

    Args:
        backbone (TVModelType): The backbone architecture of the model.
        label_info (LabelInfoTypes): The number of classes for classification.
        optimizer (OptimizerCallable, optional): The optimizer to use for training.
            Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler to use.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The metric to use for evaluation. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        freeze_backbone (bool, optional): Whether to freeze the backbone model. Defaults to False.
        train_type (Literal["supervised", "semi_supervised"], optional): The type of training.
    """

    model: TVClassificationModel

    def __init__(
        self,
        backbone: TVModelType,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
        freeze_backbone: bool = False,
        train_type: Literal["supervised", "semi_supervised"] = "supervised",
    ) -> None:
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.train_type = train_type

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self) -> nn.Module:
        return TVClassificationModel(
            backbone=self.backbone,
            num_classes=self.num_classes,
            loss=nn.CrossEntropyLoss(reduction="none"),
            freeze_backbone=self.freeze_backbone,
            task="multiclass",
            train_type=self.train_type,
        )

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        if isinstance(inputs, dict):
            # When used with an unlabeled dataset, it comes in as a dict.
            images = {key: inputs[key].stacked_images for key in inputs}
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
        if self.train_type == "semi_supervised":
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
