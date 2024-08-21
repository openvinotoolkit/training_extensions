# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""ImageClassifier Implementation.

The original source code is mmpretrain.models.classifiers.image.ImageClassifier.
you can refer https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/classifiers/image.py

"""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING

import torch

from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.classification.utils.ignored_labels import get_valid_label_mask
from otx.algo.explain.explain_algo import ReciproCAM
from otx.algo.modules.base_module import BaseModule

if TYPE_CHECKING:
    from torch import nn


class ImageClassifier(BaseModule):
    """Image classifiers for supervised classification task.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module | None): The neck module to process features from backbone.
        head (nn.Module): The head module to do prediction and calculate loss from processed features.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        loss (nn.Module): The loss module to calculate the loss.
        loss_scale (float, optional): The scaling factor for the loss. Defaults to 1.0.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        loss: nn.Module,
        loss_scale: float = 1.0,
        init_cfg: dict | list[dict] | None = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss_module = loss
        self.loss_scale = loss_scale
        self.is_ignored_label_loss = "valid_label_mask" in inspect.getfullargspec(self.loss_module.forward).args

        self.explainer = ReciproCAM(
            self._head_forward_fn,
            num_classes=head.num_classes,
            optimize_gap=isinstance(neck, GlobalAveragePooling),
        )

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        mode: str = "tensor",
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Performs forward pass of the model.

        Args:
            images (torch.Tensor): The input images.
            labels (torch.Tensor): The ground truth labels.
            mode (str, optional): The mode of the forward pass. Defaults to "tensor".

        Returns:
            torch.Tensor: The output logits or loss, depending on the training mode.
        """
        if mode == "tensor":
            return self.extract_feat(images, stage="head")
        if mode == "loss":
            return self.loss(images, labels, **kwargs)
        if mode == "predict":
            return self.predict(images)
        if mode == "explain":
            return self._forward_explain(images)

        msg = f'Invalid mode "{mode}".'
        raise RuntimeError(msg)

    def extract_feat(self, inputs: torch.Tensor, stage: str = "neck") -> torch.Tensor:
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
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
            torch.Tensor: The output of specified stage.
                In general, the output of pre_logits is a tensor.
        """
        x = self.backbone(inputs)

        if stage == "backbone":
            return x

        if self.neck is not None:
            x = self.neck(x)

        if stage == "neck":
            return x

        return self.head(x)

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            labels (torch.Tensor): The annotation data of
                every samples.

        Returns:
            torch.Tensor: loss components
        """
        cls_score = self.extract_feat(inputs, stage="head") * self.loss_scale
        imgs_info = kwargs.pop("imgs_info", None)
        if imgs_info is not None and self.is_ignored_label_loss:
            kwargs["valid_label_mask"] = get_valid_label_mask(imgs_info, self.head.num_classes).to(cls_score.device)
        return self.loss_module(cls_score, labels, **kwargs) / self.loss_scale

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, **kwargs)

    @torch.no_grad()
    def _forward_explain(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Generates explanations for the given images using the classifier.

        Args:
            images (torch.Tensor): Input images to generate explanations for.

        Returns:
            dict[str, torch.Tensor | list[torch.Tensor]]: A dictionary containing the following keys:
                - "logits" (torch.Tensor): The output logits from the classifier.
                - "preds" (torch.Tensor): The predicted class labels. Only included in non-tracing mode.
                - "scores" (torch.Tensor): The softmax scores for each class. Only included in non-tracing mode.
                - "saliency_map" (torch.Tensor): The saliency map generated by the explainer.
                - "feature_vector" (torch.Tensor): The feature vector extracted from the backbone network.
        """
        from otx.algo.explain.explain_algo import feature_vector_fn

        x = self.backbone(images)
        backbone_feat = x

        feature_vector = feature_vector_fn(backbone_feat)
        saliency_map = self.explainer.func(backbone_feat)

        if hasattr(self, "neck") and self.neck is not None:
            x = self.neck(x)

        logits = self.head(x)
        pred_results = self.head._get_predictions(logits)  # noqa: SLF001
        scores = pred_results.unbind(0)
        preds = pred_results.argmax(-1, keepdim=True).unbind(0)

        outputs = {
            "logits": logits,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

        if not torch.jit.is_tracing():
            outputs["scores"] = scores
            outputs["preds"] = preds

        return outputs

    @torch.no_grad()
    def _head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward."""
        if (neck := getattr(self, "neck", None)) is None:
            raise ValueError
        if (head := getattr(self, "head", None)) is None:
            raise ValueError

        output = neck(x)
        return head(output)
