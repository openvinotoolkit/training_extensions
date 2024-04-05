# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""ImageClassifier Implementation.

The original source code is mmpretrain.models.classifiers.image.ImageClassifier.
you can refer https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/classifiers/image.py

"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from mmengine.model.base_model import BaseModel

from otx.algo.hooks.recording_forward_hook import ReciproCAMHook
from otx.core.model.utils.mmpretrain import ClsDataPreprocessor

if TYPE_CHECKING:
    from torch import nn


class ImageClassifier(BaseModel):
    """Image classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.
            - probs (List[float], optional): The probability of every batch
              augmentation methods. If None, choose evenly. Defaults to None.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        pretrained: str | None = None,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        to_rgb: bool = False,
        init_cfg: dict | None = None,
    ):
        if pretrained is not None:
            init_cfg = {"type": "Pretrained", "checkpoint": pretrained}

        data_preprocessor = ClsDataPreprocessor(
            mean=[123.675, 116.28, 103.53] if mean is None else mean,
            std=[58.395, 57.12, 57.375] if std is None else std,
            to_rgb=to_rgb,
        )

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.explainer = ReciproCAMHook(
            self._head_forward_fn,
            num_classes=head.num_classes,
            optimize_gap=True,
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
            feats = self.extract_feat(images)
            return self.head(feats)
        if mode == "loss":
            return self.loss(images, labels, **kwargs)
        if mode == "predict":
            return self.predict(images)
        if mode == "explain":
            return self._forward_explain(images)

        msg = f'Invalid mode "{mode}".'
        raise RuntimeError(msg)

    def extract_feat(self, inputs: torch.Tensor, stage: str = "neck") -> tuple | torch.Tensor:
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
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple and the output of
            pre_logits is a tensor.
        """
        x = self.backbone(inputs)

        if stage == "backbone":
            return x

        x = self.neck(x)

        if stage == "neck":
            return x

        return self.head(x)

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            labels (torch.Tensor): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, labels, **kwargs)

    def predict(self, inputs: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, **kwargs)

    def _forward_explain(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Generates explanations for the given images using the classifier.

        Args:
            images (torch.Tensor): Input images to generate explanations for.

        Returns:
            dict[str, torch.Tensor | list[torch.Tensor]]: A dictionary containing the following keys:
                - "logits" (torch.Tensor): The output logits from the classifier.
                - "preds" (torch.Tensor): The predicted class labels.
                - "scores" (torch.Tensor): The softmax scores for each class.
                - "saliency_map" (torch.Tensor): The saliency map generated by the explainer.
                - "feature_vector" (torch.Tensor): The feature vector extracted from the backbone network.
        """
        x = self.backbone(images)
        backbone_feat = x

        from otx.algo.hooks.recording_forward_hook import get_feature_vector

        feature_vector = get_feature_vector(backbone_feat)
        saliency_map = self.explainer.func(backbone_feat)

        if hasattr(self, "neck") and self.neck is not None:
            x = self.neck(x)

        logits = self.head(x)
        pred_results = self.head._get_predictions(logits)  # noqa: SLF001
        # H-Label Classification Case
        if isinstance(pred_results, dict):
            scores = pred_results["pred_scores"]
            preds = pred_results["pred_labels"]
        else:
            scores = pred_results.unbind(0)
            preds = logits.argmax(-1, keepdim=True).unbind(0)

        return {
            "logits": logits,
            "preds": preds,
            "scores": scores,
            "saliency_map": saliency_map,
            "feature_vector": feature_vector,
        }

    @torch.no_grad()
    def _head_forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Performs model's neck and head forward."""
        if (neck := getattr(self, "neck", None)) is None:
            raise ValueError
        if (head := getattr(self, "head", None)) is None:
            raise ValueError

        output = neck(x)
        return head(output)
