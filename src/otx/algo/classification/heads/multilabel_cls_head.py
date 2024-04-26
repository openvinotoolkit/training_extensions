# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""Multi-Label Classification Model Head Implementation.

The original source code is mmpretrain.models.heads.MultiLabelClsHead.
you can refer https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/heads/multi_label_cls_head.py

"""

from __future__ import annotations

import inspect
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional

from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.weight_init import constant_init, normal_init
from otx.core.data.entity.base import ImageInfo


class AnglularLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output cosine logits.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Init fuction of AngularLinear class."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data.normal_().renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward fuction of AngularLinear class."""
        cos_theta = functional.normalize(x.view(x.shape[0], -1), dim=1).mm(
            functional.normalize(self.weight.t(), p=2, dim=0),
        )
        return cos_theta.clamp(-1, 1)


class MultiLabelClsHead(BaseModule):
    """Multi-label Classification Head module.

    This module is responsible for calculating losses and performing inference
    for multi-label classification tasks. It takes features extracted from the
    backbone network and predicts the class labels.

    Args:
        BaseModule (class): The base module class.

    Attributes:
        scale (float): The scaling factor for the classification score.

    Methods:
        loss(feats, labels, **kwargs): Calculate losses from the classification score.
        get_valid_label_mask(img_metas): Get valid label mask using ignored_label.
        predict(feats, labels): Inference without augmentation.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        normalized: bool = False,
        scale: float = 1.0,
        thr: float | None = None,
        topk: int | None = None,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.normalized = normalized
        self.scale = scale
        self.loss_module = loss
        self.is_ignored_label_loss = "valid_label_mask" in inspect.getfullargspec(self.loss_module.forward).args

        if thr is None and topk is None:
            thr = 0.5

        self.thr = thr
        self.topk = topk

    def loss(self, feats: tuple[torch.Tensor], labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            labels (torch.Tensor): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        cls_score = self(feats) * self.scale
        imgs_info = kwargs.pop("imgs_info", None)
        if imgs_info is not None and self.is_ignored_label_loss:
            kwargs["valid_label_mask"] = self.get_valid_label_mask(imgs_info).to(cls_score.device)
        loss = self.loss_module(cls_score, labels, avg_factor=cls_score.size(0), **kwargs)
        return loss / self.scale

    def get_valid_label_mask(self, img_metas: list[ImageInfo]) -> torch.Tensor:
        """Get valid label mask using ignored_label.

        Args:
            img_metas (list[ImageInfo]): The metadata of the input images.

        Returns:
            torch.Tensor: The valid label mask.
        """
        valid_label_mask = []
        for meta in img_metas:
            mask = torch.Tensor([1 for _ in range(self.num_classes)])
            if meta.ignored_labels:
                mask[meta.ignored_labels] = 0
            valid_label_mask.append(mask)
        return torch.stack(valid_label_mask, dim=0)

    # ------------------------------------------------------------------------ #
    # Copy from mmpretrain.models.heads.MultiLabelClsHead
    # ------------------------------------------------------------------------ #

    def predict(self, feats: tuple[torch.Tensor]) -> torch.Tensor:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.

        Returns:
            torch.Tensor: Sigmoid results
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        return self._get_predictions(cls_score=cls_score)

    def _get_predictions(self, cls_score: torch.Tensor) -> torch.Tensor:
        """Get the score from the classification score."""
        return torch.sigmoid(cls_score)

    def pre_logits(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``MultiLabelLinearClsHead``, we just
        obtain the feature of the last stage.
        """
        # The obtain the MultiLabelLinearClsHead doesn't have other module,
        # just return after unpacking.
        if isinstance(feats, Sequence):
            return feats[-1]
        return feats


class MultiLabelLinearClsHead(MultiLabelClsHead):
    """Custom Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        normalized (bool): Normalize input features and weights.
        scale (float): positive scale parameter.
        loss (dict): Config of classification loss.
    """

    fc: nn.Module

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        normalized: bool = False,
        scale: float = 1.0,
        thr: float | None = None,
        topk: int | None = None,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss=loss,
            normalized=normalized,
            scale=scale,
            thr=thr,
            topk=topk,
            init_cfg=init_cfg,
            **kwargs,
        )

        self._init_layers()

    def _init_layers(self) -> None:
        if self.normalized:
            self.fc = AnglularLinear(self.in_channels, self.num_classes)
        else:
            self.fc = nn.Linear(self.in_channels, self.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights of head."""
        if isinstance(self.fc, nn.Linear):
            normal_init(self.fc, mean=0, std=0.01, bias=0)

    # ------------------------------------------------------------------------ #
    # Copy from mmpretrain.models.heads.MultiLabelLinearClsHead
    # ------------------------------------------------------------------------ #

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        return self.fc(pre_logits)


class MultiLabelNonLinearClsHead(MultiLabelClsHead):
    """Non-linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        hid_channels (int): Number of channels in the hidden feature map.
        act_cfg (dict | optional): The configuration of the activation function.
        scale (float): Positive scale parameter.
        loss (dict): Config of classification loss.
        dropout (bool): Whether use the dropout or not.
        normalized (bool): Normalize input features and weights in the last linar layer.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        hid_channels: int = 1280,
        activation_callable: Callable[[], nn.Module] = nn.ReLU,
        scale: float = 1.0,
        dropout: bool = False,
        normalized: bool = False,
        thr: float | None = None,
        topk: int | None = None,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss=loss,
            normalized=normalized,
            scale=scale,
            thr=thr,
            topk=topk,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.hid_channels = hid_channels
        self.dropout = dropout
        self.activation_callable = activation_callable

        if self.num_classes <= 0:
            msg = f"num_classes={num_classes} must be a positive integer"
            raise ValueError(msg)

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize the layers."""
        modules = [
            nn.Linear(self.in_channels, self.hid_channels),
            nn.BatchNorm1d(self.hid_channels),
            self.activation_callable if isinstance(self.activation_callable, nn.Module) else self.activation_callable(),
        ]
        if self.dropout:
            modules.append(nn.Dropout(p=0.2))
        if self.normalized:
            modules.append(AnglularLinear(self.hid_channels, self.num_classes))
        else:
            modules.append(nn.Linear(self.hid_channels, self.num_classes))

        self.classifier = nn.Sequential(*modules)
        self._init_weights()

    def _init_weights(self) -> None:
        """Iniitalize weights of model."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        return self.classifier(pre_logits)
