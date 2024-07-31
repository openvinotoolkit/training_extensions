# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for defining h-label linear classification head."""

from __future__ import annotations

import inspect
from typing import Callable, Sequence

import torch
from torch import nn

from otx.algo.modules.base_module import BaseModule
from otx.algo.utils.weight_init import constant_init, normal_init
from otx.core.data.entity.base import ImageInfo


class HierarchicalClsHead(BaseModule):
    """The classification head for hierarchical classification.

    This class defines the methods for pre-processing the features,
    calculating the loss, and making predictions for hierarchical classification.
    """

    def __init__(
        self,
        num_multiclass_heads: int,
        num_multilabel_classes: int,
        head_idx_to_logits_range: dict[str, tuple[int, int]],
        num_single_label_classes: int,
        empty_multiclass_head_indices: list[int],
        in_channels: int,
        num_classes: int,
        multiclass_loss: nn.Module,
        multilabel_loss: nn.Module | None = None,
        thr: float = 0.5,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_multiclass_heads = num_multiclass_heads
        self.num_multilabel_classes = num_multilabel_classes
        self.head_idx_to_logits_range = head_idx_to_logits_range
        self.num_single_label_classes = num_single_label_classes
        self.empty_multiclass_head_indices = empty_multiclass_head_indices
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.thr = thr

        if self.num_multiclass_heads == 0:
            msg = "num_multiclass_head should be larger than 0"
            raise ValueError(msg)

        self.multiclass_loss = multiclass_loss
        self.multilabel_loss = None
        self.is_ignored_label_loss = False
        if num_multilabel_classes > 0 and multilabel_loss is not None:
            self.multilabel_loss = multilabel_loss
            self.is_ignored_label_loss = "valid_label_mask" in inspect.getfullargspec(self.multilabel_loss.forward).args

    def pre_logits(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The process before the final classification head."""
        if isinstance(feats, Sequence):
            return feats[-1]
        return feats

    def _get_head_idx_to_logits_range(self, idx: int) -> tuple[int, int]:
        """Get head_idx_to_logits_range information from hlabel information."""
        return (
            self.head_idx_to_logits_range[str(idx)][0],
            self.head_idx_to_logits_range[str(idx)][1],
        )

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
        cls_scores = self(feats)

        loss_score = torch.tensor(0.0, device=cls_scores.device)

        # Multiclass loss
        num_effective_heads_in_batch = 0  # consider the label removal case
        for i in range(self.num_multiclass_heads):
            if i not in self.empty_multiclass_head_indices:
                head_gt = labels[:, i]
                logit_range = self._get_head_idx_to_logits_range(i)
                head_logits = cls_scores[:, logit_range[0] : logit_range[1]]
                valid_mask = head_gt >= 0

                head_gt = head_gt[valid_mask]
                if len(head_gt) > 0:
                    head_logits = head_logits[valid_mask, :]
                    loss_score += self.multiclass_loss(head_logits, head_gt)
                    num_effective_heads_in_batch += 1

        if num_effective_heads_in_batch > 0:
            loss_score /= num_effective_heads_in_batch

        # Multilabel loss
        if self.num_multilabel_classes > 0:
            head_gt = labels[:, self.num_multiclass_heads :]
            head_logits = cls_scores[:, self.num_single_label_classes :]
            valid_mask = head_gt > 0
            head_gt = head_gt[valid_mask]
            if len(head_gt) > 0 and self.multilabel_loss is not None:
                head_logits = head_logits[valid_mask]
                imgs_info = kwargs.pop("imgs_info", None)
                if imgs_info is not None and self.is_ignored_label_loss:
                    valid_label_mask = self.get_valid_label_mask(imgs_info).to(head_logits.device)
                    valid_label_mask = valid_label_mask[:, self.num_single_label_classes :]
                    valid_label_mask = valid_label_mask[valid_mask]
                    kwargs["valid_label_mask"] = valid_label_mask
                loss_score += self.multilabel_loss(head_logits, head_gt, **kwargs)

        return loss_score

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

    def predict(
        self,
        feats: tuple[torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        cls_scores = self(feats)
        return self._get_predictions(cls_scores)

    def _get_predictions(
        self,
        cls_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        # Multiclass
        multiclass_pred_scores: list | torch.Tensor = []
        multiclass_pred_labels: list | torch.Tensor = []
        for i in range(self.num_multiclass_heads):
            logit_range = self._get_head_idx_to_logits_range(i)
            multiclass_logit = cls_scores[:, logit_range[0] : logit_range[1]]
            multiclass_pred = torch.softmax(multiclass_logit, dim=1)
            multiclass_pred_score, multiclass_pred_label = torch.max(multiclass_pred, dim=1)

            multiclass_pred_scores.append(multiclass_pred_score.view(-1, 1))
            multiclass_pred_labels.append(multiclass_pred_label.view(-1, 1))

        multiclass_pred_scores = torch.cat(multiclass_pred_scores, dim=1)
        multiclass_pred_labels = torch.cat(multiclass_pred_labels, dim=1)

        if self.num_multilabel_classes > 0:
            multilabel_logits = cls_scores[:, self.num_single_label_classes :]

            multilabel_pred = torch.sigmoid(multilabel_logits)
            multilabel_pred_labels = (multilabel_pred >= self.thr).int()

            pred_scores = torch.cat([multiclass_pred_scores, multilabel_pred], axis=1)
            pred_labels = torch.cat([multiclass_pred_labels, multilabel_pred_labels], axis=1)
        else:
            pred_scores = multiclass_pred_scores
            pred_labels = multiclass_pred_labels

        return {
            "scores": pred_scores,
            "labels": pred_labels,
        }


class HierarchicalLinearClsHead(HierarchicalClsHead):
    """Custom classification linear head for hierarchical classification task.

    Args:
        num_multiclass_heads (int): Number of multi-class heads.
        num_multilabel_classes (int): Number of multi-label classes.
        head_idx_to_logits_range: the logit range of each heads
        num_single_label_classes: the number of single label classes
        empty_multiclass_head_indices: the index of head that doesn't include any label
            due to the label removing
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of total classes.
        multiclass_loss (dict | None): Config of multi-class loss.
        multilabel_loss (dict | None): Config of multi-label loss.
        thr (float | None): Predictions with scores under the thresholds are considered
                            as negative. Defaults to 0.5.
    """

    def __init__(
        self,
        num_multiclass_heads: int,
        num_multilabel_classes: int,
        head_idx_to_logits_range: dict[str, tuple[int, int]],
        num_single_label_classes: int,
        empty_multiclass_head_indices: list[int],
        in_channels: int,
        num_classes: int,
        multiclass_loss: nn.Module,
        multilabel_loss: nn.Module | None = None,
        thr: float = 0.5,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            num_multiclass_heads=num_multiclass_heads,
            num_multilabel_classes=num_multilabel_classes,
            head_idx_to_logits_range=head_idx_to_logits_range,
            num_single_label_classes=num_single_label_classes,
            empty_multiclass_head_indices=empty_multiclass_head_indices,
            in_channels=in_channels,
            num_classes=num_classes,
            multiclass_loss=multiclass_loss,
            multilabel_loss=multilabel_loss,
            thr=thr,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize weights of the layers."""
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        return self.fc(pre_logits)


class HierarchicalNonLinearClsHead(HierarchicalClsHead):
    """Custom classification non-linear head for hierarchical classification task.

    Args:
        num_multiclass_heads (int): Number of multi-class heads.
        num_multilabel_classes (int): Number of multi-label classes.
        head_idx_to_logits_range: the logit range of each heads
        num_single_label_classes: the number of single label classes
        empty_multiclass_head_indices: the index of head that doesn't include any label
            due to the label removing
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of total classes.
        multiclass_loss (dict | None): Config of multi-class loss.
        multilabel_loss (dict | None): Config of multi-label loss.
        thr (float | None): Predictions with scores under the thresholds are considered
                            as negative. Defaults to 0.5.
        hid_cahnnels (int): Number of channels in the hidden feature map at the classifier.
        acivation_Cfg (dict | None): Config of activation layer at the classifier.
        dropout (bool): Flag for the enabling the dropout at the classifier.

    """

    def __init__(
        self,
        num_multiclass_heads: int,
        num_multilabel_classes: int,
        head_idx_to_logits_range: dict[str, tuple[int, int]],
        num_single_label_classes: int,
        empty_multiclass_head_indices: list[int],
        in_channels: int,
        num_classes: int,
        multiclass_loss: nn.Module,
        multilabel_loss: nn.Module | None = None,
        thr: float = 0.5,
        hid_channels: int = 1280,
        activation_callable: Callable[[], nn.Module] = nn.ReLU,
        dropout: bool = False,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            num_multiclass_heads=num_multiclass_heads,
            num_multilabel_classes=num_multilabel_classes,
            head_idx_to_logits_range=head_idx_to_logits_range,
            num_single_label_classes=num_single_label_classes,
            empty_multiclass_head_indices=empty_multiclass_head_indices,
            in_channels=in_channels,
            num_classes=num_classes,
            multiclass_loss=multiclass_loss,
            multilabel_loss=multilabel_loss,
            thr=thr,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.hid_channels = hid_channels
        self.dropout = dropout

        self.activation_callable = activation_callable

        classifier_modules = [
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            self.activation_callable if isinstance(self.activation_callable, nn.Module) else self.activation_callable(),
        ]

        if self.dropout:
            classifier_modules.append(nn.Dropout(p=0.2))

        classifier_modules.append(nn.Linear(hid_channels, num_classes))

        self.classifier = nn.Sequential(*classifier_modules)

        self._init_layers()

    def _init_layers(self) -> None:
        """Iniitialize weights of classification head."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        return self.classifier(pre_logits)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(torch.relu(self.fc1(torch.mean(x, dim=2, keepdim=True).mean(dim=3, keepdim=True))))
        max_out = self.fc2(torch.relu(self.fc1(torch.max(x, dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0])))
        return torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class HierarchicalCBAMClsHead(HierarchicalClsHead):
    """Custom classification CBAM head for hierarchical classification task.

    Args:
        num_multiclass_heads (int): Number of multi-class heads.
        num_multilabel_classes (int): Number of multi-label classes.
        head_idx_to_logits_range: the logit range of each heads
        num_single_label_classes: the number of single label classes
        empty_multiclass_head_indices: the index of head that doesn't include any label
            due to the label removing
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of total classes.
        multiclass_loss (dict | None): Config of multi-class loss.
        multilabel_loss (dict | None): Config of multi-label loss.
        thr (float | None): Predictions with scores under the thresholds are considered
                            as negative. Defaults to 0.5.
        hid_cahnnels (int): Number of channels in the hidden feature map at the classifier.
        acivation_Cfg (dict | None): Config of activation layer at the classifier.
        dropout (bool): Flag for the enabling the dropout at the classifier.

    """

    def __init__(
        self,
        num_multiclass_heads: int,
        num_multilabel_classes: int,
        head_idx_to_logits_range: dict[str, tuple[int, int]],
        num_single_label_classes: int,
        empty_multiclass_head_indices: list[int],
        in_channels: int,
        num_classes: int,
        multiclass_loss: nn.Module,
        multilabel_loss: nn.Module | None = None,
        thr: float = 0.5,
        hid_channels: int = 1280,
        activation_callable: Callable[[], nn.Module] = nn.ReLU,
        dropout: bool = False,
        init_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            num_multiclass_heads=num_multiclass_heads,
            num_multilabel_classes=num_multilabel_classes,
            head_idx_to_logits_range=head_idx_to_logits_range,
            num_single_label_classes=num_single_label_classes,
            empty_multiclass_head_indices=empty_multiclass_head_indices,
            in_channels=in_channels,
            num_classes=num_classes,
            multiclass_loss=multiclass_loss,
            multilabel_loss=multilabel_loss,
            thr=thr,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.hid_channels = hid_channels
        self.dropout = dropout

        self.activation_callable = activation_callable

        self.fc_superclass = nn.Linear(in_channels, num_multiclass_heads)
        self.attention_fc = nn.Linear(num_multiclass_heads, in_channels)
        self.cbam = CBAM(in_channels)

        classifier_modules = [
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            self.activation_callable if isinstance(self.activation_callable, nn.Module) else self.activation_callable(),
        ]

        if self.dropout:
            classifier_modules.append(nn.Dropout(p=0.2))

        classifier_modules.append(nn.Linear(hid_channels, num_classes))

        self.fc_subclass = nn.Sequential(*classifier_modules)

        self._init_layers()

    def _init_layers(self) -> None:
        """Iniitialize weights of classification head."""
        normal_init(self.fc_superclass, mean=0, std=0.01, bias=0)
        for module in self.fc_subclass:
            if isinstance(module, nn.Linear):
                normal_init(module, mean=0, std=0.01, bias=0)
            elif isinstance(module, nn.BatchNorm1d):
                constant_init(module, 1)

    def forward(self, feats: tuple[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        out_superclass = self.fc_superclass(pre_logits)

        attention_weights = torch.sigmoid(self.attention_fc(out_superclass))
        attended_features = pre_logits * attention_weights

        attended_features = attended_features.view(pre_logits.size(0), self.in_channels, 1, 1)
        attended_features = self.cbam(attended_features)
        attended_features = attended_features.view(pre_logits.size(0), self.in_channels)
        out_subclass = self.fc_subclass(attended_features)

        return out_subclass
