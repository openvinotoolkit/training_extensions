# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

# mypy: disable-error-code="attr-defined"

"""Custom MoViNet Head for video recognition."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor, nn

from otx.algo.action_classification.utils.data_sample import ActionDataSample
from otx.algo.modules.base_module import BaseModule


class BaseHead(BaseModule):
    """Classification head for MoViNet.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        hidden_dim (int): Number of channels in hidden layer.
        tf_like (bool): If True, uses TensorFlow-style padding. Default: False.
        conv_type (str): Type of convolutional layer. Default: '3d'.
        loss_cls (nn.module): Loss class like CrossEntropyLoss.
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Standard deviation for initialization. Default: 0.1.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: nn.Module,
        topk: tuple[int, int] = (1, 5),
        average_clips: str | None = None,
    ):
        super().__init__()  # Call the initializer of BaseModule

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = loss_cls
        self.average_clips = average_clips

        if not isinstance(topk, (int, tuple)):
            msg = "`topk` should be an int or a tuple of ints"
            raise TypeError(msg)

        if any(_topk <= 0 for _topk in topk):
            msg = "Top-k should be larger than 0"
            raise ValueError(msg)

        self.topk = topk

    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call."""
        raise NotImplementedError

    def loss(
        self,
        feats: torch.Tensor | tuple[torch.Tensor],
        data_samples: list[ActionDataSample],
        **kwargs,
    ) -> dict:
        """Perform forward propagation of head and loss calculation on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: torch.Tensor, data_samples: list[ActionDataSample]) -> dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        label_list = [x.gt_label for x in data_samples]
        labels: torch.Tensor = torch.stack(label_list).to(cls_scores.device).squeeze()

        losses = {}
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = self._top_k_accuracy(
                cls_scores.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                self.topk,
            )
            for k, a in zip(self.topk, top_k_acc):
                losses[f"top{k}_acc"] = torch.tensor(a, device=cls_scores.device)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses["loss_cls"] = loss_cls
        return losses

    def predict(
        self,
        feats: torch.Tensor | tuple[torch.Tensor],
        data_samples: list[ActionDataSample],
        **kwargs,
    ) -> list[ActionDataSample]:
        """Perform forward propagation of head and predict recognition results on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: torch.Tensor, data_samples: list[ActionDataSample]) -> list[ActionDataSample]:
        """Transform a batch of output features extracted from the head into prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores, pred_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
        return data_samples

    def average_clip(self, cls_scores: torch.Tensor, num_segs: int = 1) -> torch.Tensor:
        """Averaging class scores over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_scores (torch.Tensor): Class scores to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class scores.
        """
        if self.average_clips not in ["score", "prob", None]:
            msg = f"{self.average_clips} is not supported. Currently supported ones are ['score', 'prob', None]"
            raise ValueError(msg)

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view((batch_size // num_segs, num_segs) + cls_scores.shape[1:])

        if self.average_clips is None:
            return cls_scores

        if self.average_clips == "prob":
            cls_scores = nn.functional.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == "score":
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores

    @staticmethod
    def _top_k_accuracy(scores: list[np.ndarray], labels: list[int], topk: tuple[int, int] = (1, 5)) -> list[float]:
        """Calculate top k accuracy score.

        Args:
            scores (list[np.ndarray]): Prediction scores for each class.
            labels (list[int]): Ground truth labels.
            topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

        Returns:
            list[float]: Top k accuracy score for each k.
        """
        res = []
        labels = np.array(labels)[:, np.newaxis]
        for k in topk:
            max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
            match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
            topk_acc_score = match_array.sum() / match_array.shape[0]
            res.append(topk_acc_score)

        return res
