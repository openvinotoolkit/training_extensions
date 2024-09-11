# Copyright (C) 2024 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of RTMCCHead."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from otx.algo.keypoint_detection.utils.keypoint_eval import simcc_pck_accuracy
from otx.algo.keypoint_detection.utils.rtmcc_block import RTMCCBlock
from otx.algo.keypoint_detection.utils.scale_norm import ScaleNorm
from otx.algo.keypoint_detection.utils.simcc_label import SimCCLabel
from otx.algo.modules.base_module import BaseModule
from torch import Tensor, nn

if TYPE_CHECKING:
    from otx.core.data.dataset.keypoint_detection import KeypointDetBatchDataEntity


class RTMCCHead(BaseModule):
    """Top-down head introduced in RTMPose (2023).

    The head is composed of a large-kernel convolutional layer,
    a fully-connected layer and a Gated Attention Unit to
    generate 1d representation from low-resolution feature maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [h, w].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        loss (nn.module): keypoint loss.
        decoder_cfg (dict): Config dict for the keypoint decoder.
        gau_cfg (dict): Config dict for the Gated Attention Unit.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
        self,
        in_channels: int | list[int],
        out_channels: int,
        input_size: tuple[int, int],
        in_featuremap_size: tuple[int, int],
        loss: nn.Module,
        decoder_cfg: dict,
        gau_cfg: dict,
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        init_cfg: dict | list[dict] | None = None,
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = loss
        self.codec = SimCCLabel(**decoder_cfg)

        if isinstance(in_channels, (tuple, list)):
            msg = f"{self.__class__.__name__} does not support selecting multiple input features."
            raise TypeError(msg)

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
        )
        self.mlp = nn.Sequential(ScaleNorm(flatten_dims), nn.Linear(flatten_dims, gau_cfg["in_token_dims"], bias=False))
        self.gau = RTMCCBlock(**gau_cfg)
        self.cls_x = nn.Linear(gau_cfg["out_token_dims"], int(self.input_size[1] * self.simcc_split_ratio), bias=False)
        self.cls_y = nn.Linear(gau_cfg["out_token_dims"], int(self.input_size[0] * self.simcc_split_ratio), bias=False)

    def forward(self, feats: tuple[Tensor]) -> tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]

        feats = self.final_layer(feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats = self.mlp(feats)  # -> B, K, hidden

        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y

    def predict(
        self,
        feats: tuple[Tensor],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage features

        Returns:
            List[tuple[np.ndarray, np.ndarray]]: The pose predictions and scores per each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
        """
        pred_x, pred_y = self(feats)
        batch_keypoints, batch_scores = self.codec.decode(
            simcc_x=self.to_numpy(pred_x),
            simcc_y=self.to_numpy(pred_y),
        )

        preds = []
        for keypoints, scores in zip(batch_keypoints, batch_scores):
            preds.append((keypoints, scores))
        return preds

    def loss(self, x: tuple[Tensor], entity: KeypointDetBatchDataEntity) -> dict:
        """Perform forward propagation and loss calculation of the detection head.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            entity (KeypointDetBatchDataEntity): Entity from OTX dataset.

        Returns:
            dict: A dictionary of loss components.
        """
        pred_x, pred_y = self(x)

        encoded = self.codec.encode(
            keypoints=np.stack(entity.keypoints),
            keypoints_visible=np.stack(entity.keypoints_visible),
        )

        device = pred_x.device
        gt_x = torch.tensor(encoded["keypoint_x_labels"], device=device)
        gt_y = torch.tensor(encoded["keypoint_y_labels"], device=device)
        keypoints_weight = torch.tensor(encoded["keypoint_weights"], device=device)

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses: dict = {}
        loss = self.loss_module(pred_simcc, gt_simcc, keypoints_weight)
        losses.update(loss_kpt=loss)

        mask = self.to_numpy(keypoints_weight)
        if isinstance(mask, np.ndarray):
            mask = mask > 0
        elif isinstance(mask, tuple):
            mask = mask[0] > 0

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=self.to_numpy(pred_simcc),
            target=self.to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=mask,
        )

        loss_pose = -1 * torch.tensor(avg_acc, device=device)
        losses.update(loss_pose=loss_pose)

        return losses

    def to_numpy(self, x: Tensor | tuple[Tensor, Tensor]) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Convert torch tensor to numpy.ndarray.

        Args:
            x (Tensor | Sequence[Tensor]): A single tensor or a sequence of
                tensors

        Returns:
            np.ndarray | tuple: return a tuple of converted numpy array(s)
        """
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, tuple) and all(isinstance(i, Tensor) for i in x):
            return tuple([self.to_numpy(i) for i in x])
        return np.array(x)

    @property
    def default_init_cfg(self) -> list[dict]:
        """Set a default initialization config."""
        return [
            {"type": "Normal", "layer": ["Conv2d"], "std": 0.001},
            {"type": "Constant", "layer": "BatchNorm2d", "val": 1},
            {"type": "Normal", "layer": ["Linear"], "std": 0.01, "bias": 0},
        ]
