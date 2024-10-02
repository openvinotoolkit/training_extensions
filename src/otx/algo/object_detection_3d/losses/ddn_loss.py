# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ddn loss for MonoDETR model."""
from __future__ import annotations

import math

import torch
from torch import nn

from otx.algo.common.losses.focal_loss import FocalLoss


def compute_fg_mask(
    gt_boxes2d: torch.Tensor,
    shape: tuple[int, int],
    num_gt_per_img: int,
    downsample_factor: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute foreground mask for images.

    Args:
        gt_boxes2d [torch.Tensor(B, N, 4)]: 2D box labels
        shape [Tuple[int, int]]: Foreground mask desired shape
        downsample_factor [int]: Downsample factor for image
        device [torch.device]: Foreground mask desired device

    Returns:
        fg_mask [torch.Tensor(shape)]: Foreground mask
    """
    if device is None:
        device = torch.device("cpu")
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
    gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
    b = len(gt_boxes2d)
    for i in range(b):
        for n in range(gt_boxes2d[i].shape[0]):
            u1, v1, u2, v2 = gt_boxes2d[i][n]
            fg_mask[i, v1:v2, u1:u2] = True

    return fg_mask


class Balancer(nn.Module):
    """Fixed foreground/background loss balancer."""

    def __init__(self, fg_weight: float, bg_weight: float, downsample_factor: int = 1):
        """Initialize fixed foreground/background loss balancer.

        Args:
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def forward(
        self,
        loss: torch.Tensor,
        gt_boxes2d: torch.Tensor,
        num_gt_per_img: int,
    ) -> tuple[torch.Tensor, dict[float, float]]:
        """Forward pass.

        Args:
            loss [torch.Tensor(B, H, W)]: Pixel-wise loss
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing

        Returns:
            loss [torch.Tensor(1)]: Total loss after foreground/background balancing
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        # Compute masks
        fg_mask = compute_fg_mask(
            gt_boxes2d=gt_boxes2d,
            shape=loss.shape,
            num_gt_per_img=num_gt_per_img,
            downsample_factor=self.downsample_factor,
            device=loss.device,
        )
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        # return total loss
        return fg_loss + bg_loss


class DDNLoss(nn.Module):
    """DDNLoss module for computing the loss for MonoDETR model."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        fg_weight: float = 13,
        bg_weight: float = 1,
        downsample_factor: int = 1,
    ) -> None:
        """Initializes DDNLoss module.

        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.balancer = Balancer(downsample_factor=downsample_factor, fg_weight=fg_weight, bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

    def build_target_depth_from_3dcenter(
        self,
        depth_logits: torch.Tensor,
        gt_boxes2d: torch.Tensor,
        gt_center_depth: torch.Tensor,
        num_gt_per_img: int,
    ) -> torch.Tensor:
        """Builds target depth map from 3D center depth.

        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            gt_center_depth [torch.Tensor(B, N)]: 3D center depth
            num_gt_per_img: [int]: Number of ground truth boxes per image
        """
        b, _, h, w = depth_logits.shape
        depth_maps = torch.zeros((b, h, w), device=depth_logits.device, dtype=depth_logits.dtype)

        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.long()

        # Set all values within each box to True
        gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        b = len(gt_boxes2d)
        for i in range(b):
            center_depth_per_batch = gt_center_depth[i]
            center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d[i][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                depth_maps[i, v1:v2, u1:u2] = center_depth_per_batch[n]

        return depth_maps

    def bin_depths(
        self,
        depth_map: torch.Tensor,
        mode: str = "LID",
        depth_min: float = 1e-3,
        depth_max: float = 60,
        num_bins: int = 80,
        target: bool = False,
    ) -> torch.Tensor:
        """Converts depth map into bin indices.

        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison

        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = (depth_map - depth_min) / bin_size
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = (
                num_bins
                * (torch.log(1 + depth_map) - math.log(1 + depth_min))
                / (math.log(1 + depth_max) - math.log(1 + depth_min))
            )
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)

        return indices

    def forward(
        self,
        depth_logits: torch.Tensor,
        gt_boxes2d: torch.Tensor,
        num_gt_per_img: int,
        gt_center_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Gets depth_map loss.

        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img: [int]: Number of ground truth boxes per image
            gt_center_depth: [torch.Tensor(B, N)]: 3D center depth

        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """
        # Bin depth map to create target
        depth_maps = self.build_target_depth_from_3dcenter(depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img)
        depth_target = self.bin_depths(depth_maps, target=True)
        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)
        # Compute foreground/background balancing

        return self.balancer(loss=loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)
