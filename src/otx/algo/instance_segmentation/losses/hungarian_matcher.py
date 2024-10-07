# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""HungarianMatcher to compute the matching cost and solve the corresponding LSAP."""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from otx.algo.instance_segmentation.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from otx.algo.instance_segmentation.utils.utils import point_sample


def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (Tensor): A tensor representing a mask
        labels (Tensor): A tensor with the same shape as inputs.
            Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        Tensor: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def pair_wise_sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    r"""A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (Tensor): A tensor representing a mask.
        labels (Tensor): A tensor with the same shape as inputs.
            Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (Tensor): The computed loss between each pairs.
    """
    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    return loss_pos + loss_neg


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    Args:
        cost_class (float, optional): label classification cost. Defaults to 1.
        cost_mask (float, optional): mask cost. Defaults to 1.
        cost_dice (float, optional): dice loss cost. Defaults to 1.
        num_points (int, optional): number of points to sample for mask loss. Defaults to 0.
        cost_box (float, optional): box cost. Defaults to 0.
        cost_giou (float, optional): giou cost. Defaults to 0.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        cost_box: float = 0,
        cost_giou: float = 0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """More memory-friendly matching. Change cost to compute only certain loss in matching."""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_bbox = outputs["pred_boxes"][b]
            tgt_bbox = targets[b]["boxes"]
            cost_bbox = torch.cdist(out_bbox.float(), tgt_bbox.float(), p=1).to(out_bbox)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)).to(out_bbox)

            out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]
            # focal loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"]

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask.to(out_bbox),
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(out_mask, tgt_mask)
            cost_dice = pair_wise_dice_loss(out_mask, tgt_mask)

            matching_cost = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_box * cost_bbox
                + self.cost_giou * cost_giou
            )
            matching_cost = torch.minimum(matching_cost, torch.tensor(1e10))
            matching_cost = torch.maximum(matching_cost, torch.tensor(-1e10))
            matching_cost = matching_cost.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(matching_cost))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Performs the matching.

        Args:
            outputs (dict[str, Tensor]): This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets (list[dict[str, Tensor]]): This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            objects in the target) containing the class labels.
                    "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor]]: The matched indices.
        """
        return self.memory_efficient_forward(outputs, targets)
