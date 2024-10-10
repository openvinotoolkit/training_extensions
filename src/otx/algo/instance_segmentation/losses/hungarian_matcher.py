# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Modules to compute the matching cost and solve the corresponding LSAP."""
from __future__ import annotations

import torch
import torch.nn.functional as f
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.cuda.amp import autocast

from otx.algo.instance_segmentation.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from otx.algo.instance_segmentation.utils.utils import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss,
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Batch sigmoid cross entropy loss.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = f.binary_cross_entropy_with_logits(
        inputs,
        torch.ones_like(inputs),
        reduction="none",
    )
    neg = f.binary_cross_entropy_with_logits(
        inputs,
        torch.zeros_like(inputs),
        reduction="none",
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm",
        neg,
        (1 - targets),
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss,
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network."""

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
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

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
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # If there's no annotations
                if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            matching_cost = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_box * cost_bbox
                + self.cost_giou * cost_giou
            )
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
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)
