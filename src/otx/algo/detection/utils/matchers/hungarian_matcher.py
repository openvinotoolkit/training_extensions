# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Modules to compute the matching cost and solve the corresponding LSAP. Modified from https://github.com/lyuwenyu/RT-DETR."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import box_convert

from otx.algo.common.utils.bbox_overlaps import bbox_overlaps


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_dict: dict[str, float | int], alpha: float = 0.25, gamma: float = 2.0):
        """Creates the matcher.

        Args:
            weight_dict (dict[str, float | int]): A dictionary containing the weights for different costs.
                The dictionary may have the following keys:
                - "cost_class" (float | int): The weight for the class cost.
                - "cost_bbox" (float | int): The weight for the bounding box cost.
                - "cost_giou" (float | int): The weight for the generalized intersection over union (IoU) cost.
            alpha (float, optional): The alpha parameter for the cost computation. Defaults to 0.25.
            gamma (float, optional): The gamma parameter for the cost computation. Defaults to 2.0.
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Performs the matching.

        Args:
            outputs (dict[str, torch.Tensor]): A dictionary that contains at least these entries:
                - "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                    for each query.
                - "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                    for each query.
            targets (list[dict[str, torch.Tensor]]): A list of N targets where each target is a dict containing:
                - "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                    boxes in the target) containing the class labels
                - "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            list[tuple[Tensor, Tensor]]: A list of size batch_size, containing tuples of (indexes, scores).
                During training, indexes are returned as (-1, -1), and scores are returned as None.
                During testing, indexes are returned as (y_index, x_index), and scores are returned as
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = nn.functional.sigmoid(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        out_prob = out_prob[:, tgt_ids]
        neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class - neg_cost_class

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -bbox_overlaps(
            box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy"),
            mode="giou",
        )

        # Final cost matrix
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
