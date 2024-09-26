# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""HungarianMatcher3D module for 3d object detection."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from otx.algo.common.utils.bbox_overlaps import bbox_overlaps
from otx.algo.object_detection_3d.utils.utils import box_cxcylrtb_to_xyxy


class HungarianMatcher3D(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_3dcenter: float = 1.0,
        cost_bbox: float = 1.0,
        cost_giou: float = 1.0,
    ):
        """Creates the matcher.

        Args:
            cost_class (float): This is the relative weight of the classification error in the matching cost.
            cost_3dcenter (float): This is the relative weight of the L1 error of the 3d center in the matching cost.
            cost_bbox (float): This is the relative weight of the L1 error of the bbox coordinates in the matching cost.
            cost_giou (float): This is the relative weight of the giou loss of the bbox in the matching cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dcenter = cost_3dcenter
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list, group_num: int = 11) -> list:
        """Performs the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        out_3dcenter = outputs["pred_boxes"][:, :, 0:2].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0:2] for v in targets])

        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)

        out_2dbbox = outputs["pred_boxes"][:, :, 2:6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2:6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)

        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
        cost_giou = -bbox_overlaps(
            box_cxcylrtb_to_xyxy(out_bbox),
            box_cxcylrtb_to_xyxy(tgt_bbox),
            mode="giou",
        )
        # Final cost matrix
        c = (
            self.cost_bbox * cost_bbox
            + self.cost_3dcenter * cost_3dcenter
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        c = c.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        g_num_queries = num_queries // group_num
        c_list = c.split(g_num_queries, dim=1)
        for g_i in range(group_num):
            c_g = c_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(c_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
