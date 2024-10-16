# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DETR style Hungarian matcher for bipartite matching."""

from __future__ import annotations

from functools import partial

import torch
from otx.algo.common.utils.bbox_overlaps import bbox_overlaps
from otx.algo.common.utils.utils import sample_point
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops import box_convert


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
    """A pair wise version of the cross entropy loss.

    Args:
        inputs (Tensor): A tensor representing a mask.
        labels (Tensor): A tensor with the same shape as inputs.
            Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        Tensor: The computed loss between each pairs.
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

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_dict: dict[str, float | int],
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        """Creates the matcher.

        Args:
            cost_dict (dict[str, float | int]): A dictionary containing the cost for each annotation type.
                The dictionary may have the following keys:
                - "cost_class" (float | int): The weight for the class cost.
                - "cost_bbox" (float | int): The weight for the bounding box cost.
                - "cost_giou" (float | int): The weight for the generalized intersection over union (IoU) cost.
                - "cost_mask" (float | int): The weight for the mask cost.
                - "cost_dice" (float | int): The weight for the dice cost.
            alpha (float, optional): The alpha parameter for the cost computation. Defaults to 0.25.
            gamma (float, optional): The gamma parameter for the cost computation. Defaults to 2.0.
        """
        super().__init__()
        self.cost_functions = self.build_cost_functions(cost_dict)
        self.alpha = alpha
        self.gamma = gamma

    def build_cost_functions(self, cost_dict: dict[str, float | int]) -> list[partial]:
        """Return the cost functions based on the provided cost dictionary.

        Args:
            cost_dict (dict[str, float | int]): A dictionary containing the cost for each annotation type.

        Returns:
            list[partial callable]: A list of partial functions for computing the costs.
        """
        cost_functions = []
        if "cost_class" in cost_dict:
            cost_functions.append(
                partial(self.label_cost, cost_class=cost_dict["cost_class"]),
            )
        if "cost_bbox" in cost_dict:
            cost_functions.append(
                partial(self.bbox_cost, cost_bbox=cost_dict["cost_bbox"]),
            )
        if "cost_giou" in cost_dict:
            cost_functions.append(
                partial(self.giou_cost, cost_giou=cost_dict["cost_giou"]),
            )
        if "cost_mask" in cost_dict and "cost_dice" in cost_dict:
            cost_functions.append(
                partial(self.mask_cost, cost_mask=cost_dict["cost_mask"], cost_dice=cost_dict["cost_dice"]),
            )
        return cost_functions

    @torch.no_grad()
    def label_cost(self, output: dict[str, Tensor], cost_class: float | int) -> Tensor:
        """Compute the classification cost.

        Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
        The 1 is a constant that doesn't change the matching, it can be ommitted.

        Args:
            output (dict[str, Tensor]): A dictionary containing the following:
                - "pred_logits": The predicted logits for each query.
                - "pred_boxes": The predicted box coordinates for each query.
                - "pred_masks": The predicted masks for each query. Can be None.
                - "target_labels": The target class labels.
                - "target_boxes": The target box coordinates.
                - "target_mask": The target masks. Can be None.
            cost_weight (float | int): The weight for the cost.

        Returns:
            Tensor: label classification cost
        """
        pred_probs = output["pred_logits"].sigmoid()
        target_labels = output["target_labels"]

        # compute class cost
        neg_cost_class = (1 - self.alpha) * (pred_probs**self.gamma) * (-(1 - pred_probs + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - pred_probs) ** self.gamma) * (-(pred_probs + 1e-8).log())
        cost = pos_cost_class[:, target_labels] - neg_cost_class[:, target_labels]
        return cost * cost_class

    @torch.no_grad()
    def bbox_cost(self, output: dict[str, Tensor], cost_bbox: float | int) -> Tensor:
        """Compute the L1 cost between boxes.

        Args:
            output (dict[str, Tensor]): A dictionary containing the following:
                - "pred_logits": The predicted logits for each query.
                - "pred_boxes": The predicted box coordinates for each query.
                - "pred_masks": The predicted masks for each query. Can be None.
                - "target_labels": The target class labels.
                - "target_boxes": The target box coordinates.
                - "target_mask": The target masks. Can be None.
            cost_bbox (float | int): The weight for the cost.

        Returns:
            Tensor: The L1 cost between boxes
        """
        pred_bboxes = output["pred_boxes"]
        target_bboxes = output["target_boxes"]

        # Compute the L1 cost between boxes
        return torch.cdist(pred_bboxes, target_bboxes, p=1) * cost_bbox

    @torch.no_grad()
    def giou_cost(self, output: dict[str, Tensor], cost_giou: float | int) -> Tensor:
        """Compute the giou cost betwen boxes.

        Args:
            output (dict[str, Tensor]): A dictionary containing the following:
                - "pred_logits": The predicted logits for each query.
                - "pred_boxes": The predicted box coordinates for each query.
                - "pred_masks": The predicted masks for each query. Can be None.
                - "target_labels": The target class labels.
                - "target_boxes": The target box coordinates.
                - "target_mask": The target masks. Can be None.
            cost_giou (float | int): The weight for the cost.

        Returns:
            Tensor: The L1 cost between boxes
        """
        pred_bboxes = output["pred_boxes"]
        target_bboxes = output["target_boxes"]

        # Compute the giou cost betwen boxes
        cost = -bbox_overlaps(
            box_convert(pred_bboxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_bboxes, in_fmt="cxcywh", out_fmt="xyxy"),
            mode="giou",
        )
        return cost * cost_giou

    @torch.no_grad()
    def mask_cost(
        self,
        output: dict[str, Tensor],
        cost_mask: float | int,
        cost_dice: float | int,
        num_points: int = 12544,
    ) -> Tensor:
        """Compute the mask cost.

        Args:
            output (dict[str, Tensor]): A dictionary containing the following:
                - "pred_logits": The predicted logits for each query.
                - "pred_boxes": The predicted box coordinates for each query.
                - "pred_masks": The predicted masks for each query. Can be None.
                - "target_labels": The target class labels.
                - "target_boxes": The target box coordinates.
                - "target_mask": The target masks. Can be None.
            cost_weight (float | int): The weight for the cost.
            num_points (int, optional): The number of points to sample. Defaults to 12544.

        Returns:
            Tensor: The mask cost
        """
        out_mask = output["pred_masks"]
        target_mask = output["target_mask"]

        out_mask = out_mask[:, None]
        target_mask = target_mask[:, None]

        # Sample ground truth and predicted masks
        point_coordinates = torch.rand(1, num_points, 2, device=out_mask.device).type_as(out_mask)

        # get gt labels
        target_mask = sample_point(
            target_mask.to(out_mask),
            point_coordinates.repeat(target_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        out_mask = sample_point(
            out_mask,
            point_coordinates.repeat(out_mask.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)

        cost = pair_wise_sigmoid_cross_entropy_loss(out_mask, target_mask) * cost_mask
        cost += pair_wise_dice_loss(out_mask, target_mask) * cost_dice
        return cost

    def batch_preparation(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[dict[str, Tensor | None]]:
        """Prepare annotation and prediction into a list of dictionaries. .

        Args:
            outputs (dict[str, Tensor]): A dictionary containing the model's predictions with the following keys:
            - "pred_logits": Tensor of shape [batch_size, num_queries, num_classes] with the classification logits.
            - "pred_boxes": Tensor of shape [batch_size, num_queries, 4] with the predicted box coordinates.
            - "pred_masks": Tensor of shape [batch_size, num_queries, h, w] with the predicted masks (optional).
            targets (list[dict[str, Tensor]]): A list of dictionaries, each containing the ground truth annotations
                with the following keys:
            - "labels": Tensor of shape [num_target_boxes] with the class labels.
            - "boxes": Tensor of shape [num_target_boxes, 4] with the target box coordinates.
            - "masks": Tensor of shape [num_target_boxes, h, w] with the segmentation masks (optional).

        Returns:
            list[dict[str, Tensor | None]]: A list of dictionaries, each containing the following keys:
            - "pred_logits": The predicted logits for each query.
            - "pred_boxes": The predicted box coordinates for each query.
            - "pred_masks": The predicted masks for each query (optional).
            - "target_boxes": The target box coordinates.
            - "target_labels": The target class labels.
            - "target_mask": The target masks (optional).
        """
        batch_size = len(targets)
        return [
            {
                "pred_logits": outputs["pred_logits"][i],
                "pred_boxes": outputs["pred_boxes"][i],
                "pred_masks": outputs["pred_masks"][i] if "pred_masks" in outputs else None,
                "target_boxes": targets[i]["boxes"],
                "target_labels": targets[i]["labels"],
                "target_mask": targets[i]["masks"] if "masks" in targets[i] else None,
            }
            for i in range(batch_size)
        ]

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        """Performs the matching.

        Args:
            outputs (dict[str, Tensor]): A dictionary that contains at least these entries:
                - "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                    for each query.
                - "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                    for each query.
                - "pred_masks": Tensor of dim [batch_size, num_queries, h, w] with the predicted masks for each query.
            targets (list[dict[str, Tensor]]): A list of N targets where each target is a dict containing:
                - "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                    boxes in the target) containing the class labels
                - "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.
                - "masks": Tensor of dim [num_target_boxes, h, w] containing the segmentation masks for each target box.

        Returns:
            list[tuple[Tensor, Tensor]]: A list of size batch_size, containing tuples of (indexes, scores).
                During training, indexes are returned as (-1, -1), and scores are returned as None.
                During testing, indexes are returned as (y_index, x_index), and scores are returned as
        """
        batch_size = len(targets)

        indices = []
        formatted_batch = self.batch_preparation(outputs, targets)

        # Iterate through batch size
        for i in range(batch_size):
            # Compute the cost matrix for each cost function and sum them.
            cost_matrix = torch.stack(
                [cost_func(formatted_batch[i]) for cost_func in self.cost_functions],
            ).sum(dim=0)

            # Eliminate infinite values in cost_matrix to avoid the error ``ValueError: cost matrix is infeasible``
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))

            # Perform assignment using the hungarian algorithm in scipy
            assigned_indices = linear_sum_assignment(cost_matrix.cpu())
            indices.append(assigned_indices)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
