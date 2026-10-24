# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""D-FINE criterion implementations. Modified from https://github.com/Peterande/D-FINE."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torchvision.ops import box_convert

from getitune.backend.lightning.models.common.utils.assigners.hungarian_matcher import HungarianMatcher
from getitune.backend.lightning.models.common.utils.bbox_overlaps import bbox_overlaps
from getitune.backend.lightning.models.detection.utils.utils import dfine_bbox2distance


class DFINECriterion(nn.Module):
    """D-Fine criterion with FGL and DDF losses.

    TODO(Eugene): Consider merge with RTDETRCriterion in the next PR.

    The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    Args:
        weight_dict (dict[str, int | float]): A dictionary containing the weights for different loss components.
        alpha (float, optional): The alpha parameter for the loss calculation. Defaults to 0.2.
        gamma (float, optional): The gamma parameter for the loss calculation. Defaults to 2.0.
        num_classes (int, optional): The number of classes. Defaults to 80.
        reg_max (int, optional): The maximum number of bin targets. Defaults to 32.
    """

    def __init__(
        self,
        weight_dict: dict[str, int | float],
        alpha: float = 0.2,
        gamma: float = 2.0,
        num_classes: int = 80,
        reg_max: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(
            cost_dict={
                "cost_class": 2.0,
                "cost_bbox": 5.0,
                "cost_giou": 2.0,
            },
        )
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.num_pos, self.num_neg = 0.0, 0.0

    def loss_labels_vfl(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
    ) -> dict[str, Tensor]:
        """Varifocal Loss (VFL) for label prediction.

        Args:
            outputs (dict[str, Tensor]): Model outputs.
            targets (List[Dict[str, Tensor]]): List of target dictionaries.
            indices (List[Tuple[int, int]]): List of tuples of indices.
            num_boxes (int): Number of predicted boxes.

        Returns:
            dict[str, Tensor]: The loss dictionary.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious = bbox_overlaps(
            box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
        )
        ious = torch.diag(ious).detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = f.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = f.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = f.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
    ) -> dict[str, Tensor]:
        """Compute the losses re)L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.

        Args:
            outputs (dict[str, Tensor]): The outputs of the model.
            targets (list[dict[str, Tensor]]): The targets.
            indices (list[tuple[int, int]]): The indices of the matched boxes.
            num_boxes (int): The number of boxes.

        Returns:
            dict[str, Tensor]: The losses.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = f.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            bbox_overlaps(
                box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                mode="giou",
            ),
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
        temperature: int = 5,
    ) -> dict[str, Tensor]:
        """Compute Fine-Grained Localization (FGL) Loss and Decoupled Distillation Focal (DDF) Loss.

        Args:
            outputs (dict[str, Tensor]): The outputs of the model.
            targets (list[dict[str, Tensor]]): The targets.
            indices (list[tuple[int, int]]): The indices of the matched boxes.
            num_boxes (int): The number of boxes.
            temperature (int, optional): Temperature for distillation. Defaults to 5.

        Returns:
            dict[str, Tensor]: FGL and DDF losses.
        """
        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)],
                dim=0,
            )

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                target_corners, weight_right, weight_left = dfine_bbox2distance(
                    ref_points,
                    box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                    self.reg_max,
                    outputs["reg_scale"],
                    outputs["up"],
                )

            ious = torch.diag(
                bbox_overlaps(
                    box_convert(outputs["pred_boxes"][idx], in_fmt="cxcywh", out_fmt="xyxy"),
                    box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
                ),
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = DFINECriterion.fgl_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            # Compute Decoupled Distillation Focal (DDF) Loss
            if "teacher_corners" in outputs and outputs["teacher_corners"] is not None:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype,
                    )
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = (
                        weight_targets_local
                        * (temperature**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                f.log_softmax(pred_corners / temperature, dim=1),
                                f.softmax(target_corners.detach() / temperature, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = 8 / outputs["pred_boxes"].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (
                        self.num_pos + self.num_neg
                    )

        return losses

    def _get_src_permutation_idx(
        self,
        indices: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_go_indices(
        self,
        indices: list[tuple[Tensor, Tensor]],
        indices_aux_list: list[list[tuple[Tensor, Tensor]]],
    ) -> list[Tensor]:
        """Get a matching union set across all decoder layers.

        Args:
            indices: matching indices of the last decoder layer
            indices_aux_list: matching indices of all decoder layers
        """
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    @property
    def _available_losses(self) -> tuple[Callable]:
        return (self.loss_boxes, self.loss_labels_vfl, self.loss_local)  # type: ignore[return-value]

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """This performs the loss computation.

        Args:
            outputs (dict[str, torch.Tensor]): dict of tensors, see the output
                specification of the model for the format
            targets (list[dict[str, torch.Tensor]]): list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc
        Returns:
            dict[str, torch.Tensor]: dict of losses
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Get the matching union set across all decoder layers.
        indices_aux_list, cached_indices, cached_indices_enc = [], [], []
        for aux_outputs in outputs["aux_outputs"] + [outputs["pre_outputs"]]:
            indices_aux = self.matcher(aux_outputs, targets)
            cached_indices.append(indices_aux)
            indices_aux_list.append(indices_aux)
        for aux_outputs in outputs["enc_aux_outputs"]:
            indices_enc = self.matcher(aux_outputs, targets)
            cached_indices_enc.append(indices_enc)
            indices_aux_list.append(indices_enc)
        indices_go = self._get_go_indices(indices, indices_aux_list)

        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor(
            [num_boxes_go],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        num_boxes_go = torch.clamp(num_boxes_go, min=1).item()

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self._available_losses:
            indices_in = indices_go if loss in [self.loss_boxes, self.loss_local] else indices
            num_boxes_in = num_boxes_go if loss in [self.loss_boxes, self.loss_local] else num_boxes
            l_dict = loss(outputs, targets, indices_in, num_boxes_in)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self._available_losses:
                    if loss in [self.loss_boxes, self.loss_local]:
                        indices_in = indices_go
                        num_boxes_in = num_boxes_go
                    else:
                        indices_in = cached_indices[i]
                        num_boxes_in = num_boxes
                    l_dict = loss(aux_outputs, targets, indices_in, num_boxes_in)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self._available_losses:
                if loss in [self.loss_boxes, self.loss_local]:
                    indices_in = indices_go
                    num_boxes_in = num_boxes_go
                else:
                    indices_in = cached_indices[-1]
                    num_boxes_in = num_boxes
                l_dict = loss(aux_outputs, targets, indices_in, num_boxes_in)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            enc_targets = targets
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self._available_losses:
                    if loss == self.loss_boxes:
                        indices_in = indices_go
                        num_boxes_in = num_boxes_go
                    else:
                        indices_in = cached_indices_enc[i]
                        num_boxes_in = num_boxes
                    l_dict = loss(aux_outputs, enc_targets, indices_in, num_boxes_in)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For dfine
        if "dn_outputs" in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self._available_losses:
                    l_dict = loss(aux_outputs, targets, indices_dn, dn_num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self._available_losses:
                    l_dict = loss(aux_outputs, targets, indices_dn, dn_num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(
        dn_meta: dict[str, list[Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """get_cdn_matched_indices.

        Args:
            dn_meta (dict[str, list[torch.Tensor]]): meta data for cdn
            targets (list[dict[str, torch.Tensor]]): targets
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        # Use subsampled num_gts from dn_meta if available (when ground truth was capped)
        # Otherwise fall back to counting from original targets
        num_gts = dn_meta["dn_num_gts"] if "dn_num_gts" in dn_meta else [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                if len(dn_positive_idx[i]) != len(gt_idx):
                    msg = "The number of positive indices should be equal to the number of ground truths."
                    raise ValueError(msg)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    ),
                )

        return dn_match_indices

    @staticmethod
    def fgl_loss(
        preds: Tensor,
        targets: Tensor,
        weight_right: Tensor,
        weight_left: Tensor,
        iou_weight: Tensor | None = None,
        reduction: str = "sum",
        avg_factor: float | None = None,
    ) -> Tensor:
        """Fine-Grained Localization (FGL) Loss.

        Args:
            preds (Tensor): predicted distances
            targets (Tensor): target distances
            weight_right (Tensor): weight for right distance
            weight_left (Tensor): weight for left distance
            iou_weight (Tensor, optional): IoU weight. Defaults to None.
            reduction (str, optional): reduction method. Defaults to "sum".
            avg_factor (float, optional): average factor. Defaults to None.

        Returns:
            Tensor: FGL loss
        """
        dis_left = targets.long()
        dis_right = dis_left + 1

        loss_left = f.cross_entropy(
            preds,
            dis_left,
            reduction="none",
        ) * weight_left.reshape(-1)

        loss_right = f.cross_entropy(
            preds,
            dis_right,
            reduction="none",
        ) * weight_right.reshape(-1)

        loss = loss_left + loss_right

        if iou_weight is not None:
            iou_weight = iou_weight.float()
            loss = loss * iou_weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
