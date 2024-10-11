# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""main loss for MonoDETR model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import box_convert

from otx.algo.common.losses.focal_loss import py_sigmoid_focal_loss
from otx.algo.common.losses.iou_loss import giou_loss
from otx.algo.object_detection_3d.matchers.matcher_3d import HungarianMatcher3D
from otx.algo.object_detection_3d.utils.utils import box_cxcylrtb_to_xyxy

from .ddn_loss import DDNLoss

if TYPE_CHECKING:
    from torch import Tensor


class MonoDETRCriterion(nn.Module):
    """This class computes the loss for MonoDETR."""

    def __init__(self, num_classes: int, weight_dict: dict, focal_alpha: float, group_num: int = 11) -> None:
        """MonoDETRCriterion.

        Args:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            focal_alpha: alpha in Focal Loss
            group_num: number of groups for data parallelism
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher3D(cost_class=2, cost_3dcenter=10, cost_bbox=5, cost_giou=2)
        self.weight_dict = weight_dict
        for name in self.loss_map:
            if name not in self.weight_dict:
                self.weight_dict[name] = 1
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num

    def loss_labels(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Classification loss."""
        src_logits = outputs["scores"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = py_sigmoid_focal_loss(
            pred=src_logits,
            target=target_classes_onehot,
            avg_factor=num_boxes,
            alpha=self.focal_alpha,
            reduction="mean",
        )

        return {"loss_ce": loss_ce}

    def loss_3dcenter(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Compute the loss for the 3D center prediction."""
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs["boxes_3d"][:, :, 0:2][idx]
        target_3dcenter = torch.cat([t["boxes_3d"][:, 0:2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = functional.l1_loss(src_3dcenter, target_3dcenter, reduction="none")
        return {"loss_center": loss_3dcenter.sum() / num_boxes}

    def loss_boxes(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Compute l1 loss."""
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs["boxes_3d"][:, :, 2:6][idx]
        target_2dboxes = torch.cat([t["boxes_3d"][:, 2:6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = functional.l1_loss(src_2dboxes, target_2dboxes, reduction="none")
        return {"loss_bbox": loss_bbox.sum() / num_boxes}

    def loss_giou(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Compute the GIoU loss."""
        # giou
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["boxes_3d"][idx]
        target_boxes = torch.cat([t["boxes_3d"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = giou_loss(box_cxcylrtb_to_xyxy(src_boxes), box_cxcylrtb_to_xyxy(target_boxes))
        return {"loss_giou": loss_giou}

    def loss_depths(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Compute the loss for the depth prediction."""
        idx = self._get_src_permutation_idx(indices)

        src_depths = outputs["depth"][idx]
        target_depths = torch.cat([t["depth"][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1]
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + torch.abs(
            depth_log_variance,
        )
        return {"loss_depth": depth_loss.sum() / num_boxes}

    def loss_dims(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Compute the loss for the dimension prediction."""
        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs["size_3d"][idx]
        target_dims = torch.cat([t["size_3d"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = functional.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        return {"loss_dim": dim_loss.sum() / num_boxes}

    def loss_angles(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Compute the loss for the angle prediction."""
        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs["heading_angle"][idx]
        target_heading_angle = torch.cat([t["heading_angle"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        heading_target_cls = target_heading_angle[:, 0].view(-1).long()
        heading_target_res = target_heading_angle[:, 1].view(-1)

        heading_input = heading_input.view(-1, 24)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = functional.cross_entropy(heading_input_cls, heading_target_cls, reduction="none")

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = (
            torch.zeros(heading_target_cls.shape[0], 12)
            .to(device=heading_input.device)
            .scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        )
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = functional.l1_loss(heading_input_res, heading_target_res, reduction="none")

        angle_loss = cls_loss + reg_loss
        return {"loss_angle": angle_loss.sum() / num_boxes}

    def loss_depth_map(self, outputs: dict, targets: list, indices: list, num_boxes: int) -> dict[str, Tensor]:
        """Depth map loss."""
        depth_map_logits = outputs["pred_depth_map_logits"]

        num_gt_per_img = [len(t["boxes"]) for t in targets]
        gt_boxes2d = torch.cat([t["boxes"] for t in targets], dim=0) * torch.tensor(
            [80, 24, 80, 24],
            device=depth_map_logits.device,
        )
        gt_boxes2d = box_convert(gt_boxes2d, "cxcywh", "xyxy")
        gt_center_depth = torch.cat([t["depth"] for t in targets], dim=0).squeeze(dim=1)
        return {"loss_depth_map": self.ddn_loss(depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)}

    def _get_src_permutation_idx(
        self,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(
        self,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @property
    def loss_map(self) -> dict[str, Callable]:
        """Return the loss map."""
        return {
            "loss_ce": self.loss_labels,
            "loss_bbox": self.loss_boxes,
            "loss_giou": self.loss_giou,
            "loss_depth": self.loss_depths,
            "loss_dim": self.loss_dims,
            "loss_angle": self.loss_angles,
            "loss_center": self.loss_3dcenter,
            "loss_depth_map": self.loss_depth_map,
        }

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """This performs the loss computation.

        Args:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes_int = sum([len(t["labels"]) for t in targets]) * group_num
        num_boxes = torch.as_tensor([num_boxes_int], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1)

        # Compute all the requested losses
        losses = {}
        for loss in self.loss_map.values():
            losses.update(loss(outputs, targets, indices, num_boxes))

        losses = {k: losses[k] * self.weight_dict[k] for k in losses}

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for name, loss in self.loss_map.items():
                    if name == "loss_depth_map":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = loss(aux_outputs, targets, indices, num_boxes.item())
                    l_dict = {k + f"_aux_{i}": v * self.weight_dict[k] for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
