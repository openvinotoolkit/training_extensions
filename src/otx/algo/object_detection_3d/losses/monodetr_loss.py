"""MonoDETR loss"""

import torch
import torch.nn.functional as F
from torch import nn

from otx.algo.object_detection_3d.matcher import HungarianMatcher
from otx.algo.object_detection_3d.utils import box_ops

from .ddn_loss import DDNLoss
from .focal_loss import sigmoid_focal_loss


class MonoDETRCriterion(nn.Module):
    """This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, weight_dict, focal_alpha, group_num=11):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class=2, cost_3dcenter=10, cost_bbox=5, cost_giou=2)
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

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
        loss_ce = (
            sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs["pred_boxes"][:, :, 0:2][idx]
        target_3dcenter = torch.cat([t["boxes_3d"][:, 0:2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction="none")
        losses = {}
        losses["loss_center"] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs["pred_boxes"][:, :, 2:6][idx]
        target_2dboxes = torch.cat([t["boxes_3d"][:, 2:6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction="none")
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes_3d"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcylrtb_to_xyxy(src_boxes),
                box_ops.box_cxcylrtb_to_xyxy(target_boxes),
            ),
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)

        src_depths = outputs["pred_depth"][idx]
        target_depths = torch.cat([t["depth"][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1]
        depth_loss = (
            1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance
        )
        losses = {}
        losses["loss_depth"] = depth_loss.sum() / num_boxes
        return losses

    def loss_dims(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs["pred_3d_dim"][idx]
        target_dims = torch.cat([t["size_3d"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses["loss_dim"] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs["pred_angle"][idx]
        target_heading_angle = torch.cat([t["heading_angle"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        heading_target_cls = target_heading_angle[:, 0].view(-1).long()
        heading_target_res = target_heading_angle[:, 1].view(-1)

        heading_input = heading_input.view(-1, 24)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction="none")

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = (
            torch.zeros(heading_target_cls.shape[0], 12)
            .to(device=heading_input.device)
            .scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        )
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction="none")

        angle_loss = cls_loss + reg_loss
        losses = {}
        losses["loss_angle"] = angle_loss.sum() / num_boxes
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs["pred_depth_map_logits"]

        num_gt_per_img = [len(t["boxes"]) for t in targets]
        gt_boxes2d = torch.cat([t["boxes"] for t in targets], dim=0) * torch.tensor(
            [80, 24, 80, 24],
            device=depth_map_logits.device,
        )
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t["depth"] for t in targets], dim=0).squeeze(dim=1)

        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @property
    def loss_map(self):
        return {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "depths": self.loss_depths,
            "dims": self.loss_dims,
            "angles": self.loss_angles,
            "center": self.loss_3dcenter,
            "depth_map": self.loss_depth_map,
        }

    def forward(self, outputs, targets, mask_dict=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses

        losses = {}
        for name, loss in self.loss_map.items():
            # ipdb.set_trace()
            losses.update(loss(outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for name, loss in self.loss_map.items():
                    if name == "depth_map":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if name == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = loss(aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # TODO (KIRILL): refactor
        losses_weighted = {k: losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict}
        return losses_weighted
