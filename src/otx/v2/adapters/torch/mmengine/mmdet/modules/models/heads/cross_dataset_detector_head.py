"""Cross Dataset Detector head for Ignore labels."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mmdet.models.utils.misc import images_to_levels, multi_apply
from mmdet.registry import MODELS
from torch import Tensor

if TYPE_CHECKING:
    from mmdet.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class CrossDatasetDetectorHead:
    """Head class for Ignore labels."""

    num_classes: int = -1

    def get_atss_targets(
        self,
        anchor_list: list[Any],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.
        """
        num_imgs = len(batch_img_metas)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        (
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
            sampling_results_list,
        ) = multi_apply(
            self._get_targets_single,  # type: ignore[attr-defined]
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs,
        )
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum([results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        # Changed part from mmdet
        valid_label_mask = self.get_valid_label_mask(img_metas=batch_img_metas, all_labels=all_labels)
        valid_label_mask = [i.to(anchor_list[0].device) for i in valid_label_mask]
        if len(valid_label_mask) > 0:
            valid_label_mask = images_to_levels(valid_label_mask, num_level_anchors)
        return (
            anchors_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor,
            valid_label_mask,
        )

    def get_valid_label_mask(
        self,
        img_metas: list[dict],
        all_labels: list[Tensor],
        use_background: bool = False,
    ) -> list[Tensor]:
        """Getter function valid_label_mask."""
        num_classes = self.num_classes + 1 if use_background else self.num_classes
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(num_classes)])
            if "ignored_labels" in meta and len(meta["ignored_labels"]) > 0:
                mask[meta["ignored_labels"]] = 0
                if use_background:
                    mask[self.num_classes] = 0
            mask = mask.repeat(len(all_labels[i]), 1)
            valid_label_mask.append(mask)
        return valid_label_mask
