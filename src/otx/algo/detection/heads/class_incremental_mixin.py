# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Cross Dataset Detector head for Ignore labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from otx.algo.common.utils.utils import multi_apply
from otx.algo.detection.utils.utils import images_to_levels

if TYPE_CHECKING:
    from otx.algo.utils.mmengine_utils import InstanceData


class ClassIncrementalMixin:
    """Head class for Ignore labels."""

    def get_atss_targets(
        self,
        anchor_list: list,
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: list[InstanceData],
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list[InstanceData] | None = None,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(batch_img_metas)
        if not len(anchor_list) == len(valid_flag_list) == num_imgs:
            msg = f"Invalid inputs, anchor_list: {len(anchor_list)}, \
                   valid_flag_list: {len(valid_flag_list)}, num_imgs: {num_imgs}"
            raise ValueError(msg)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            if len(anchor_list[i]) != len(valid_flag_list[i]):
                msg = "anchor_list and valid_flag_list have different shape"
                raise ValueError(msg)
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs  # type: ignore[list-item]
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
        use_bg: bool = False,
    ) -> list[Tensor]:
        """Calculate valid label mask with ignored labels."""
        num_classes = self.num_classes + 1 if use_bg else self.num_classes  # type: ignore[attr-defined]
        valid_label_mask = []
        for i, meta in enumerate(img_metas):
            mask = torch.Tensor([1 for _ in range(num_classes)])
            if "ignored_labels" in meta and meta["ignored_labels"]:
                mask[meta["ignored_labels"]] = 0
                if use_bg:
                    mask[self.num_classes] = 0  # type: ignore[attr-defined]
            mask = mask.repeat(len(all_labels[i]), 1)
            valid_label_mask.append(mask)
        return valid_label_mask
