# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Instance Segmentation Utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from otx.algo.common.utils.utils import sample_point
from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity


def unpack_inst_seg_entity(entity: InstanceSegBatchDataEntity) -> tuple:
    """Unpack gt_instances, gt_instances_ignore and img_metas based on batch_data_samples.

    Args:
        batch_data_samples (DetBatchDataEntity): Data entity from dataset.

    Returns:
        tuple:

            - batch_gt_instances (list[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_img_metas = []
    for img_info, masks, polygons, bboxes, labels in zip(
        entity.imgs_info,
        entity.masks,
        entity.polygons,
        entity.bboxes,
        entity.labels,
    ):
        metainfo = {
            "img_id": img_info.img_idx,
            "img_shape": img_info.img_shape,
            "ori_shape": img_info.ori_shape,
            "scale_factor": img_info.scale_factor,
            "ignored_labels": img_info.ignored_labels,
        }
        batch_img_metas.append(metainfo)

        gt_masks = masks if len(masks) > 0 else polygons

        batch_gt_instances.append(
            InstanceData(
                metainfo=metainfo,
                masks=gt_masks,
                bboxes=bboxes,
                labels=labels,
            ),
        )

    return batch_gt_instances, batch_img_metas


def empty_instances(
    batch_img_metas: list[dict],
    device: torch.device,
    task_type: str,
    instance_results: list[InstanceData] | None = None,
    mask_thr_binary: int | float = 0,
    num_classes: int = 80,
    score_per_cls: bool = False,
) -> list[InstanceData]:
    """Handle predicted instances when RoI is empty.

    Note: If ``instance_results`` is not None, it will be modified
    in place internally, and then return ``instance_results``

    Args:
        batch_img_metas (list[dict]): List of image information.
        device (torch.device): Device of tensor.
        task_type (str): Expected returned task type. it currently
            supports bbox and mask.
        instance_results (list[InstanceData]): List of instance
            results.
        mask_thr_binary (int, float): mask binarization threshold.
            Defaults to 0.
        box_type (str or type): The empty box type. Defaults to `hbox`.
        use_box_type (bool): Whether to warp boxes with the box type.
            Defaults to False.
        num_classes (int): num_classes of bbox_head. Defaults to 80.
        score_per_cls (bool):  Whether to generate classwise score for
            the empty instance. ``score_per_cls`` will be True when the model
            needs to produce raw results without nms. Defaults to False.

    Returns:
        list[InstanceData]: Detection results of each image
    """
    if task_type not in ("bbox", "mask"):
        msg = f"Only support bbox and mask, but got {task_type}"
        raise ValueError(msg)

    if instance_results is not None and len(instance_results) != len(batch_img_metas):
        msg = "The length of instance_results should be the same as batch_img_metas"
        raise ValueError(msg)

    results_list = []
    for img_id in range(len(batch_img_metas)):
        results = instance_results[img_id] if instance_results is not None else InstanceData()

        if task_type == "bbox":
            bboxes = torch.zeros(0, 4, device=device)
            results.bboxes = bboxes
            score_shape = (0, num_classes + 1) if score_per_cls else (0,)
            results.scores = torch.zeros(score_shape, device=device)
            results.labels = torch.zeros((0,), device=device, dtype=torch.long)
        else:
            img_h, img_w = batch_img_metas[img_id]["ori_shape"][:2]
            # the type of `im_mask` will be torch.bool or torch.uint8,
            # where uint8 if for visualization and debugging.
            im_mask = torch.zeros(
                0,
                img_h,
                img_w,
                device=device,
                dtype=torch.bool if mask_thr_binary >= 0 else torch.uint8,
            )
            results.masks = im_mask
        results_list.append(results)
    return results_list


@dataclass
class ShapeSpec:
    """A simple structure that contains basic shape specification about a tensor.

    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: int = -1
    stride: int = 1


def sample_points_using_uncertainty(
    logits: Tensor,
    uncertainty_function: Callable,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> Tensor:
    """This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty.

    The uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
    prediction as input.

    Source: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former

    Args:
        logits (Tensor): Logit predictions for P points.
        uncertainty_function (Callable): A function that takes logit predictions for P points and
            returns their uncertainties.
        num_points (int): The number of points P to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importance sampling.

    Returns:
        point_coordinates (Tensor): Coordinates for P sampled points.
    """
    num_boxes = logits.shape[0]
    num_points_sampled = int(num_points * oversample_ratio)

    # Get random point coordinates
    point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
    # Get sampled prediction value for the point coordinates
    point_logits = sample_point(logits, point_coordinates, align_corners=False)
    # Calculate the uncertainties based on the sampled prediction values of the points
    point_uncertainties = uncertainty_function(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
    idx += shift[:, None]
    point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

    if num_random_points > 0:
        point_coordinates = torch.cat(
            [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
            dim=1,
        )
    return point_coordinates
