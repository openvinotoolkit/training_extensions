# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Instance Segmentation Utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as f
from torch import Tensor, nn

from otx.algo.common.utils.utils import point_sample
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


def c2_xavier_fill(module: nn.Module) -> None:
    """Initialize `module.weight` using the "XavierFill" implemented in Caffe2.

    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.

    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


def check_if_dynamo_compiling() -> bool:
    """Check if the current code is being compiled by TorchScript."""
    from torch._dynamo import is_compiling

    return is_compiling()


@dataclass
class ShapeSpec:
    """A simple structure that contains basic shape specification about a tensor.

    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: int = -1
    stride: int = 1


class Conv2d(torch.nn.Conv2d):
    """A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features."""

    def __init__(self, *args, **kwargs) -> None:
        """Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`.

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = f.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_uncertain_point_coords_with_randomness(
    coarse_logits: Tensor,
    uncertainty_func: Callable,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> Tensor:
    """Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty.

    The unceratinties are calculated for each point using 'uncertainty_func' function that takes point's logit
    prediction as input. See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes,
        num_uncertain_points,
        2,
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords
