# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Utils for otx detection algo."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.distributed as dist
from torch import Tensor

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.detection import DetBatchDataEntity


# Methods below come from mmdet.utils and slightly modified.
# https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/utils/misc.py
def reduce_mean(tensor: Tensor) -> Tensor:
    """Obtain the mean of tensor on different GPUs.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/utils/dist_utils.py#L59-L65
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def multi_apply(func: Callable, *args, **kwargs) -> tuple:
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)  # type: ignore[call-overload]
    return tuple(map(list, zip(*map_results)))


def anchor_inside_flags(
    flat_anchors: Tensor,
    valid_flags: Tensor,
    img_shape: tuple[int, ...],
    allowed_border: int = 0,
) -> Tensor:
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = (
            valid_flags
            & (flat_anchors[:, 0] >= -allowed_border)
            & (flat_anchors[:, 1] >= -allowed_border)
            & (flat_anchors[:, 2] < img_w + allowed_border)
            & (flat_anchors[:, 3] < img_h + allowed_border)
        )
    else:
        inside_flags = valid_flags
    return inside_flags


def images_to_levels(target: list[Tensor], num_levels: list[int]) -> list[Tensor]:
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    stacked_target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(stacked_target[:, start:end])
        start = end
    return level_targets


def unmap(data: Tensor, count: int, inds: Tensor, fill: int = 0) -> Tensor:
    """Unmap a subset of item (data) back to the original set of items (of size count)."""
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def filter_scores_and_topk(
    scores: Tensor,
    score_thr: float,
    topk: int,
    results: dict | list | Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict | list | Tensor | None]:
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results
            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results: dict | list | Tensor | None = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            msg = f"Only supports dict or list or Tensor, but get {type(results)}."
            raise NotImplementedError(msg)
    return scores, labels, keep_idxs, filtered_results


def select_single_mlvl(mlvl_tensors: list[Tensor], batch_id: int, detach: bool = True) -> list[Tensor]:
    """Extract a multi-scale single image tensor from a multi-scale batch tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id].detach() for i in range(num_levels)]
    else:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


def unpack_det_entity(entity: DetBatchDataEntity) -> tuple:
    """Unpack gt_instances, gt_instances_ignore and img_metas based on batch_data_samples.

    Args:
        batch_data_samples (DetBatchDataEntity): Data entity from dataset.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_img_metas = []
    for img_info, bboxes, labels in zip(entity.imgs_info, entity.bboxes, entity.labels):
        metainfo = {
            "img_id": img_info.img_idx,
            "img_shape": img_info.img_shape,
            "ori_shape": img_info.ori_shape,
            "pad_shape": img_info.pad_shape,
            "scale_factor": img_info.scale_factor,
            "ignored_labels": img_info.ignored_labels,
        }
        batch_img_metas.append(metainfo)
        batch_gt_instances.append(InstanceData(bboxes=bboxes, labels=labels))

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
        instance_results (list[:obj:`InstanceData`]): List of instance
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
        list[:obj:`InstanceData`]: Detection results of each image
    """
    if task_type not in ("bbox", "mask"):
        msg = f"Only support bbox and mask, but got {task_type}"
        raise ValueError(msg)

    if instance_results is not None and len(instance_results) != len(batch_img_metas):
        msg = "The length of instance_results should be the same as batch_img_metas"
        raise ValueError(msg)

    results_list = []
    for img_id in range(len(batch_img_metas)):
        if instance_results is not None:
            results = instance_results[img_id]
            if not isinstance(results, InstanceData):
                msg = f"instance_results should be InstanceData, but got {type(results)}"
                raise TypeError(msg)
        else:
            results = InstanceData()

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


def dynamic_topk(input: Tensor, k: int, dim: int | None = None, largest: bool = True, sorted: bool = True) -> Tensor:  # noqa: A002
    """Cast k to tensor and make sure k is smaller than input.shape[dim].

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/pytorch/functions/topk.py#L13-L34
    """
    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k, device=input.device, dtype=torch.long)
    # Always keep topk op for dynamic input
    if isinstance(size, torch.Tensor):
        # size would be treated as cpu tensor, trick to avoid that.
        size = k.new_zeros(()) + size
    k = torch.where(k < size, k, size)
    return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)


def unpack_gt_instances(batch_data_samples: list[InstanceData]) -> tuple:
    """Unpack gt_instances, gt_instances_ignore and img_metas based on batch_data_samples.

    Args:
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    # TODO(Eugene): remove this when inst-seg data pipeline decoupling is ready
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)  # type: ignore[attr-defined]
        if "ignored_instances" in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)  # type: ignore[attr-defined]
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


def gather_topk(
    *inputs: tuple[torch.Tensor],
    inds: torch.Tensor,
    batch_size: int,
    is_batched: bool = True,
) -> list[torch.Tensor] | torch.Tensor:
    """Gather topk of each tensor.

    Args:
        inputs (tuple[torch.Tensor]): Tensors to be gathered.
        inds (torch.Tensor): Topk index.
        batch_size (int): batch_size.
        is_batched (bool): Inputs is batched or not.

    Returns:
        Tuple[torch.Tensor]: Gathered tensors.
    """
    if is_batched:
        batch_inds = torch.arange(batch_size, device=inds.device).unsqueeze(-1)
        outputs = [x[batch_inds, inds, ...] if x is not None else None for x in inputs]  # type: ignore[call-overload]
    else:
        prior_inds = inds.new_zeros((1, 1))
        outputs = [x[prior_inds, inds, ...] if x is not None else None for x in inputs]  # type: ignore[call-overload]

    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def distance2bbox(
    points: Tensor,
    distance: Tensor,
    max_shape: Tensor | None = None,
) -> Tensor:
    """Decode distance prediction to bounding box."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            if bboxes.ndim != 3:
                msg = "The dimension of bboxes should be 3."
                raise ValueError(msg)
            if max_shape.size(0) != bboxes.size(0):
                msg = "The size of max_shape should be the same as bboxes."
                raise ValueError(msg)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape], dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


def bbox2distance(
    points: Tensor,
    bbox: Tensor,
    max_dis: float | None = None,
    eps: float = 0.1,
) -> Tensor:
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2) or (b, n, 2), [x, y].
        bbox (Tensor): Shape (n, 4) or (b, n, 4), "xyxy" format
        max_dis (float, optional): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Inverse function of sigmoid."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
