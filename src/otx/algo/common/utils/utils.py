# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Utils for otx detection algo.

Reference :
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/utils.
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/utils.
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/bbox/transforms.
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/layers/transformer/utils.
    - https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/pytorch/functions/topk.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


def reduce_mean(tensor: Tensor) -> Tensor:
    """Obtain the mean of tensor on different GPUs."""
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
        elif isinstance(results, Tensor):
            filtered_results = results[keep_idxs]
        else:
            msg = f"Only supports dict or list or Tensor, but get {type(results)}."
            raise NotImplementedError(msg)
    return scores, labels, keep_idxs, filtered_results


def select_single_mlvl(mlvl_tensors: list[Tensor] | tuple[Tensor], batch_id: int, detach: bool = True) -> list[Tensor]:
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


def dynamic_topk(input: Tensor, k: int, dim: int | None = None, largest: bool = True, sorted: bool = True) -> Tensor:  # noqa: A002
    """Cast k to tensor and make sure k is smaller than input.shape[dim].

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/pytorch/functions/topk.py#L13-L34
    """
    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if not isinstance(k, Tensor):
        k = torch.tensor(k, device=input.device, dtype=torch.long)
    # Always keep topk op for dynamic input
    if isinstance(size, Tensor):
        # size would be treated as cpu tensor, trick to avoid that.
        size = k.new_zeros(()) + size
    k = torch.where(k < size, k, size)
    return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)


def gather_topk(
    *inputs: tuple[Tensor],
    inds: Tensor,
    batch_size: int,
    is_batched: bool = True,
) -> list[Tensor] | Tensor:
    """Gather topk of each tensor.

    Args:
        inputs (tuple[Tensor]): Tensors to be gathered.
        inds (Tensor): Topk index.
        batch_size (int): batch_size.
        is_batched (bool): Inputs is batched or not.

    Returns:
        list[Tensor] or Tensor: Gathered tensor(s).
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
    """Decode distance prediction to bounding box.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/bbox/transforms.py#L147-L203
    """
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

        if not isinstance(max_shape, Tensor):
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

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/bbox/transforms.py#L206-L230

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
    """Inverse function of sigmoid.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/layers/transformer/utils.py#L100-L113
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def cut_mixer(images: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    """Applies cut-mix augmentation to the input images and masks.

    Args:
        images (Tensor): The input images tensor.
        masks (Tensor): The input masks tensor.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the augmented images and masks tensors.
    """

    def rand_bbox(size: tuple[int, ...], lam: float) -> tuple[list[int], ...]:
        """Generates random bounding box coordinates.

        Args:
            size (tuple[int, ...]): The size of the input tensor.
            lam (float): The lambda value for cut-mix augmentation.

        Returns:
            tuple[list[int, ...], ...]: The bounding box coordinates (bbx1, bby1, bbx2, bby2).
        """
        # past implementation
        w = size[2]
        h = size[3]
        b = size[0]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(size=[b], low=int(w / 8), high=w)
        cy = np.random.randint(size=[b], low=int(h / 8), high=h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)

        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    target_device = images.device
    mix_data = images.clone()
    mix_masks = masks.clone()
    u_rand_index = torch.randperm(images.size()[0])[: images.size()[0]].to(target_device)
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(images.size(), lam=np.random.beta(4, 4))

    for i in range(mix_data.shape[0]):
        mix_data[i, :, u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]] = images[
            u_rand_index[i],
            :,
            u_bbx1[i] : u_bbx2[i],
            u_bby1[i] : u_bby2[i],
        ]

        mix_masks[i, :, u_bbx1[i] : u_bbx2[i], u_bby1[i] : u_bby2[i]] = masks[
            u_rand_index[i],
            :,
            u_bbx1[i] : u_bbx2[i],
            u_bby1[i] : u_bby2[i],
        ]

    del images, masks

    return mix_data, mix_masks.squeeze(dim=1)
