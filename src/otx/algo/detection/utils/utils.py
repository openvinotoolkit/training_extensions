# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Utils for otx detection algo.

Reference :
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/utils.
    - https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/structures/bbox/transforms.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.detection import DetBatchDataEntity


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


def unpack_det_entity(entity: DetBatchDataEntity) -> tuple:
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
    for img_info, bboxes, labels in zip(entity.imgs_info, entity.bboxes, entity.labels):
        metainfo = {
            "img_id": img_info.img_idx,
            "img_shape": img_info.img_shape,
            "ori_shape": img_info.ori_shape,
            "scale_factor": img_info.scale_factor,
            "ignored_labels": img_info.ignored_labels,
        }
        batch_img_metas.append(metainfo)
        batch_gt_instances.append(InstanceData(bboxes=bboxes, labels=labels))

    return batch_gt_instances, batch_img_metas


def distance2bbox_export(points: Tensor, distance: Tensor, max_shape: Tensor | None = None) -> Tensor:
    """Decode distance prediction to bounding box for export.

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/structures/bbox/transforms.py#L11-L43
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        # clip bboxes with dynamic `min` and `max`
        x1, y1, x2, y2 = clip_bboxes(x1, y1, x2, y2, max_shape)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    return bboxes


def clip_bboxes(
    x1: Tensor,
    y1: Tensor,
    x2: Tensor,
    y2: Tensor,
    max_shape: Tensor | tuple[int, ...],
) -> tuple[Tensor, ...]:
    """Clip bboxes for onnx.

    Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/deploy/utils.py#L31-L72

    Since torch.clamp cannot have dynamic `min` and `max`, we scale the
      boxes by 1/max_shape and clamp in the range [0, 1] if necessary.

    Args:
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor | Sequence[int]): The (H,W) of original image.

    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    if len(max_shape) != 2:
        msg = "`max_shape` should be [h, w]."
        raise ValueError(msg)

    if isinstance(max_shape, Tensor):
        # scale by 1/max_shape
        x1 = x1 / max_shape[1]
        y1 = y1 / max_shape[0]
        x2 = x2 / max_shape[1]
        y2 = y2 / max_shape[0]

        # clamp [0, 1]
        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)

        # scale back
        x1 = x1 * max_shape[1]
        y1 = y1 * max_shape[0]
        x2 = x2 * max_shape[1]
        y2 = y2 * max_shape[0]
    else:
        x1 = torch.clamp(x1, 0, max_shape[1])
        y1 = torch.clamp(y1, 0, max_shape[0])
        x2 = torch.clamp(x2, 0, max_shape[1])
        y2 = torch.clamp(y2, 0, max_shape[0])
    return x1, y1, x2, y2


class SigmoidGeometricMean(Function):
    """Forward and backward function of geometric mean of two sigmoid functions.

    This implementation with analytical gradient function substitutes
    the autograd function of (x.sigmoid() * y.sigmoid()).sqrt(). The
    original implementation incurs none during gradient backprapagation
    if both x and y are very small values.
    """

    @staticmethod
    def forward(ctx, x, y) -> Tensor:  # noqa: D102, ANN001
        x_sigmoid = x.sigmoid()
        y_sigmoid = y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx, grad_output) -> tuple[Tensor, Tensor]:  # noqa: D102, ANN001
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        grad_x = grad_output * z * (1 - x_sigmoid) / 2
        grad_y = grad_output * z * (1 - y_sigmoid) / 2
        return grad_x, grad_y


sigmoid_geometric_mean = SigmoidGeometricMean.apply


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute the inverse of sigmoid function."""
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))
