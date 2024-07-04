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
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.ops.boxes import box_area
import math

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


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()

    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def normalize_bounding_boxes(target_boxes: torch.Tensor, image_width: int, image_height: int):
    # Convert BoundingBoxes to tensor; shape: [N, 4], where N is the number of boxes,
    # and each box is [xmin, ymin, xmax, ymax]
    # Normalize xmin and xmax by image width
    bounding_boxes = target_boxes.clone()
    bounding_boxes[:, 0] = bounding_boxes[:, 0] / image_width
    bounding_boxes[:, 2] = bounding_boxes[:, 2] / image_width

    # Normalize ymin and ymax by image height
    bounding_boxes[:, 1] = bounding_boxes[:, 1] / image_height
    bounding_boxes[:, 3] = bounding_boxes[:, 3] / image_height

    return bounding_boxes