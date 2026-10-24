# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.

"""Utils for getitune detection algo.

Reference :
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/utils.
    - https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/structures/bbox/transforms.
    - https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_utils.py
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torchvision.ops import box_convert

from getitune.backend.lightning.models.utils.utils import InstanceData
from getitune.data.entity.sample import SampleBatch


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


def unpack_det_entity(entity: SampleBatch) -> tuple:
    """Unpack gt_instances, gt_instances_ignore and img_metas based on batch_data_samples.

    Args:
        entity (TorchDataBatch): Data entity from dataset.

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
    imgs_infos = entity.imgs_info if entity.imgs_info is not None else [[] for _ in range(entity.batch_size)]  # type: ignore[union-attr,misc]

    for idx, img_info in enumerate(imgs_infos):
        metainfo = {
            "img_id": img_info.img_idx,  # type: ignore[union-attr]
            "img_shape": img_info.img_shape,  # type: ignore[union-attr]
            "ori_shape": img_info.ori_shape,  # type: ignore[union-attr]
            "scale_factor": img_info.scale_factor,  # type: ignore[union-attr]
            "ignored_labels": img_info.ignored_labels,  # type: ignore[union-attr]
        }
        batch_img_metas.append(metainfo)
        _bbox = entity.bboxes[idx] if entity.bboxes is not None else []
        _label = entity.labels[idx] if entity.labels is not None else []
        batch_gt_instances.append(InstanceData(bboxes=_bbox, labels=_label))

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


def auto_pad(kernel_size: int | tuple[int, int], dilation: int | tuple[int, int] = 1, **kwargs) -> tuple[int, int]:  # noqa: ARG001
    """Auto Padding for the convolution blocks.

    Args:
        kernel_size (int | tuple[int, int]): The kernel size of the convolution block.
        dilation (int | tuple[int, int]): The dilation rate of the convolution block. Defaults to 1.

    Returns:
        tuple[int, int]: The padding size for the convolution block.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return pad_h, pad_w


def round_up(x: int | Tensor, div: int = 1) -> int | Tensor:
    """Rounds up `x` to the bigger-nearest multiple of `div`.

    Args:
        x (int | Tensor): The input value.
        div (int): The divisor value. Defaults to 1.

    Returns:
        int | Tensor: The rounded up value.
    """
    return x + (-x % div)


def generate_anchors(image_size: tuple[int, int], strides: list[int]) -> tuple[Tensor, Tensor]:
    """Find the anchor maps for each height and width.

    TODO (sungchul): check if it can be integrated with getitune anchor generators

    Args:
        image_size (tuple[int, int]): the image size of augmented image size.
        strides list[int]: the stride size for each predicted layer.

    Returns:
        tuple[Tensor, Tensor]: The anchor maps with (HW x 2) and the scaler maps with (HW,).
    """
    height, width = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = width // stride * height // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, height, stride) + shift
        w = torch.arange(0, width, stride) + shift
        anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


def generate_scales(input_size: int, base_size_repeat: int = 3, divisior: int = 32) -> list[int]:
    """Generates scales for multi-scale training."""
    scale_repeat = (input_size - int(input_size * 0.75 / divisior) * divisior) // divisior
    scales = [int(input_size * 0.75 / divisior) * divisior + i * divisior for i in range(scale_repeat)]
    scales += [input_size] * base_size_repeat
    scales += [int(input_size * 1.25 / divisior) * divisior - i * divisior for i in range(scale_repeat)]
    return scales


def set_info_into_instance(layer_dict: dict[str, Any]) -> nn.Module:
    """Set the information into the instance.

    Args:
        layer_dict (dict[str, Any]): The dictionary of instance with given information.

    Returns:
        nn.Module: The instance with given information.
    """
    layer = layer_dict.pop("module")
    for k, v in layer_dict.items():
        setattr(layer, k, v)
    return layer


def dfine_weighting_function(reg_max: int, up: Tensor, reg_scale: Tensor) -> Tensor:
    """Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence, where maximum offset is ±up * H / W.
        reg_scale (Tensor): Controls the curvature of the Weighting Function.
                        Larger values result in flatter weights near the central axis W(reg_max/2)=0
                        and steeper weights at both ends.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    return torch.cat(
        [
            -upper_bound2,
            torch.cat(left_values),
            torch.zeros_like(up[0][None]),
            torch.cat(right_values),
            upper_bound2,
        ],
        0,
    )


def dfine_distance2bbox(points: Tensor, distance: Tensor, reg_scale: Tensor) -> Tensor:
    """Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                        where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                        point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return box_convert(bboxes, in_fmt="xyxy", out_fmt="cxcywh")


def dfine_bbox2distance(
    points: Tensor,
    bbox: Tensor,
    reg_max: int,
    reg_scale: Tensor,
    up: Tensor,
    eps: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Converts bounding box coordinates to distances from a reference point.

    Refer to D-Fine: https://github.com/Peterande/D-FINE.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """

    def _translate_gt(gt: Tensor, reg_max: int, reg_scale: Tensor, up: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Decodes bounding box ground truth (GT) values into distribution-based GT representations.

        This function maps continuous GT values into discrete distribution bins, which can be used
        for regression tasks in object detection models.

        It calculates the indices of the closest bins to each GT value and assigns interpolation weights
        to these bins based on their proximity to the GT value.

        In the paper:
            'a' (up) controlling the upper bounds.
            'c' (reg_scale) controlling the curvature.

        Args:
            gt (Tensor): Ground truth bounding box values, shape (N, ).
            reg_max (int): Maximum number of discrete bins for the distribution.
            reg_scale (Tensor): Controls the curvature of the Weighting Function.
            up (Tensor): Controls the upper bounds of the Weighting Function.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
                - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
                - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
        """
        gt = gt.reshape(-1)
        function_values = dfine_weighting_function(reg_max, up, reg_scale)

        # Find the closest left-side indices for each value
        diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
        mask = diffs <= 0
        closest_left_indices = torch.sum(mask, dim=1) - 1

        # Calculate the weights for the interpolation
        indices = closest_left_indices.float()

        weight_right = torch.zeros_like(indices)
        weight_left = torch.zeros_like(indices)

        valid_idx_mask = (indices >= 0) & (indices < reg_max)
        valid_indices = indices[valid_idx_mask].long()

        # Obtain distances
        left_values = function_values[valid_indices]
        right_values = function_values[valid_indices + 1]

        left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
        right_diffs = torch.abs(right_values - gt[valid_idx_mask])

        # Valid weights
        weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
        weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

        # Invalid weights (out of range)
        invalid_idx_mask_neg = indices < 0
        weight_right[invalid_idx_mask_neg] = 0.0
        weight_left[invalid_idx_mask_neg] = 1.0
        indices[invalid_idx_mask_neg] = 0.0

        invalid_idx_mask_pos = indices >= reg_max
        weight_right[invalid_idx_mask_pos] = 1.0
        weight_left[invalid_idx_mask_pos] = 0.0
        indices[invalid_idx_mask_pos] = reg_max - 0.1

        return indices, weight_right, weight_left

    reg_scale = abs(reg_scale)
    # ϕ = (dᴳᵀ- d⁰) / {H, H, W, W}
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = _translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
