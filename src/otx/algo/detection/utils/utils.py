# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Utils for otx detection algo.

Reference :
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/utils.
    - https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/structures/bbox/transforms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.autograd import Function

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.detection import DetBatchDataEntity

if TYPE_CHECKING:
    from otx.algo.detection.detectors.single_stage_detector import SingleStageDetector


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

    TODO (sungchul): check if it can be integrated with otx anchor generators

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


class Vec2Box:
    """Convert the vector to bounding box.

    Args:
        detector (SingleStageDetector): The single stage detector instance.
        image_size (tuple[int, int]): The image size.
        strides (list[int] | None): The strides for each predicted layer. Defaults to None.
        device (str): The device to use. Defaults to "cpu".
    """

    def __init__(
        self,
        detector: SingleStageDetector,
        image_size: tuple[int, int],
        strides: list[int] | None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.strides = strides if strides else self.create_auto_anchor(detector, image_size)
        self.update(image_size, device)

    def create_auto_anchor(self, detector: SingleStageDetector, image_size: tuple[int, int]) -> list[int]:
        """Create the auto anchor for the given detector.

        Args:
            detector (SingleStageDetector): The single stage detector instance.
            image_size (tuple[int, int]): The image size.

        Returns:
            list[int]: The strides for each predicted layer.
        """
        dummy_input = torch.zeros(1, 3, *image_size).to(self.device)
        dummy_main_preds, _ = detector(dummy_input)
        strides = []
        for predict_head in dummy_main_preds:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(image_size[1] // anchor_num[1])
        return strides

    def update(self, image_size: tuple[int, int], device: str | torch.device) -> None:
        """Update the anchor grid and scaler.

        Args:
            image_size (tuple[int, int]): The image size.
            device (str | torch.device): The device to use.
        """
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    def __call__(self, predicts: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Convert the vector to bounding box.

        Args:
            predicts (tuple[Tensor, Tensor, Tensor]): The list of prediction results.

        Returns:
            tuple[Tensor, Tensor, Tensor]: The converted results.
        """
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in predicts:
            pred_cls, pred_anc, pred_box = layer_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        pred_lt_rb = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_lt_rb.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        return preds_cls, preds_anc, preds_box


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
