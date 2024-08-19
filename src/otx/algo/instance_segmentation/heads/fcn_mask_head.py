# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.roi_heads.mask_heads.fcn_mask_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn.functional
from torch import Tensor, nn
from torch.nn.modules.utils import _pair

from otx.algo.common.utils.structures import SamplingResult
from otx.algo.instance_segmentation.utils.structures.mask import mask_target
from otx.algo.instance_segmentation.utils.utils import empty_instances
from otx.algo.modules.base_module import BaseModule, ModuleList
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer

BYTES_PER_FLOAT = 4
#  determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


if TYPE_CHECKING:
    from otx.algo.utils.mmengine_utils import InstanceData


class FCNMaskHead(BaseModule):
    """FCNMaskHead."""

    def __init__(
        self,
        loss_mask: nn.Module,
        num_convs: int = 4,
        roi_feat_size: int = 14,
        in_channels: int = 256,
        conv_kernel_size: int = 3,
        conv_out_channels: int = 256,
        num_classes: int = 80,
        class_agnostic: int = False,
        normalization: Callable[..., nn.Module] | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        if init_cfg is not None:
            msg = "To prevent abnormal initialization behavior, init_cfg is not allowed to be set"
            raise ValueError(msg)

        super().__init__(init_cfg=init_cfg)
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalization = normalization

        self.loss_mask = loss_mask

        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                Conv2dModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    normalization=build_norm_layer(normalization, num_features=self.conv_out_channels),
                ),
            )
        upsample_in_channels = self.conv_out_channels if self.num_convs > 0 else in_channels

        _scale_factor = 2
        upsample_cfg = {
            "in_channels": upsample_in_channels,
            "out_channels": self.conv_out_channels,
            "kernel_size": _scale_factor,
            "stride": _scale_factor,
        }
        self.upsample = nn.ConvTranspose2d(**upsample_cfg)
        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = self.conv_out_channels
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            if hasattr(m, "weight") and hasattr(m, "bias"):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Extract mask RoI features.

        Returns:
            Tensor: Predicted foreground masks.
        """
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            x = self.relu(x)
        return self.conv_logits(x)

    def get_targets(
        self,
        sampling_results: list[SamplingResult],
        batch_gt_instances: list[InstanceData],
        rcnn_train_cfg: dict,
    ) -> Tensor:
        """Calculate the ground truth for all samples in a batch according to the sampling_results.

        Args:
            sampling_results (List[SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (dict): `train_cfg` of RCNN.

        Returns:
            Tensor: Mask target of each positive proposals in the image.
        """
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]  # type: ignore[attr-defined]
        meta_infos = [gt_instances.metainfo for gt_instances in batch_gt_instances]
        return mask_target(
            pos_proposals,
            pos_assigned_gt_inds,
            gt_masks,
            rcnn_train_cfg,
            meta_infos,
        )

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: list[SamplingResult],
        batch_gt_instances: list[InstanceData],
        rcnn_train_cfg: dict,
    ) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (num_pos, num_classes, h, w).
            sampling_results (List[SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (dict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
        """
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg,
        )

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        loss = {}
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        elif self.class_agnostic:
            loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
        else:
            loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
        loss["loss_mask"] = loss_mask
        return {"loss_mask": loss, "mask_targets": mask_targets}

    def predict_by_feat(
        self,
        mask_preds: tuple[Tensor],
        results_list: list[InstanceData],
        batch_img_metas: list[dict],
        rcnn_test_cfg: dict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> list[InstanceData]:
        """Transform a batch of output features extracted from the head into mask results.

        Args:
            mask_preds (tuple[Tensor]): Tuple of predicted foreground masks,
                each has shape (n, num_classes, h, w).
            results_list (list[InstanceData]): Detection results of
                each image.
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (dict): `test_cfg` of Bbox Head.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            list[InstanceData]: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if len(mask_preds) != len(results_list) != len(batch_img_metas):
            msg = "The number of inputs should be the same."
            raise ValueError(msg)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes  # type: ignore[attr-defined]
            if bboxes.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type="mask",
                    instance_results=[results],
                    mask_thr_binary=rcnn_test_cfg["mask_thr_binary"],
                )[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=mask_preds[img_id],
                    bboxes=bboxes,
                    labels=results.labels,  # type: ignore[attr-defined]
                    img_meta=img_meta,
                    rcnn_test_cfg=rcnn_test_cfg,
                    rescale=rescale,
                    activate_map=activate_map,
                )
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(
        self,
        mask_preds: Tensor,
        bboxes: Tensor,
        labels: Tensor,
        img_meta: dict,
        rcnn_test_cfg: dict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> Tensor:
        """Get segmentation masks from mask_preds and bboxes.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (n, num_classes, h, w).
            bboxes (Tensor): Predicted bboxes, has shape (n, 4)
            labels (Tensor): Labels of bboxes, has shape (n, )
            img_meta (dict): image information.
            rcnn_test_cfg (dict): `test_cfg` of Bbox Head.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            Tensor: Encoded masks, has shape (n, img_w, img_h)
        """
        scale_factor = bboxes.new_tensor(img_meta["scale_factor"][::-1]).repeat((1, 2))
        img_h, img_w = img_meta["ori_shape"][:2]
        device = bboxes.device

        mask_preds = mask_preds.sigmoid() if not activate_map else bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        num_preds = len(mask_preds)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = num_preds
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(np.ceil(num_preds * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            if num_chunks > num_preds:
                msg = "Default GPU_MEM_LIMIT is too small; try increasing it"
                raise ValueError(msg)
        chunks = torch.chunk(torch.arange(num_preds, device=device), num_chunks)

        threshold = rcnn_test_cfg["mask_thr_binary"]
        im_mask = torch.zeros(
            num_preds,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8,
        )

        if not self.class_agnostic:
            mask_preds = mask_preds[range(num_preds), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_preds[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == "cpu",
            )

            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            im_mask[(inds, *spatial_inds)] = masks_chunk
        return im_mask

    def export_by_feat(
        self,
        mask_preds: Tensor,
        results_list: tuple[Tensor, ...],
        batch_img_metas: list[dict],
        rcnn_test_cfg: dict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> torch.Tensor:
        """Transform a batch of output features extracted from the head into mask results."""
        warnings.warn(f"rescale: {rescale} is not supported in deploy mode", stacklevel=2)
        warnings.warn(f"activate_map: {activate_map} is not supported in deploy mode", stacklevel=2)

        dets, det_labels = results_list
        dets = dets.view(-1, 5)
        det_labels = det_labels.view(-1)
        mask_preds = mask_preds.sigmoid()
        bboxes = dets[:, :4]
        labels = det_labels
        if not self.class_agnostic:
            box_inds = torch.arange(mask_preds.shape[0], device=bboxes.device)
            mask_pred = mask_preds[box_inds, labels][:, None]
        return mask_pred


def _do_paste_mask(masks: Tensor, boxes: Tensor, img_h: int, img_w: int, skip_empty: bool = True) -> tuple:
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
        is the slice object.

            If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.

            If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        box_values, _ = boxes.min(dim=0)
        x0_int, y0_int = torch.clamp(box_values.floor()[:2] - 1, min=0).to(torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    num_preds = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(num_preds, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(num_preds, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = torch.nn.functional.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    return img_masks[:, 0], ()
