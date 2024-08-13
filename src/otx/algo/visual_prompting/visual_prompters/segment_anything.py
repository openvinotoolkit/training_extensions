# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
"""Segment Anything model for the OTX visual prompting."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import tv_tensors

from otx.algo.visual_prompting.utils.postprocess import postprocess_masks
from otx.core.data.entity.base import Points


class SegmentAnything(nn.Module):
    """Visual prompting model class for Segment Anything."""

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        criterion: nn.Module,
        image_size: int = 1024,
        mask_threshold: float = 0.0,
        use_stability_score: bool = False,
        return_single_mask: bool = False,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.use_stability_score = use_stability_score
        self.return_single_mask = return_single_mask
        self.return_extra_metrics = return_extra_metrics
        self.stability_score_offset = stability_score_offset

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.criterion = criterion

    def forward(
        self,
        images: tv_tensors.Image,
        ori_shapes: list[Tensor],
        bboxes: list[tv_tensors.BoundingBoxes | None],
        points: list[tuple[Points, Tensor] | None],
        gt_masks: list[tv_tensors.Mask] | None = None,
    ) -> Tensor | tuple[list[Tensor], list[Tensor]]:
        """Forward method for SAM training/validation/prediction.

        Args:
            images (tv_tensors.Image): Images with shape (B, C, H, W).
            ori_shapes (List[Tensor]): List of original shapes per image.
            bboxes (List[tv_tensors.BoundingBoxes], optional): A Nx4 array given a box prompt to the model,
                in XYXY format.
            points (List[Tuple[Points, Tensor]], optional): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            gt_masks (List[tv_tensors.Mask], optional): Ground truth masks for loss calculation.

        Returns:
            (Tensor): Calculated loss values.
            (Tuple[List[Tensor], List[Tensor]]): Tuple of list with predicted masks with shape (B, 1, H, W)
                and List with IoU predictions with shape (N, 1).
        """
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for idx, embedding in enumerate(image_embeddings):
            low_res_masks, iou_predictions = [], []
            for prompt in [bboxes[idx], points[idx]]:
                if prompt is None:
                    continue

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=prompt if isinstance(prompt[0], Points) else None,
                    boxes=prompt if isinstance(prompt, tv_tensors.BoundingBoxes) else None,
                    masks=None,
                )
                _low_res_masks, _iou_predictions = self.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,  # when given multiple prompts. if there is single prompt True would be better. # noqa: E501
                )
                low_res_masks.append(_low_res_masks)
                iou_predictions.append(_iou_predictions)

            pred_masks.append(torch.cat(low_res_masks, dim=0))
            ious.append(torch.cat(iou_predictions, dim=0))

        if self.training:
            return self.criterion(pred_masks, gt_masks, ious, ori_shapes)

        post_processed_pred_masks: list[Tensor] = []
        for pred_mask, ori_shape in zip(pred_masks, ori_shapes):
            post_processed_pred_mask = postprocess_masks(pred_mask, self.image_size, ori_shape)
            post_processed_pred_masks.append(post_processed_pred_mask.squeeze(1).sigmoid())
        return post_processed_pred_masks, ious
