# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR detector wrapper for getitune integration.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes

from getitune.backend.lightning.models.modules.base_module import BaseModule

if TYPE_CHECKING:
    from types import SimpleNamespace


def _compute_multi_scale_scales(
    resolution: int,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> list[int]:
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    return [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]


class RFDETRDetector(BaseModule):
    """Wrapper around RF-DETR's LWDETR model for getitune integration.

    This wrapper handles the interface between getitune's training pipeline and
    the rfdetr package's LWDETR model and SetCriterion.

    Args:
        lwdetr_model: The LWDETR model instance from rfdetr package.
        criterion: The SetCriterion loss function from rfdetr package.
        postprocessor: The PostProcess module from rfdetr package.
        input_size: The input resolution of the model.
        multi_scale: Whether to enable multi-scale training.
    """

    def __init__(
        self,
        lwdetr_model: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        rfdetr_args: SimpleNamespace,
        input_size: int = 560,
        multi_scale: bool = False,
    ) -> None:
        super().__init__()
        self.lwdetr = lwdetr_model
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.input_size = input_size
        self.rng = np.random.default_rng(42)

        # Store scales for multi-scale training
        self.scales = (
            _compute_multi_scale_scales(
                rfdetr_args.resolution, rfdetr_args.expanded_scales, rfdetr_args.patch_size, rfdetr_args.num_windows
            )
            if multi_scale
            else []
        )

    def forward(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Args:
            images: NestedTensor with images and masks from _customize_inputs.
            targets: List of target dictionaries with 'boxes' and 'labels'.

        Returns:
            During training: Loss dictionary.
            During inference: Predictions dictionary with 'pred_logits' and 'pred_boxes'.
        """
        # Multi-scale training - need to handle NestedTensor
        if self.training and self.scales:
            sz = int(self.rng.choice(self.scales))
            images = nn.functional.interpolate(images, size=[sz, sz], mode="bilinear", align_corners=False)

        # Convert to list of tensors if needed
        if isinstance(images, Tensor) and images.dim() == 4:
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = list(images)

        samples = nested_tensor_from_tensor_list(image_list)

        # Forward through model - images is already a NestedTensor
        outputs = self.lwdetr(samples)

        if self.training:
            self.criterion.train()
            if targets is None:
                msg = "targets should not be None"
                raise ValueError(msg)

            loss_dict = self.criterion(outputs, targets)
            weight_dict: dict[str, float] = self.criterion.weight_dict  # pyrefly: ignore[bad-assignment]
            return {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}

        return outputs

    def postprocess(
        self,
        outputs: dict[str, Tensor],
        original_sizes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[BoundingBoxes], list[Tensor], list[Tensor]]:
        """Post-process model outputs to get final predictions.

        Args:
            outputs: Model outputs with 'pred_logits' and 'pred_boxes'.
            original_sizes: List of original image sizes (H, W).

        Returns:
            Tuple of (scores_list, boxes_list, labels_list, masks_list).
        """
        pred_logits = outputs["pred_logits"]
        # Clamp num_select to available elements
        num_elements = pred_logits.shape[1] * pred_logits.shape[2]
        self.postprocessor.num_select = min(int(self.postprocessor.num_select), num_elements)  # pyrefly: ignore

        target_sizes = torch.tensor(original_sizes, device=pred_logits.device)
        results = self.postprocessor(outputs, target_sizes)

        num_fg = pred_logits.shape[-1] - 1
        for r in results:
            fg = r["labels"] < num_fg
            r["scores"] = r["scores"][fg]
            r["labels"] = r["labels"][fg]
            r["boxes"] = r["boxes"][fg]
            if "masks" in r:
                r["masks"] = r["masks"][fg]

        scores_list: list[Tensor] = []
        boxes_list: list[BoundingBoxes] = []
        labels_list: list[Tensor] = []
        masks_list: list[Tensor] = []

        for result, orig_size in zip(results, original_sizes):
            scores_list.append(result["scores"])
            boxes_list.append(
                BoundingBoxes(  # type: ignore[call-overload]
                    result["boxes"],
                    format="xyxy",
                    canvas_size=orig_size,
                ),
            )
            labels_list.append(result["labels"].long())
            if "masks" in result:
                masks_list.append(result["masks"].squeeze(1).to(dtype=torch.uint8))

        return scores_list, boxes_list, labels_list, masks_list

    def export(
        self,
        batch_inputs: Tensor,
        num_select: int = 300,
        merge_scores: bool = False,
        with_nms: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
        """Export function for model tracing with mask support.

        Args:
            batch_inputs: Input images tensor.
            num_select: Number of top predictions to select.
            merge_scores: If True, concatenate ``scores`` as the last column of
                ``boxes``
            with_nms: Whether to embed NMS in the exported graph. Not supported.

        Returns:
            When ``merge_scores`` is ``False`` (default):
                - With masks:    ``(boxes, labels, scores, masks)``
                - Without masks: ``(boxes, labels, scores)``
            When ``merge_scores`` is ``True``:
                - With masks:    ``(boxes_with_scores, labels, masks)``
                - Without masks: ``(boxes_with_scores, labels)``
        """
        if with_nms:
            msg = "RFDETRDetector does not support embedded NMS export."
            raise ValueError(msg)
        outputs = self.lwdetr(batch_inputs)
        # outputs may be dict or tuple in export mode
        if isinstance(outputs, dict):
            pred_boxes = outputs["pred_boxes"]
            pred_logits = outputs["pred_logits"]
            pred_masks = outputs.get("pred_masks")
        elif len(outputs) == 3:
            pred_boxes, pred_logits, pred_masks = outputs
        else:
            pred_boxes, pred_logits = outputs
            pred_masks = None
        # Process outputs similar to PostProcess, but exclude background logit
        # so that labels stay in [0, N-1] (trace-safe; mask-based filtering is not).
        scores = torch.sigmoid(pred_logits[:, :, :-1])
        # Clamp num_select to available elements
        num_elements = scores.shape[1] * scores.shape[2]
        k = min(num_select, num_elements)
        num_fg = pred_logits.shape[-1] - 1
        scores, index = torch.topk(scores.flatten(1), k, dim=-1)

        labels = index % num_fg
        box_index = index // num_fg
        boxes = pred_boxes.gather(
            dim=1,
            index=box_index.unsqueeze(-1).repeat(1, 1, pred_boxes.shape[-1]),
        )
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # Handle masks
        if pred_masks is not None:
            # pred_masks shape: [B, num_queries, H, W]
            # We need to gather masks for selected indices
            masks = pred_masks.gather(
                dim=1,
                index=box_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1]),
            )
            # Apply sigmoid to get mask probabilities
            masks = torch.sigmoid(masks)
            if merge_scores:
                boxes_with_scores = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
                return boxes_with_scores, labels, masks
            return boxes, labels, scores, masks

        if merge_scores:
            boxes_with_scores = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
            return boxes_with_scores, labels
        return boxes, labels, scores
