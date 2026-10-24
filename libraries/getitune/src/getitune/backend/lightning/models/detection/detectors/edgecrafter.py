# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter DETR-style detector wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from getitune.backend.lightning.models.detection.heads.ec_decoder import ECTransformer

import torch
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes

from .detection_transformer import DETR


class ECDETRDetector(DETR):
    """EdgeCrafter DETR-style detector wrapping :class:`ECTransformer`.

    Extends :class:`DETR` with:

    * ``_forward_features`` — passes the stride-8 backbone feature to the decoder
      so that the optional segmentation head (ECSeg) has access to fine-grained
      spatial context.
    * ``postprocess`` — converts ``ECTransformer.postprocess``'s per-image
      list-of-dicts output into the tuple format expected by the getitune
      training/inference pipeline.  When the model produces masks a 4-tuple
      is returned; otherwise a 3-tuple.
    * ``export`` — strips the ``explain_mode`` flag (not supported by
      ECTransformer) and runs the inference path without teacher inputs.

    Args:
        backbone: :class:`ECViTAdapter` feature extractor.
        encoder: :class:`HybridEncoder` neck.
        decoder: :class:`ECTransformer` head.
        num_classes: Number of object classes.
        criterion: Loss function (:class:`ECCriterion`).
        optimizer_configuration: Per-param-group LR / WD overrides.
        multi_scale: Whether to use multi-scale training.
        num_top_queries: Number of top scoring queries to keep. Defaults to 300.
        input_size: Training input resolution for multi-scale sampling. Defaults to 640.
    """

    def _forward_features(
        self,
        images: Tensor,
        targets: object = None,
    ) -> dict[str, Any]:
        """Backbone → encoder → decoder forward with spatial feature propagation.

        The stride-8 backbone feature (``backbone_feats[0]``) is forwarded to
        :class:`ECTransformer` as ``spatial_feat`` so that the optional
        :class:`SegmentationHead` (ECSeg variants) can produce instance masks.
        For detection-only models the decoder ignores this argument.

        Args:
            images: Input image tensor [B, C, H, W].
            targets: Ground-truth target dicts (used during training for CDN denoising).

        Returns:
            ECTransformer output dict.
        """
        backbone_feats = self.backbone(images)
        encoder_feats = self.encoder(backbone_feats)
        return self.decoder(encoder_feats, cast("list[dict] | None", targets), spatial_feat=backbone_feats[0])

    def postprocess(
        self,
        outputs: dict[str, Tensor],
        original_sizes: list[tuple[int, int]],
        deploy_mode: bool = False,
    ) -> dict[str, Tensor] | tuple:
        """Post-process :class:`ECTransformer` outputs to getitune format.

        Delegates actual decoding to ``self.decoder.postprocess`` which handles
        box de-normalisation, top-k selection, and optional mask upsampling.

        Args:
            outputs: Raw model output dict (``pred_logits``, ``pred_boxes``,
                optionally ``pred_masks``).
            original_sizes: Per-image ``(H, W)`` tuples for rescaling.
            deploy_mode: If ``True``, returns a compact dict of stacked tensors
                suitable for ONNX / OpenVINO export.

        Returns:
            *deploy mode*: ``dict(bboxes, labels, scores[, masks])``.

            *inference mode*: ``(scores_list, boxes_list, labels_list)``
            or ``(scores_list, boxes_list, labels_list, masks_list)``
            when instance-segmentation masks are present.
        """
        device = outputs["pred_logits"].device
        sizes_tensor = torch.tensor(original_sizes, device=device, dtype=torch.float32)  # [B, 2] as (H, W)

        results = cast("ECTransformer", self.decoder).postprocess(outputs, sizes_tensor, self.num_top_queries)

        has_masks = bool(results) and "masks" in results[0]

        if deploy_mode:
            scores = torch.stack([r["scores"] for r in results])
            labels = torch.stack([r["labels"] for r in results])
            boxes = torch.stack([r["boxes"] for r in results])
            out: dict[str, Tensor] = {"bboxes": boxes, "labels": labels, "scores": scores}
            if has_masks:
                out["masks"] = torch.stack([r["masks"].float() for r in results])
            return out

        scores_list: list[Tensor] = []
        boxes_list: list[BoundingBoxes] = []
        labels_list: list[Tensor] = []
        masks_list: list[Tensor] = []

        for res, orig_size in zip(results, original_sizes):
            scores_list.append(res["scores"])
            boxes_list.append(
                BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    res["boxes"], format="xyxy", canvas_size=orig_size
                )
            )
            labels_list.append(res["labels"].long())
            if has_masks:
                masks_list.append(res["masks"])

        if has_masks:
            return scores_list, boxes_list, labels_list, masks_list
        return scores_list, boxes_list, labels_list

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
        explain_mode: bool = False,
        with_nms: bool = False,
    ) -> dict[str, Tensor]:
        """Export-mode forward pass (no teacher, no denoising, deploy postprocess).

        ``explain_mode`` is accepted for API compatibility with the base class and
        the instance-segmentation base's ``forward_for_tracing``, but is not used
        because ``ECTransformer`` does not implement XAI features.

        Args:
            batch_inputs: Input image batch [B, C, H, W].
            batch_img_metas: List of per-image meta dicts; ``"img_shape"`` key
                is used for rescaling.
            explain_mode: Ignored.
            with_nms: Whether to embed NMS in the exported graph. Not supported.

        Returns:
            Dict with ``bboxes``, ``labels``, ``scores`` (and ``masks`` for ECSeg).
        """
        if with_nms:
            msg = "ECDETRDetector does not support embedded NMS export."
            raise ValueError(msg)
        backbone_feats = self.backbone(batch_inputs)
        encoder_feats = self.encoder(backbone_feats)
        predictions = self.decoder(encoder_feats, spatial_feat=backbone_feats[0])
        return self.postprocess(  # type: ignore[return-value]
            predictions,
            [meta["img_shape"] for meta in batch_img_metas],
            deploy_mode=True,
        )
