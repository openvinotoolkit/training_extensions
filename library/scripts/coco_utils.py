# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert getitune prediction entities to COCO prediction JSON."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pycocotools.mask as mask_utils  # type: ignore[import-untyped]
from torchvision import tv_tensors

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from getitune.types.label import LabelInfo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_files(input_path: Path) -> list[Path]:
    """Return image files under ``input_path`` (recursively) in sorted order."""
    if input_path.is_file():
        return [input_path]
    files = [p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def _xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]


def _encode_mask(mask: torch.Tensor) -> dict:
    if mask.ndim == 3:
        mask = mask[0]
    mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask_np))
    rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
    return rle


def _build_categories(label_info: LabelInfo) -> list[dict]:
    names = getattr(label_info, "label_names", None) or []
    return [{"id": idx + 1, "name": name, "supercategory": "none"} for idx, name in enumerate(names)]


def predictions_to_coco(predictions: list, label_info: LabelInfo, image_files: list[Path] | None = None) -> dict:
    """Convert a list of ``PredictionBatch`` objects to a COCO prediction dict.

    Args:
        predictions: Output of ``engine.predict()`` (list of ``PredictionBatch``).
        label_info: Model ``label_info`` used to build the categories section.
        image_files: Optional list of image paths aligned with the prediction order.
            When omitted, images are referenced by a generated index.

    Returns:
        A dict in COCO result format with ``images``, ``categories`` and ``annotations``.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    image_id = 1

    for batch in predictions:
        batch_labels = getattr(batch, "labels", None)
        batch_scores = getattr(batch, "scores", None)
        batch_boxes = getattr(batch, "bboxes", None)
        batch_masks = getattr(batch, "masks", None)
        batch_keypoints = getattr(batch, "keypoints", None)
        imgs_info = getattr(batch, "imgs_info", None) or []

        for i in range(len(imgs_info)):
            info = imgs_info[i]
            height, width = info.ori_shape[:2] if info is not None else (None, None)
            file_name = str(image_files[len(images)]) if image_files else f"image_{image_id:06d}"
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    **({"height": int(height), "width": int(width)} if height is not None else {}),
                }
            )

            labels = batch_labels[i] if batch_labels is not None else None
            scores = batch_scores[i] if batch_scores is not None else None
            boxes = batch_boxes[i] if batch_boxes is not None else None
            masks = batch_masks[i] if batch_masks is not None else None
            keypoints = batch_keypoints[i] if batch_keypoints is not None else None

            if labels is None:
                image_id += 1
                continue

            labels_list = labels.detach().cpu().numpy().tolist()
            scores_list = scores.detach().cpu().numpy().tolist() if scores is not None else [None] * len(labels_list)

            for j, (label, score) in enumerate(zip(labels_list, scores_list)):
                ann: dict = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "score": float(score) if score is not None else None,
                }
                if boxes is not None and j < len(boxes):
                    box = boxes[j]
                    if isinstance(box, tv_tensors.BoundingBoxes):
                        box = box.tensor
                    ann["bbox"] = _xyxy_to_xywh(np.asarray(box.detach().cpu().numpy()).reshape(-1))
                if masks is not None and j < len(masks):
                    ann["segmentation"] = _encode_mask(masks[j])
                if keypoints is not None and j < len(keypoints):
                    ann["keypoints"] = keypoints[j].detach().cpu().numpy().reshape(-1).tolist()
                annotations.append(ann)
                ann_id += 1

            image_id += 1

    return {
        "images": images,
        "categories": _build_categories(label_info),
        "annotations": annotations,
    }


def write_coco(
    predictions: list,
    output: Path,
    label_info: LabelInfo,
    image_files: list[Path] | None = None,
) -> None:
    """Convert predictions to COCO and write them to ``output`` as JSON."""
    coco = predictions_to_coco(predictions, label_info, image_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2))
