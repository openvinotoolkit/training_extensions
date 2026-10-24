# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter: VisionDataset sample -> Ultralytics sample dict."""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from getitune.data.dataset.base import VisionDataset
from getitune.data.entity.base import ImageInfo

from .geometry import build_ratio_pad, xyxy_abs_to_xywh_norm

logger = logging.getLogger(__name__)


class UltralyticsDatasetAdapter(TorchDataset):
    """Wrap a getitune ``VisionDataset`` as Ultralytics samples."""

    def __init__(
        self,
        vision_dataset: VisionDataset,
        *,
        task_kind: str = "detect",
    ) -> None:
        """Initialize the adapter.

        Args:
            vision_dataset: The getitune VisionDataset to wrap.
            task_kind: Task kind — one of ``"detect"``, ``"segment"``,
            ``"classify"``, ``"multilabel"``, or ``"semantic"``.  Each value
            dispatches to a dedicated ``_getitem_*`` method that builds the
            exact fields the corresponding Ultralytics trainer/validator needs.
        """
        self._dataset = vision_dataset
        self._task_kind = task_kind

        # Pre-populate cacheable augmentation caches (CachedMosaic, CachedMixUp)
        # before multi-worker DataLoaders spawn workers via ``spawn`` context,
        # so every worker inherits a full, diverse, frozen cache instead of
        # independently building a fragmented one. Mirrors the pre-caching
        # done by ``DataModule.train_dataloader()`` for the Lightning backend.
        # No-op for pipelines without a ``CacheableMixin`` augmentation (e.g.
        # val/test) or transforms without a ``prepare`` method.
        transforms = getattr(vision_dataset, "transforms", None)
        if hasattr(transforms, "prepare"):
            transforms.prepare(vision_dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Convert a single getitune sample to an Ultralytics sample dict."""
        sample = self._dataset[index]

        img: torch.Tensor = sample.image  # float32 CHW [0, 1]
        if img.ndim != 3:
            msg = f"Expected 3-D image tensor (CHW), got shape {img.shape}"
            raise ValueError(msg)

        _getitem_mapping: dict[str, Callable[..., dict[str, Any]]] = {
            "classify": self._getitem_classify,
            "multilabel": self._getitem_multilabel,
            "segment": self._getitem_segment,
            "semantic": self._getitem_semantic,
            "detect": self._getitem_detect,
        }
        get_item_fn = _getitem_mapping.get(self._task_kind)
        if get_item_fn is None:
            msg = f"Unknown task_kind: {self._task_kind}"
            raise ValueError(msg)
        return get_item_fn(sample, img)

    def _extract_geometry(
        self,
        sample: object,
        img: torch.Tensor,
    ) -> tuple[
        tuple[int, int], tuple[int, int], tuple[int, int, int, int], tuple[tuple[float, float], tuple[int, int]]
    ]:
        """Return ``(ori_shape, resized_shape, padding, ratio_pad)`` for ``img``."""
        _, tensor_h, tensor_w = img.shape

        img_info: ImageInfo | None = getattr(sample, "img_info", None)
        if img_info is not None:
            ori_shape = img_info.ori_shape
            resized_shape = img_info.img_shape
            padding = img_info.padding
        else:
            ori_shape = (tensor_h, tensor_w)
            resized_shape = (tensor_h, tensor_w)
            padding = (0, 0, 0, 0)

        ratio_pad = build_ratio_pad(ori_shape, resized_shape, padding)
        return ori_shape, resized_shape, padding, ratio_pad

    def _extract_bboxes(
        self,
        sample: object,
        img: torch.Tensor,
        padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return normalised XYWH bboxes in Ultralytics format."""
        _, tensor_h, tensor_w = img.shape
        img_info: ImageInfo | None = getattr(sample, "img_info", None)

        bboxes_raw = getattr(sample, "bboxes", None)
        if bboxes_raw is not None and len(bboxes_raw) > 0:
            canvas = getattr(bboxes_raw, "canvas_size", None)
            canvas_size = cast("tuple[int, int] | None", tuple(canvas) if canvas is not None else None)
            return xyxy_abs_to_xywh_norm(
                bboxes_raw,
                img_w=tensor_w,
                img_h=tensor_h,
                canvas_size=canvas_size,
                scale_factor=img_info.scale_factor if img_info is not None else None,
                padding=padding,
            )
        return np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def _extract_cls(sample: object) -> np.ndarray:
        """Return ``(N, 1)`` float32 class array from a sample."""
        labels_raw = getattr(sample, "label", None)
        if labels_raw is not None:
            if isinstance(labels_raw, torch.Tensor):
                cls = labels_raw.detach().cpu().numpy().astype(np.float32)
            else:
                cls = np.asarray(labels_raw, dtype=np.float32)
            if cls.ndim == 1:
                cls = cls[:, np.newaxis]  # (N,) -> (N, 1)
            return cls
        return np.zeros((0, 1), dtype=np.float32)

    def _getitem_detect(self, sample: object, img: torch.Tensor) -> dict[str, Any]:
        """Return a sample dict for object detection."""
        ori_shape, resized_shape, padding, ratio_pad = self._extract_geometry(sample, img)
        bboxes_xywh = self._extract_bboxes(sample, img, padding)
        cls = self._extract_cls(sample)

        return {
            "im_file": "",
            "img": img,
            "cls": cls,
            "bboxes": bboxes_xywh,
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
        }

    def _getitem_segment(self, sample: object, img: torch.Tensor) -> dict[str, Any]:
        """Return a sample dict for instance segmentation.

        Builds the same fields as detection plus ``masks`` (overlap index map)
        and ``sem_masks`` (per-pixel class labels) required by YOLO-seg's loss.
        """
        ori_shape, resized_shape, padding, ratio_pad = self._extract_geometry(sample, img)
        bboxes_xywh = self._extract_bboxes(sample, img, padding)
        cls = self._extract_cls(sample)
        _, tensor_h, tensor_w = img.shape

        result: dict[str, Any] = {
            "im_file": "",
            "img": img,
            "cls": cls,
            "bboxes": bboxes_xywh,
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
        }

        masks_raw = getattr(sample, "masks", None)
        if masks_raw is not None and len(masks_raw) > 0:
            mask_tensor = masks_raw
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.as_tensor(mask_tensor)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

            if mask_tensor.shape[1:] != (tensor_h, tensor_w):
                pad_left, pad_top, pad_right, pad_bottom = padding
                content_h = tensor_h - pad_top - pad_bottom
                content_w = tensor_w - pad_left - pad_right

                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).float(),
                    size=(content_h, content_w),
                    mode="nearest",
                )[0]

                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                    mask_tensor = torch.nn.functional.pad(
                        mask_tensor,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0,
                    )

            result["masks"] = mask_tensor.to(torch.uint8)

            # Generate per-pixel semantic class labels at the same spatial
            # resolution as masks (required by YOLO26-seg's auxiliary sem
            # loss which indexes sem_masks using instance mask shapes).
            cls_tensor = torch.as_tensor(cls, dtype=torch.float32).squeeze(-1)  # (N,)
            # Per-pixel class = max(class_id * mask_presence); background stays 0.
            sem_masks = (mask_tensor.float() * cls_tensor[:, None, None]).max(0).values  # (H, W)
            result["sem_masks"] = sem_masks
        else:
            logger.debug(
                "Instance segmentation sample has no masks; returning empty masks/sem_masks.",
            )
            result["masks"] = torch.zeros((0, tensor_h, tensor_w), dtype=torch.uint8)
            result["sem_masks"] = torch.zeros((tensor_h, tensor_w), dtype=torch.float32)

        return result

    def _getitem_classify(self, sample: object, img: torch.Tensor) -> dict[str, Any]:
        """Return a compact sample dict for classification tasks.

        Args:
            sample: The raw getitune sample.
            img: Pre-validated CHW float32 image tensor.

        Returns:
            Dict with ``img`` (CHW float32) and ``cls`` (scalar int), plus
            geometry fields required by the standalone-eval validator.
        """
        ori_shape, resized_shape, _, ratio_pad = self._extract_geometry(sample, img)

        labels_raw = getattr(sample, "label", None)
        if labels_raw is None:
            msg = "Classification sample is missing a label"
            raise ValueError(msg)
        cls_idx = int(labels_raw.item()) if isinstance(labels_raw, torch.Tensor) else int(labels_raw)

        return {
            "im_file": "",
            "img": img,
            "cls": cls_idx,
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
        }

    def _getitem_multilabel(self, sample: object, img: torch.Tensor) -> dict[str, Any]:
        """Return a compact sample dict for multi-label classification tasks.

        Args:
            sample: The raw getitune sample.
            img: Pre-validated CHW float32 image tensor.

        Returns:
            Dict with ``img`` (CHW float32) and ``cls`` (multi-hot float
            vector of length ``num_classes``), plus geometry fields required
            by the standalone-eval validator.
        """
        ori_shape, resized_shape, _, ratio_pad = self._extract_geometry(sample, img)

        labels_raw = getattr(sample, "label", None)
        if labels_raw is None:
            msg = "Multi-label sample is missing a label tensor"
            raise ValueError(msg)

        multi_hot = torch.as_tensor(labels_raw, dtype=torch.float32).flatten()
        if multi_hot.ndim != 1:
            msg = f"Multi-label label must be a 1-D multi-hot vector, got shape {multi_hot.shape}"
            raise ValueError(msg)

        return {
            "im_file": "",
            "img": img,
            "cls": multi_hot,
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
        }

    def _getitem_semantic(self, sample: object, img: torch.Tensor) -> dict[str, Any]:
        """Return a sample dict for semantic segmentation.

        Args:
            sample: The raw getitune ``SegmentationSample``.
            img: Pre-validated CHW float32 image tensor.

        Returns:
            Dict with ``img`` (CHW float32) and ``semantic_mask`` (2-D int32
            dense class map), plus geometry fields required by the standalone
            eval validator.
        """
        ori_shape, resized_shape, _, ratio_pad = self._extract_geometry(sample, img)

        masks = getattr(sample, "masks", None)
        if masks is None:
            msg = "Semantic segmentation sample is missing masks"
            raise ValueError(msg)

        mask_tensor = torch.as_tensor(masks, dtype=torch.int32)
        if mask_tensor.ndim == 3:
            if mask_tensor.shape[0] != 1:
                msg = f"Expected single-channel semantic mask, got shape {mask_tensor.shape}"
                raise ValueError(msg)
            mask_tensor = mask_tensor.squeeze(0)
        if mask_tensor.ndim != 2:
            msg = f"Expected 2-D semantic mask, got shape {mask_tensor.shape}"
            raise ValueError(msg)

        return {
            "im_file": "",
            "img": img,
            "semantic_mask": mask_tensor,
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
        }
