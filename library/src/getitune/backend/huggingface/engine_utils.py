# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Free-function helpers for :class:`~getitune.backend.huggingface.engine.HFEngine`.

Pulled out of ``engine.py`` so the engine class itself only has to hold the
lifecycle methods (``train``/``test``/``predict``/``export``); the plumbing
around metric formatting, precision mapping, and unbatching predictions lives
here instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from torchvision import tv_tensors

from getitune.data.entity.sample import Prediction

from .trainers.utils import remap_log_key, resolve_greater_is_better

if TYPE_CHECKING:
    import torch

    from getitune.data.entity.sample import PredictionBatch
    from getitune.types.types import METRICS

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_greater_is_better",
    "resolve_precision",
    "summarize_log_history",
    "unbatch_predictions",
]


def resolve_precision(precision: str | int | None) -> tuple[bool, bool]:
    """Map a Lightning-style precision value to ``TrainingArguments(fp16=, bf16=)``.

    Args:
        precision: ``None``, an int (``16``/``32``/``64``), or a string such
            as ``"bf16-mixed"``, matching Lightning's ``_PRECISION_INPUT``.

    Returns:
        ``(fp16, bf16)``. Both ``False`` for 32/64-bit precision, since
        ``Trainer`` trains in fp32 by default when neither flag is set.

    Raises:
        ValueError: If *precision* isn't a recognised value.
    """
    if precision is None:
        return False, False

    value = str(precision).strip().lower()
    if value in {"bf16", "bf16-mixed", "bf16-true"}:
        return False, True
    if value in {"16", "16-mixed", "16-true", "fp16"}:
        return True, False
    if value in {"32", "32-true", "fp32", "64", "64-true", "fp64"}:
        return False, False

    msg = f"Unsupported precision value for the Hugging Face backend: {precision!r}"
    raise ValueError(msg)


def summarize_log_history(log_history: list[dict[str, Any]]) -> METRICS:
    """Collapse ``Trainer.state.log_history`` into one flat metrics dict.

    Later entries win, so the result holds the most recent value logged for
    each key, with the same ``train/`` / ``val/`` naming the metrics CSV uses.
    """
    metrics: dict[str, float] = {}
    for entry in log_history:
        for key, value in entry.items():
            if key in ("epoch", "step") or not isinstance(value, (int, float)):
                continue
            metrics[remap_log_key(key)] = float(value)
    return metrics


def _rewrap(sliced: torch.Tensor, like: tv_tensors.TVTensor) -> torch.Tensor:
    """Re-wrap a boolean-indexed slice as the same ``tv_tensors`` subclass as *like*.

    Plain tensor indexing (``bboxes[keep]``) silently drops the
    ``BoundingBoxes``/``Mask`` subclass, which downstream consumers of
    :class:`Prediction` rely on.
    """
    # pyrefly mis-resolves ``tv_tensors.wrap`` to ``torch._dynamo`` due to a
    # name collision and rejects the valid ``like=`` kwarg; the call is correct.
    return tv_tensors.wrap(sliced, like=like)  # pyrefly: ignore[bad-argument-type, unexpected-keyword, bad-return]


def unbatch_predictions(batch: PredictionBatch, confidence_threshold: float) -> list[Prediction]:
    """Split a :class:`PredictionBatch` into one :class:`Prediction` per image.

    Detection and instance segmentation results are filtered by
    ``scores > confidence_threshold`` (a per-box/per-instance score);
    classification scores (per-class probabilities) and semantic
    segmentation (no scores at all) pass through unfiltered.

    Args:
        batch: A postprocessed batch, as returned by ``HFModel.postprocess()``.
        confidence_threshold: Minimum score to keep a box/instance.

    Returns:
        One prediction per image in *batch*, in the same order.
    """
    predictions: list[Prediction] = []
    for i in range(batch.batch_size):
        img_info = batch.imgs_info[i] if batch.imgs_info is not None else None
        label = batch.labels[i] if batch.labels is not None else None
        bboxes = batch.bboxes[i] if batch.bboxes is not None else None
        masks = batch.masks[i] if batch.masks is not None else None
        scores = batch.scores[i] if batch.scores is not None else None

        if scores is not None and bboxes is not None:
            keep = scores > confidence_threshold
            bboxes = _rewrap(bboxes[keep], bboxes)
            scores = scores[keep]
            if label is not None:
                label = label[keep]
            if masks is not None:
                masks = _rewrap(masks[keep], masks)

        predictions.append(
            Prediction(
                image=batch.images[i],
                img_info=img_info,
                label=label,
                bboxes=bboxes,  # pyrefly: ignore[bad-argument-type]
                masks=masks,  # pyrefly: ignore[bad-argument-type]
                scores=scores,
            )
        )
    return predictions
