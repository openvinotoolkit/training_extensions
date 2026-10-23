# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic formatting of the per-epoch training summary line.

Every training backend (Lightning, Hugging Face, Ultralytics) emits one plain
``logging`` record per epoch so that captured logs (application job logs,
``nohup``, CI artifacts) show training progress even though the interactive
progress bars are not persisted. This module holds the shared formatting so
the line looks identical regardless of the backend, e.g.::

    epoch   8/200 | train/total_loss=1.1291  val/map_50=0.2816  lr=0.0001
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ["DEFAULT_EPOCH_SUMMARY_KEYS", "format_epoch_summary", "to_scalar"]

# Ordered, task-agnostic shortlist. Missing keys are skipped silently, so the
# same list works for every task. New tasks can extend it without breaking
# existing ones.
DEFAULT_EPOCH_SUMMARY_KEYS: tuple[str, ...] = (
    "train/total_loss",
    "train/loss",
    "val/PCK",
    "val/accuracy",
    "val/Dice",
    "val/mIoU",
    "val/map_50",
    "val/map",
    "val/f1-score",
    "val/image_AUROC",
    "lr",
)


def to_scalar(value: object) -> float | None:
    """Best-effort conversion of a metric value to ``float``.

    Returns ``None`` for missing, non-numeric, or multi-element tensor values.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def format_epoch_summary(
    epoch: int,
    max_epochs: int | None,
    metrics: Mapping[str, object],
    keys: Iterable[str] = DEFAULT_EPOCH_SUMMARY_KEYS,
    decimals: int = 4,
) -> str:
    """Format a one-line summary of the metrics of a finished epoch.

    Args:
        epoch: 1-based index of the finished epoch.
        max_epochs: Total number of epochs, or ``None`` if unbounded/unknown.
        metrics: Metric values keyed by name. Values may be tensors or numbers.
        keys: Ordered metric keys to include; keys absent from ``metrics`` or
            holding non-scalar values are skipped.
        decimals: Number of decimal digits used when formatting values.

    Returns:
        The formatted summary line.
    """
    parts: list[str] = []
    for key in keys:
        value = to_scalar(metrics.get(key))
        if value is None:
            continue
        parts.append(f"{key}={value:.{decimals}f}")

    header = f"epoch {epoch:>3}/{max_epochs}" if max_epochs is not None else f"epoch {epoch:>3}"
    body = "  ".join(parts) if parts else "(no whitelisted metrics)"
    return f"{header} | {body}"
