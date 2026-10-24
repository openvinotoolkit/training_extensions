# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Writes ``transformers.Trainer``'s log history to a Geti-shaped metrics CSV (G17)."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


__all__ = ["remap_log_key", "resolve_greater_is_better", "write_metrics_csv"]


def resolve_greater_is_better(monitor: str | None) -> bool:
    """Return ``True`` when a higher ``monitor`` value is better.

    Loss-like keys are lower-is-better; everything else (mAP, Dice, accuracy,
    F1, IoU, ...) is higher-is-better. A recipe can override this explicitly
    via ``training.greater_is_better``.
    """
    if not monitor:
        return True
    return "loss" not in monitor.lower()


def plateau_warmup_lr(base_lr: float, optimizer_step: int, warmup_steps: int) -> float:
    """Linear warmup learning rate for plateau schedulers.

    ``transformers.get_scheduler()`` ignores ``num_warmup_steps`` for
    ``reduce_lr_on_plateau``, so the trainer ramps the LR itself for the first
    ``warmup_steps`` optimizer steps and hands control back to the plateau
    afterwards.

    Args:
        base_lr: LR configured for training.
        optimizer_step: Number of completed optimizer steps (0-based).
        warmup_steps: Ramp length in optimizer steps; ``0`` disables warmup.

    Returns:
        The learning rate to apply for this step.
    """
    if warmup_steps <= 0 or optimizer_step >= warmup_steps:
        return base_lr
    return base_lr * (optimizer_step + 1) / warmup_steps


def remap_log_key(key: str) -> str:
    """Rename a ``Trainer`` log key to the ``train/`` / ``val/`` convention.

    ``Trainer`` logs ``loss``, ``learning_rate``, and ``grad_norm`` during
    training, ``eval_*`` during evaluation, and a final summary entry with
    ``train_runtime``-style keys. Map step metrics to the names consumed by
    the application while preserving canonical ``train/`` and ``val/`` keys.
    """
    if key in ("epoch", "step"):
        return key
    if key in ("loss", "train/total_loss"):
        return "train/total_loss"
    if key in ("learning_rate", "lr"):
        return "lr"
    if key.startswith(("train/", "val/", "validation/")):
        return key
    if key.startswith("eval_val/"):
        return key[len("eval_") :]
    if key.startswith("eval_"):
        return f"val/{key[len('eval_') :]}"
    if key.startswith("train_"):
        return f"train/{key[len('train_') :]}"
    return f"train/{key}"


def write_metrics_csv(log_history: list[dict[str, float]], work_dir: Path) -> Path:
    """Write ``Trainer.state.log_history`` to ``<work_dir>/csv/version_0/metrics.csv``.

    One row per log entry (a training-step log, an eval log, or the final
    training summary), each holding only the columns that entry actually
    reported — the same shape Lightning's ``CSVLogger`` produces when
    different metrics are logged at different frequencies.

    Args:
        log_history: ``trainer.state.log_history`` after training.
        work_dir: The engine's work directory. The file lands at the
            ``csv/version_0/metrics.csv``, so downstream consumers have
            one place to look regardless of backend.

    Returns:
        Path to the written CSV file.
    """
    rows = [{remap_log_key(key): value for key, value in entry.items()} for entry in log_history]

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    csv_dir = work_dir / "csv" / "version_0"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
