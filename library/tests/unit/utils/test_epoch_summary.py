# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the backend-agnostic epoch summary formatter."""

from __future__ import annotations

import torch

from getitune.utils.epoch_summary import DEFAULT_EPOCH_SUMMARY_KEYS, format_epoch_summary


def test_formats_known_keys_in_order() -> None:
    line = format_epoch_summary(
        epoch=8,
        max_epochs=200,
        metrics={"lr": 1e-4, "val/map_50": torch.tensor(0.2816), "train/total_loss": 1.1291, "custom": 1.0},
    )
    assert line == "epoch   8/200 | train/total_loss=1.1291  val/map_50=0.2816  lr=0.0001"


def test_skips_non_scalar_values_and_handles_missing_max_epochs() -> None:
    line = format_epoch_summary(
        epoch=3,
        max_epochs=None,
        metrics={"val/map": torch.tensor([0.1, 0.2]), "val/accuracy": "n/a"},
    )
    assert line == "epoch   3 | (no whitelisted metrics)"


def test_custom_keys_and_decimals() -> None:
    line = format_epoch_summary(
        epoch=1,
        max_epochs=2,
        metrics={"train/loss_bbox": 0.123456, "lr": 0.01},
        keys=("train/loss_bbox", *DEFAULT_EPOCH_SUMMARY_KEYS),
        decimals=2,
    )
    assert line == "epoch   1/2 | train/loss_bbox=0.12  lr=0.01"
