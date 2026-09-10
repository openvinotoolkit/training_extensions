# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the free-function helpers in ``engine_utils.py``."""

from __future__ import annotations

import pytest

from getitune.backend.huggingface.engine_utils import (
    resolve_greater_is_better,
    resolve_precision,
    summarize_log_history,
)


class TestResolvePrecision:
    def test_none_means_fp32(self) -> None:
        assert resolve_precision(None) == (False, False)

    @pytest.mark.parametrize("value", [16, "16", "16-mixed", "16-true", "fp16"])
    def test_16_bit_variants_select_fp16(self, value: int | str) -> None:
        assert resolve_precision(value) == (True, False)

    @pytest.mark.parametrize("value", ["bf16", "bf16-mixed", "bf16-true"])
    def test_bf16_variants_select_bf16(self, value: str) -> None:
        assert resolve_precision(value) == (False, True)

    @pytest.mark.parametrize("value", [32, "32", "32-true", "fp32", 64, "64-true"])
    def test_32_and_64_bit_variants_select_neither(self, value: int | str) -> None:
        assert resolve_precision(value) == (False, False)

    def test_rejects_unknown_values(self) -> None:
        with pytest.raises(ValueError, match="Unsupported precision"):
            resolve_precision("int8")


class TestResolveGreaterIsBetter:
    @pytest.mark.parametrize("monitor", ["val/loss", "train/loss", "eval_loss", "loss"])
    def test_loss_is_lower_better(self, monitor: str) -> None:
        assert resolve_greater_is_better(monitor) is False

    @pytest.mark.parametrize(
        "monitor", ["val/map", "val/map_50", "val/f1-score", "val/Dice", "test/accuracy", "val/iou"]
    )
    def test_known_metrics_are_higher_better(self, monitor: str) -> None:
        assert resolve_greater_is_better(monitor) is True

    @pytest.mark.parametrize("monitor", [None, "", "val/custom_metric"])
    def test_unknown_defaults_to_higher_better(self, monitor: str | None) -> None:
        assert resolve_greater_is_better(monitor) is True


class TestSummarizeLogHistory:
    def test_later_entries_win(self) -> None:
        log_history = [
            {"loss": 2.0, "epoch": 1.0, "step": 4},
            {"loss": 1.5, "epoch": 2.0, "step": 8},
        ]
        assert summarize_log_history(log_history) == {"train/loss": 1.5}

    def test_epoch_and_step_are_excluded(self) -> None:
        metrics = summarize_log_history([{"loss": 1.0, "epoch": 1.0, "step": 4}])
        assert "epoch" not in metrics
        assert "step" not in metrics

    def test_non_numeric_values_are_skipped(self) -> None:
        metrics = summarize_log_history([{"loss": 1.0, "some_string": "ignored"}])
        assert metrics == {"train/loss": 1.0}

    def test_empty_history_yields_empty_metrics(self) -> None:
        assert summarize_log_history([]) == {}
