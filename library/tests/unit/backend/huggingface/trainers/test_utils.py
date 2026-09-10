# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``Trainer`` log-history to metrics-CSV writer (G17)."""

from __future__ import annotations

import csv
from pathlib import Path

from getitune.backend.huggingface.trainers.utils import remap_log_key, write_metrics_csv


class TestRemapLogKey:
    def test_epoch_and_step_stay_bare(self) -> None:
        assert remap_log_key("epoch") == "epoch"
        assert remap_log_key("step") == "step"

    def test_eval_prefixed_keys_become_val(self) -> None:
        assert remap_log_key("eval_loss") == "val/loss"
        assert remap_log_key("eval_runtime") == "val/runtime"

    def test_train_prefixed_keys_become_train(self) -> None:
        assert remap_log_key("train_runtime") == "train/runtime"
        assert remap_log_key("train_loss") == "train/loss"

    def test_unprefixed_training_keys_default_to_train(self) -> None:
        assert remap_log_key("loss") == "train/loss"
        assert remap_log_key("grad_norm") == "train/grad_norm"
        assert remap_log_key("learning_rate") == "train/learning_rate"


class TestWriteMetricsCsv:
    def test_writes_one_row_per_log_entry(self, tmp_path: Path) -> None:
        log_history = [
            {"loss": 2.0, "learning_rate": 5e-5, "epoch": 1.0, "step": 4},
            {"eval_loss": 1.5, "epoch": 1.0, "step": 4},
        ]

        csv_path = write_metrics_csv(log_history, tmp_path)

        assert csv_path == tmp_path / "csv" / "version_0" / "metrics.csv"
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert rows[0]["train/loss"] == "2.0"
        assert rows[0]["val/loss"] == ""
        assert rows[1]["val/loss"] == "1.5"
        assert rows[1]["train/loss"] == ""

    def test_columns_are_the_union_across_all_entries(self, tmp_path: Path) -> None:
        log_history = [
            {"loss": 2.0, "epoch": 1.0, "step": 1},
            {"eval_loss": 1.5, "eval_runtime": 0.1, "epoch": 1.0, "step": 1},
            {"train_runtime": 5.0, "train_loss": 1.8, "epoch": 1.0, "step": 1},
        ]

        csv_path = write_metrics_csv(log_history, tmp_path)

        with csv_path.open() as fh:
            fieldnames = csv.DictReader(fh).fieldnames
        assert fieldnames is not None
        assert set(fieldnames) == {
            "train/loss",
            "epoch",
            "step",
            "val/loss",
            "val/runtime",
            "train/runtime",
        }

    def test_empty_log_history_writes_an_empty_csv(self, tmp_path: Path) -> None:
        csv_path = write_metrics_csv([], tmp_path)
        assert csv_path.exists()
        with csv_path.open() as fh:
            assert list(csv.DictReader(fh)) == []
