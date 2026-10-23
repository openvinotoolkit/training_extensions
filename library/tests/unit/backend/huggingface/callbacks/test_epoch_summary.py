# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hugging Face per-epoch summary callback."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from getitune.backend.huggingface.callbacks.epoch_summary import HFEpochSummaryCallback

_LOGGER = "getitune.backend.huggingface.callbacks.epoch_summary"


def _state(epoch: float | None = None, *, num_train_epochs: int = 10, is_world_process_zero: bool = True) -> Any:  # noqa: ANN401
    return SimpleNamespace(
        epoch=epoch,
        num_train_epochs=num_train_epochs,
        is_world_process_zero=is_world_process_zero,
    )


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


class TestHFEpochSummaryCallback:
    def test_emits_after_validation_with_mean_train_loss(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = HFEpochSummaryCallback()
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            cb.on_epoch_begin(None, _state(0.0), None)  # type: ignore[arg-type]
            cb.on_log(None, _state(0.5), None, logs={"loss": 2.0, "learning_rate": 1e-3})  # type: ignore[arg-type]
            cb.on_log(None, _state(1.0), None, logs={"loss": 1.0, "learning_rate": 5e-4})  # type: ignore[arg-type]
            cb.on_epoch_end(None, _state(1.0), None)  # type: ignore[arg-type]
            assert _messages(caplog) == []  # waits for validation
            metrics = {"val/map_50": 0.25, "eval_val/map_50": 0.25, "epoch": 1}
            cb.on_log(None, _state(1.0), None, logs=metrics)  # type: ignore[arg-type]
            cb.on_evaluate(None, _state(1.0), None, metrics=metrics)  # type: ignore[arg-type]

        assert _messages(caplog) == ["epoch   1/10 | train/total_loss=1.5000  val/map_50=0.2500  lr=0.0005"]

    def test_flushes_epochs_without_validation(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = HFEpochSummaryCallback()
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            for epoch in (1, 2):
                cb.on_epoch_begin(None, _state(epoch - 1), None)  # type: ignore[arg-type]
                cb.on_log(None, _state(epoch), None, logs={"loss": float(epoch)})  # type: ignore[arg-type]
                cb.on_epoch_end(None, _state(epoch), None)  # type: ignore[arg-type]
            cb.on_train_end(None, _state(2.0), None)  # type: ignore[arg-type]

        assert _messages(caplog) == [
            "epoch   1/10 | train/total_loss=1.0000",
            "epoch   2/10 | train/total_loss=2.0000",
        ]

    def test_silent_on_non_main_process(self, caplog: pytest.LogCaptureFixture) -> None:
        cb = HFEpochSummaryCallback()
        state = _state(1.0, is_world_process_zero=False)
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            cb.on_log(None, state, None, logs={"loss": 1.0})  # type: ignore[arg-type]
            cb.on_epoch_end(None, state, None)  # type: ignore[arg-type]
            cb.on_evaluate(None, state, None, metrics={})  # type: ignore[arg-type]
        assert _messages(caplog) == []


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"loss": nn.functional.mse_loss(self.linear(x).squeeze(-1), y)}


class _ToyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": torch.full((2,), float(index)), "y": torch.tensor(float(index))}


@pytest.mark.parametrize("eval_strategy", ["epoch", "no"])
def test_runs_once_per_epoch_inside_transformers_trainer(
    tmp_path: Any,  # noqa: ANN401
    caplog: pytest.LogCaptureFixture,
    eval_strategy: str,
) -> None:
    """End-to-end: one line per epoch, with or without per-epoch evaluation."""
    args = TrainingArguments(
        output_dir=str(tmp_path),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy=eval_strategy,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to=[],
        label_names=["y"],
        use_cpu=True,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=_ToyModel(),
        args=args,
        train_dataset=_ToyDataset(),
        eval_dataset=_ToyDataset() if eval_strategy == "epoch" else None,
        callbacks=[HFEpochSummaryCallback(keys=("train/total_loss", "val/loss", "lr"))],
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        trainer.train()

    messages = _messages(caplog)
    assert len(messages) == 2
    assert messages[0].startswith("epoch   1/2 | train/total_loss=")
    assert messages[1].startswith("epoch   2/2 | train/total_loss=")
    assert all(("val/loss=" in m) == (eval_strategy == "epoch") for m in messages)
