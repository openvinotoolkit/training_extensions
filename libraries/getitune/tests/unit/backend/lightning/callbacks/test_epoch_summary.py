# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``EpochSummary`` callback."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

from getitune.backend.lightning.callbacks.epoch_summary import EpochSummary, _scalar

_LOGGER = "getitune.backend.lightning.callbacks.epoch_summary"


def _trainer(
    metrics: dict[str, Any],
    *,
    sanity_checking: bool = False,
    current_epoch: int = 0,
    max_epochs: int | None = 200,
) -> Trainer:
    """Minimal trainer-shaped object the callback can consume.

    Cast to ``Trainer`` to satisfy static checkers; the callback only reads a
    handful of attributes so a ``SimpleNamespace`` is sufficient at runtime.
    """
    return cast(
        "Trainer",
        SimpleNamespace(
            sanity_checking=sanity_checking,
            current_epoch=current_epoch,
            max_epochs=max_epochs,
            callback_metrics=metrics,
        ),
    )


_PL_MODULE = cast("LightningModule", None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (torch.tensor(0.5), 0.5),
        (torch.tensor([1.0, 2.0]), None),  # multi-element tensors are skipped, not crashed on
        (3.14, 3.14),
        ("not-a-number", None),
    ],
)
def test_scalar_conversion(value: Any, expected: float | None) -> None:  # noqa: ANN401
    result = _scalar(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_renders_known_metrics_in_declared_order(caplog: pytest.LogCaptureFixture) -> None:
    """Whitelisted metrics render in ``DEFAULT_KEYS`` order with the configured precision.

    Also verifies that non-whitelisted keys, multi-element tensors, and non-numeric values
    are silently skipped so the callback works for every task without per-task wiring.
    """
    cb = EpochSummary()
    trainer = _trainer(
        {
            "train/total_loss": torch.tensor(-0.5290),
            "val/PCK": torch.tensor(0.0021),
            "lr": torch.tensor(0.001),
            "val/per_class": torch.tensor([0.1, 0.2]),  # not scalar -> skipped
            "val/note": "warming-up",  # non-numeric -> skipped
            "custom/metric": 1.0,  # not in whitelist -> skipped
        },
        current_epoch=3,
    )

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        cb.on_validation_epoch_end(trainer, pl_module=_PL_MODULE)

    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "epoch   4/200" in msg
    assert "train/total_loss=-0.5290" in msg
    assert "val/PCK=0.0021" in msg
    assert "lr=0.0010" in msg
    # Order follows DEFAULT_KEYS, not dict insertion order.
    assert msg.index("train/total_loss") < msg.index("val/PCK") < msg.index("lr")
    for skipped in ("val/per_class", "val/note", "custom/metric"):
        assert skipped not in msg


def test_skips_during_sanity_check(caplog: pytest.LogCaptureFixture) -> None:
    """Lightning's pre-training sanity validation must not produce output."""
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        EpochSummary().on_validation_epoch_end(
            _trainer({"val/PCK": torch.tensor(0.0)}, sanity_checking=True),
            pl_module=_PL_MODULE,
        )
    assert caplog.records == []


def test_emits_placeholder_when_no_known_metrics(caplog: pytest.LogCaptureFixture) -> None:
    """Always emit a line so the log shows training is alive even before metrics flush."""
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        EpochSummary().on_validation_epoch_end(_trainer({}), pl_module=_PL_MODULE)
    assert "(no whitelisted metrics)" in caplog.records[0].getMessage()


def test_header_without_max_epochs(caplog: pytest.LogCaptureFixture) -> None:
    """``max_epochs`` is optional in Lightning; header must still render cleanly."""
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        EpochSummary().on_validation_epoch_end(
            _trainer({"val/accuracy": torch.tensor(0.5)}, current_epoch=12, max_epochs=None),
            pl_module=_PL_MODULE,
        )
    msg = caplog.records[0].getMessage()
    assert "epoch  13 |" in msg
    assert "/None" not in msg


def test_custom_keys_and_decimals(caplog: pytest.LogCaptureFixture) -> None:
    """The metric whitelist and precision are configurable; defaults are not auto-merged."""
    cb = EpochSummary(keys=("val/custom",), decimals=2)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        cb.on_validation_epoch_end(
            _trainer({"val/custom": torch.tensor(0.123456), "val/PCK": torch.tensor(0.99)}),
            pl_module=_PL_MODULE,
        )
    msg = caplog.records[0].getMessage()
    assert "val/custom=0.12" in msg
    assert "val/PCK" not in msg


def test_runs_inside_lightning_trainer(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    """End-to-end: the callback fires exactly once per real validation epoch."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        callbacks=[EpochSummary()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        trainer.fit(BoringModel())

    assert len(caplog.records) == 2
    assert "epoch   1/2" in caplog.records[0].getMessage()
    assert "epoch   2/2" in caplog.records[1].getMessage()
