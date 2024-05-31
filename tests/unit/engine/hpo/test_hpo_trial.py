# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for HPO API utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from otx.algo.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling
from otx.engine.hpo import hpo_trial as target_file
from otx.engine.hpo.hpo_trial import (
    HPOCallback,
    HPOInitWeightCallback,
    _get_hpo_initial_weight,
    _register_hpo_callback,
    _set_to_validate_every_epoch,
    run_hpo_trial,
    update_hyper_parameter,
)
from otx.engine.hpo.utils import get_hpo_weight_dir
from otx.hpo import TrialStatus
from torch import tensor

if TYPE_CHECKING:
    from lightning import Callback


@pytest.fixture()
def mock_callback1() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_callback2() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_engine(mock_callback1, mock_callback2) -> MagicMock:
    engine = MagicMock()

    def train_side_effect(*args, **kwargs) -> None:
        if isinstance(engine.work_dir, str):
            work_dir = Path(engine.work_dir)
            for i in range(3):
                (work_dir / f"epoch_{i}.ckpt").touch()
            (work_dir / "last.ckpt").write_text("last_ckpt")

    engine.train.side_effect = train_side_effect
    engine._cache.args = {"callbacks": [mock_callback1, mock_callback2]}

    return engine


def test_update_hyper_parameter(mock_engine):
    hyper_parameter = {
        "a.b.c": 1,
        "d.e.f": 2,
    }

    update_hyper_parameter(mock_engine, hyper_parameter)

    assert mock_engine.a.b.c == 1
    assert mock_engine.d.e.f == 2


@pytest.fixture()
def mock_report_func() -> MagicMock:
    return MagicMock()


class TestHPOCallback:
    @pytest.fixture()
    def metric(self) -> str:
        return "metric"

    def test_init(self, mock_report_func, metric):
        HPOCallback(mock_report_func, metric)

    @pytest.fixture()
    def hpo_callback(self, mock_report_func, metric) -> HPOCallback:
        return HPOCallback(mock_report_func, metric)

    def test_on_train_epoch_end(self, hpo_callback: HPOCallback, mock_report_func, metric):
        cur_eph = 10
        score = 0.1

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = cur_eph
        mock_trainer.callback_metrics = {metric: tensor(score)}
        mock_report_func.return_value = TrialStatus.STOP

        hpo_callback.on_train_epoch_end(mock_trainer, MagicMock())

        mock_report_func.assert_called_once_with(pytest.approx(score), cur_eph + 1)
        assert mock_trainer.should_stop is True  # if report_func returns STOP, it should be set to True

    def test_on_train_epoch_end_no_score(self, hpo_callback: HPOCallback, mock_report_func):
        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {}

        hpo_callback.on_train_epoch_end(mock_trainer, MagicMock())

        mock_report_func.assert_not_called()


@pytest.fixture()
def mock_checkpoint_callback() -> MagicMock:
    return MagicMock(spec=ModelCheckpoint)


@pytest.fixture()
def mock_adaptive_schedule_hook() -> MagicMock:
    return MagicMock(spec=AdaptiveTrainScheduling)


@pytest.fixture()
def mock_callbacks(mock_checkpoint_callback, mock_adaptive_schedule_hook) -> list[Callback]:
    return [mock_checkpoint_callback, mock_adaptive_schedule_hook]


@pytest.fixture()
def trial_id() -> str:
    return "0"


@pytest.fixture()
def max_epochs() -> int:
    return 10


@pytest.fixture()
def hp_config(trial_id, max_epochs) -> dict[str, Any]:
    return {
        "id": trial_id,
        "configuration": {
            "iterations": max_epochs,
            "a.b.c": 1,
            "d.e.f": 2,
        },
    }


def test_run_hpo_trial(
    trial_id,
    max_epochs,
    hp_config,
    mocker,
    mock_callbacks,
    mock_report_func,
    tmp_path,
    mock_engine,
    mock_checkpoint_callback,
):
    hpo_weight_dir = get_hpo_weight_dir(tmp_path, trial_id)
    last_weight = hpo_weight_dir / "last.ckpt"  # last checkpoint so far. will be used to resume
    last_weight.write_text("prev_weight")
    best_weight = hpo_weight_dir / "epoch_2.ckpt"
    mocker.patch.object(target_file, "find_trial_file", return_value=Path("fake.json"))
    mocker.patch.object(target_file, "get_best_hpo_weight", return_value=best_weight)

    run_hpo_trial(
        hp_config=hp_config,
        report_func=mock_report_func,
        hpo_workdir=tmp_path,
        engine=mock_engine,
        callbacks=mock_callbacks,
        metric_name="metric",
    )

    train_work_dir = mock_engine.work_dir
    mock_engine.train.assert_called_once()
    # HPOCallback should be added to callback list
    for callback in mock_engine.train.call_args.kwargs["callbacks"]:
        if isinstance(callback, HPOCallback):
            break
    else:
        msg = "There is no HPOCallback in callback list."
        raise AssertionError(msg)
    # check training is resumed if model checkpoint exists
    assert mock_engine.train.call_args.kwargs["checkpoint"] == last_weight
    assert mock_engine.train.call_args.kwargs["resume"] is True
    # check given hyper parameters are set well
    assert mock_engine.train.call_args.kwargs["max_epochs"] == max_epochs
    assert mock_engine.a.b.c == 1
    assert mock_engine.d.e.f == 2
    # check train work directory are changed well
    assert mock_checkpoint_callback.dirpath == train_work_dir
    # check final report is executed well
    mock_report_func.assert_called_once()
    mock_report_func.call_args.kwargs["done"] = True
    # check all model weights in train directory are moved to hpo weight directory
    assert len(list(Path(train_work_dir).rglob("*.ckpt"))) == 0
    # check all model checkpoint are removed except last and best weight
    hpo_weights = list(hpo_weight_dir.rglob("*.ckpt"))
    assert len(hpo_weights) == 2
    assert best_weight in hpo_weights
    assert last_weight in hpo_weights
    assert last_weight.read_text() == "last_ckpt"


def test_run_hpo_trial_wo_hpo_init_weigeht(hp_config, mock_callbacks, mock_report_func, tmp_path, mock_engine):
    """Check if checkpoint is None and HPO initial weight doesn't exist, HPOInitWeightCallback is registered."""
    run_hpo_trial(
        hp_config=hp_config,
        report_func=mock_report_func,
        hpo_workdir=tmp_path,
        engine=mock_engine,
        callbacks=mock_callbacks,
        metric_name="metric",
    )

    callbacks = mock_engine.train.call_args.kwargs["callbacks"]
    hpo_initial_weight_callback_exist = False
    for callback in callbacks:
        if isinstance(callback, HPOInitWeightCallback):
            hpo_initial_weight_callback_exist = True
    assert hpo_initial_weight_callback_exist


def test_run_hpo_trial_w_hpo_init_weigeht(hp_config, mock_callbacks, mock_report_func, tmp_path, mock_engine):
    """Check if checkpoint is None and HPO initial weight exist, the weight is set to checkpoint."""
    init_weight = _get_hpo_initial_weight(tmp_path)
    init_weight.touch()

    run_hpo_trial(
        hp_config=hp_config,
        report_func=mock_report_func,
        hpo_workdir=tmp_path,
        engine=mock_engine,
        callbacks=mock_callbacks,
        metric_name="metric",
    )

    assert init_weight.samefile(mock_engine.train.call_args.kwargs["checkpoint"])


def test_register_hpo_callback(mock_report_func, mock_engine):
    """Check it returns list including only HPOCallback if any callbacks are passed."""
    callabcks = _register_hpo_callback(
        report_func=mock_report_func,
        engine=mock_engine,
        metric_name="metric",
    )
    assert len(callabcks) == 3
    hpo_callback_exist = False
    for callback in callabcks:
        if isinstance(callback, HPOCallback):
            hpo_callback_exist = True
            break
    assert hpo_callback_exist


def test_register_hpo_callback_given_callback(mock_report_func, mock_checkpoint_callback):
    """Check it returns list including HPOCallback if single callback is passed."""
    new_callabcks = _register_hpo_callback(
        report_func=mock_report_func,
        callbacks=mock_checkpoint_callback,
        metric_name="metric",
    )
    assert len(new_callabcks) == 2
    for callback in new_callabcks:
        if isinstance(callback, HPOCallback):
            break
    else:
        msg = "There is no HPOCallback in callback list."
        raise AssertionError(msg)
    assert mock_checkpoint_callback in new_callabcks


def test_register_hpo_callback_given_callbacks_arr(mock_report_func, mock_checkpoint_callback, mock_callbacks):
    """Check it returns list including HPOCallback if callback array is passed."""
    new_callabcks = _register_hpo_callback(
        report_func=mock_report_func,
        callbacks=mock_callbacks,
        metric_name="metric",
    )
    assert len(new_callabcks) == 3
    for callback in new_callabcks:
        if isinstance(callback, HPOCallback):
            break
    else:
        msg = "There is no HPOCallback in callback list."
        raise AssertionError(msg)
    assert mock_checkpoint_callback in new_callabcks


def test_set_to_validate_every_epoch(mock_callbacks, mock_adaptive_schedule_hook):
    """Check AdaptiveTrainScheduling.max_iterval is changed if AdaptiveTrainScheduling is in callback list."""
    train_args = {}
    _set_to_validate_every_epoch(mock_callbacks, train_args)

    assert mock_adaptive_schedule_hook.max_interval == 1
    assert train_args == {}


def test_set_to_validate_every_epoch_no_adap_schedule():
    """Check check_val_every_n_epoch is added to train_args if AdaptiveTrainScheduling isn't in callback list."""
    train_args = {}
    _set_to_validate_every_epoch(callbacks=[], train_args=train_args)

    assert train_args["check_val_every_n_epoch"] == 1
