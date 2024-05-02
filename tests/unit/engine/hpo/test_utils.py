# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for HPO API utility functions."""

import json
from unittest.mock import MagicMock

import pytest
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from otx.engine.hpo.utils import (
    find_trial_file,
    get_best_hpo_weight,
    get_callable_args_name,
    get_hpo_weight_dir,
    get_metric,
)


@pytest.fixture()
def trial_id():
    return "1"


@pytest.fixture()
def trial_file(tmp_path, trial_id):
    trial_file = tmp_path / "hpo" / "trial" / "0" / f"{trial_id}.json"
    trial_file.parent.mkdir(parents=True)
    with trial_file.open("w") as f:
        json.dump(
            {
                "id": trial_id,
                "rung": 0,
                "configuration": {"lr": 0.1, "iterations": 10},
                "score": {"1": 0.1, "2": 0.2, "3": 0.5, "4": 0.4, "5": 0.5},
            },
            f,
        )
    return trial_file


def test_find_trial_file(tmp_path, trial_file, trial_id):
    assert trial_file == find_trial_file(tmp_path, trial_id)


def test_find_trial_file_file_not_exist(tmp_path, trial_file):
    assert find_trial_file(tmp_path, "2") is None


@pytest.fixture()
def hpo_weight_dir(tmp_path, trial_id):
    weight_dir = tmp_path / "weight" / trial_id
    weight_dir.mkdir(parents=True)
    return weight_dir


def test_get_best_hpo_weight(trial_file, hpo_weight_dir):
    for eph in range(1, 6):
        (hpo_weight_dir / f"epoch_{eph}.ckpt").touch()

    assert hpo_weight_dir / "epoch_4.ckpt" == get_best_hpo_weight(hpo_weight_dir, trial_file)


def test_get_absent_best_hpo_weight(trial_file, hpo_weight_dir):
    assert get_best_hpo_weight(hpo_weight_dir, trial_file) is None


def test_get_hpo_weight_dir(tmp_path, hpo_weight_dir, trial_id):
    assert hpo_weight_dir == get_hpo_weight_dir(tmp_path, trial_id)


def test_get_absent_hpo_weight_dir(tmp_path, hpo_weight_dir, trial_id):
    hpo_weight_dir.rmdir()
    assert hpo_weight_dir == get_hpo_weight_dir(tmp_path, trial_id)
    assert hpo_weight_dir.exists()


def test_get_callable_args_name():
    def func(arg1, arg2) -> None:
        pass

    assert get_callable_args_name(func) == ["arg1", "arg2"]


def test_get_callable_args_name_no_args():
    def func() -> None:
        pass

    assert get_callable_args_name(func) == []


@pytest.fixture()
def mock_model_ckpt_hook() -> MagicMock:
    model_ckpt_hook = MagicMock(spec=ModelCheckpoint)
    model_ckpt_hook.monitor = "val/accuracy"
    return model_ckpt_hook


def test_get_metric(mock_model_ckpt_hook):
    assert get_metric(mock_model_ckpt_hook) == "val/accuracy"


def test_get_metric_list_callback(mock_model_ckpt_hook):
    callbacks = [mock_model_ckpt_hook]
    assert get_metric(callbacks) == "val/accuracy"


def test_get_metric_no_model_ckpt_callback():
    callbacks = [MagicMock()]
    with pytest.raises(RuntimeError, match="Failed to find a metric"):
        get_metric(callbacks)


def test_get_metric_list_monitor_value_none(mock_model_ckpt_hook):
    mock_model_ckpt_hook.monitor = None
    with pytest.raises(ValueError, match="Failed to find a metric"):
        get_metric(mock_model_ckpt_hook)
