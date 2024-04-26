# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for HPO API utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from otx.engine.hpo import hpo_api as target_file
from otx.engine.hpo.hpo_api import execute_hpo, HPOConfigurator, _update_hpo_progress, _adjust_train_args, _remove_unused_model_weights
from otx.core.config.hpo import HpoConfig

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def engine_work_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def mock_engine(engine_work_dir) -> MagicMock:
    engine = MagicMock()
    engine.work_dir = engine_work_dir
    engine.datamodule.train_dataloader.return_value = range(10)
    engine.datamodule.config.train_subset.batch_size = 8
    engine.optimizer.keywords = {"lr" : 0.01}
    return engine


@pytest.fixture
def mock_hpo_algo() -> MagicMock:
    hpo_algo = MagicMock()
    hpo_algo.get_best_config.return_value = {"configuration" : "best_config", "id" : "best_id"}
    return hpo_algo

@pytest.fixture
def mock_hpo_configurator(mocker, mock_hpo_algo) -> HPOConfigurator:
    hpo_configurator = MagicMock()
    hpo_configurator.get_hpo_algo.return_value = mock_hpo_algo
    mocker.patch.object(target_file, "HPOConfigurator", return_value=hpo_configurator)
    return hpo_configurator


@pytest.fixture
def mock_run_hpo_loop(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "run_hpo_loop")

    
@pytest.fixture
def mock_thread(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "Thread")

@pytest.fixture
def mock_get_best_hpo_weight(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "get_best_hpo_weight")


@pytest.fixture
def mock_find_trial_file(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "find_trial_file")


def test_execute_hpo(
    mock_engine,
    engine_work_dir,
    mock_run_hpo_loop,
    mock_thread,
    mock_hpo_configurator,
    mock_hpo_algo,
    mock_get_best_hpo_weight,
    mock_find_trial_file,
):
    mock_progress_update_callback = MagicMock()

    best_config, best_hpo_weight = execute_hpo(mock_engine, 10, progress_update_callback=mock_progress_update_callback)

    assert (engine_work_dir / "hpo").exists()  # check hpo workdir exists

    # check a case where progress_update_callback exists
    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs["target"] == _update_hpo_progress
    assert mock_thread.call_args.kwargs["args"][0] == mock_progress_update_callback
    assert mock_thread.call_args.kwargs["daemon"] is True
    mock_thread.return_value.start.assert_called_once()

    # check whether run_hpo_loop is called well
    mock_run_hpo_loop.assert_called_once()
    assert mock_run_hpo_loop.call_args.args[0] == mock_hpo_algo

    mock_hpo_algo.print_result.assert_called_once()
    assert best_config == "best_config"
    assert best_hpo_weight == mock_get_best_hpo_weight.return_value


class TestHPOConfigurator:
    def test_init(self, mock_engine):
        HPOConfigurator(mock_engine, 10)

    def test_hpo_config(self, mock_engine):
        num_trials = 123
        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config=HpoConfig(num_trials=num_trials))
        hpo_config = hpo_configurator.hpo_config
        
        # check default hpo config is set well
        assert hpo_config["save_path"] == str(mock_engine.work_dir / "hpo")
        assert hpo_config["num_full_iterations"] == 10
        assert hpo_config["full_dataset_size"] == 10

        assert hpo_config["num_trials"] == num_trials  # check hpo_config argument is applied well
        assert hpo_config["search_space"] is not None  # check search_space is set automatically
        # check max range of batch size isn't bigger than dataset size
        assert hpo_config["search_space"]["datamodule.config.train_subset.batch_size"]["max"] == 10
        # check current hyper parameter will be tested first
        assert hpo_config["prior_hyper_parameters"] == {'optimizer.keywords.lr': 0.01, 'datamodule.config.train_subset.batch_size': 8}

    def test_get_default_search_space(self, mock_engine):
        hpo_configurator = HPOConfigurator(mock_engine, 10)
        search_sapce = hpo_configurator._get_default_search_space()

        for hp_name in ["optimizer.keywords.lr", "datamodule.config.train_subset.batch_size"]:
            assert hp_name in search_sapce

    def test_align_hp_name(self, mock_engine):
        hpo_configurator = HPOConfigurator(mock_engine, 10)
        search_space = {
            "optimizer.lr" : MagicMock(),
            "scheduler.hp" : MagicMock(),
            "data.config.train_subset.batch_size" : MagicMock()
        }
        hpo_configurator._align_hp_name(search_space)

        for new_name in ["optimizer.keywords.lr", "scheduler.keywords.hp", "datamodule.config.train_subset.batch_size"]:
            assert new_name in search_space

    def test_remove_wrong_search_space(self, mock_engine):
        hpo_configurator = HPOConfigurator(mock_engine, 10)
        wrong_search_space = {
            "wrong_choice" : {
                "type" : "choice",
                "choice_list" : []  # choice shouldn't be empty
            },
            "wrong_quniform" : {
                "type" : "quniform",
                "min" : 2,
                "max" : 3,  # max should be larger than min + step
                "step" : 2
            }
        }
        hpo_configurator._remove_wrong_search_space(wrong_search_space)
        assert wrong_search_space == {}

    def test_get_hpo_algo(self, mocker, mock_engine):
        hpo_configurator = HPOConfigurator(mock_engine, 10)
        mock_hyper_band = mocker.patch.object(target_file, "HyperBand")
        hpo_configurator.get_hpo_algo()

        mock_hyper_band.assert_called_once()
        assert mock_hyper_band.call_args.kwargs == hpo_configurator.hpo_config


def test_update_hpo_progress(mocker):
    mock_hpo_algo = MagicMock()
    mock_hpo_algo.is_done.side_effect = [False, False, False, True]
    progress_arr = [0.3, 0.6, 1]
    mock_hpo_algo.get_progress.side_effect = progress_arr
    mock_progress_update_callback = MagicMock()
    mocker.patch.object(target_file, "time")

    _update_hpo_progress(mock_progress_update_callback, mock_hpo_algo)

    mock_progress_update_callback.assert_called()
    for i in range(3):
        mock_progress_update_callback.call_args_list[i].args[0] == pytest.approx(progress_arr[i] * 100)


def test_adjust_train_args():
    train_args = {
        "self" : "self",
        "run_hpo" : "run_hpo",
        "kwargs" : {
            "kwargs_1" : "kwargs_1",
            "kwargs_2" : "kwargs_2",
        }
    }

    new_train_args = _adjust_train_args(train_args)

    assert "self" not in new_train_args
    assert "run_hpo" not in new_train_args
    assert "kwargs" not in new_train_args
    assert "kwargs_1" in new_train_args
    assert "kwargs_2" in new_train_args
    

def test_remove_unused_model_weights(tmp_path):
    (tmp_path / "1.ckpt").touch()
    sub_dir = tmp_path / "a"
    sub_dir.mkdir()
    (sub_dir / "2.ckpt").touch()
    best_weight = sub_dir / "3.ckpt"
    best_weight.touch()

    _remove_unused_model_weights(tmp_path, best_weight)

    ckpt_files = list(tmp_path.rglob("*.ckpt"))
    assert len(ckpt_files) == 1
    assert ckpt_files[0] == best_weight
