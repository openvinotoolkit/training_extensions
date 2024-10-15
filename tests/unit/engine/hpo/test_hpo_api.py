# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for HPO API utility functions."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from otx.core.config.hpo import HpoConfig
from otx.core.optimizer.callable import OptimizerCallableSupportHPO
from otx.core.schedulers import LinearWarmupSchedulerCallable, SchedulerCallableSupportHPO
from otx.engine.hpo import hpo_api as target_file
from otx.engine.hpo.hpo_api import (
    HPOConfigurator,
    _adjust_train_args,
    _remove_unused_model_weights,
    _update_hpo_progress,
    execute_hpo,
)

if TYPE_CHECKING:
    from pathlib import Path

HPO_NAME_MAP: dict[str, str] = {
    "lr": "model.optimizer_callable.optimizer_kwargs.lr",
    "bs": "datamodule.train_subset.batch_size",
}


@pytest.fixture()
def engine_work_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def dataset_size() -> int:
    return 32


@pytest.fixture()
def default_bs() -> int:
    return 8


@pytest.fixture()
def default_lr() -> float:
    return 0.001


@pytest.fixture()
def mock_engine(engine_work_dir: Path, dataset_size: int, default_bs: int, default_lr: float) -> MagicMock:
    engine = MagicMock()
    engine.work_dir = engine_work_dir
    engine.datamodule.subsets = {engine.datamodule.train_subset.subset_name: range(dataset_size)}
    engine.datamodule.train_subset.batch_size = default_bs
    engine.model.optimizer_callable = MagicMock(spec=OptimizerCallableSupportHPO)
    engine.model.optimizer_callable.lr = default_lr
    engine.model.optimizer_callable.optimizer_kwargs = {"lr": default_lr}
    return engine


@pytest.fixture()
def mock_hpo_algo() -> MagicMock:
    hpo_algo = MagicMock()
    hpo_algo.get_best_config.return_value = {"configuration": "best_config", "id": "best_id"}
    return hpo_algo


@pytest.fixture()
def mock_hpo_configurator(mocker, mock_hpo_algo: MagicMock) -> HPOConfigurator:
    hpo_configurator = MagicMock()
    hpo_configurator.get_hpo_algo.return_value = mock_hpo_algo
    mocker.patch.object(target_file, "HPOConfigurator", return_value=hpo_configurator)
    return hpo_configurator


@pytest.fixture()
def mock_run_hpo_loop(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "run_hpo_loop")


@pytest.fixture()
def mock_thread(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "Thread")


@pytest.fixture()
def mock_best_hpo_weight(tmp_path) -> Path:
    model_weight_path = tmp_path / "best_model_weight.pth"
    model_weight = {
        "callbacks": {
            "ModelCheckpoint": {
                "best_model_path": "fake_value",
                "kth_best_model_path": "fake_value",
                "best_k_models": "fake_value",
                "dirpath": "fake_value",
            },
        },
    }
    torch.save(model_weight, model_weight_path)
    return model_weight_path


@pytest.fixture()
def mock_get_best_hpo_weight(mocker, mock_best_hpo_weight) -> MagicMock:
    return mocker.patch.object(target_file, "get_best_hpo_weight", return_value=mock_best_hpo_weight)


@pytest.fixture()
def mock_find_trial_file(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "find_trial_file")


@pytest.fixture()
def hpo_config() -> HpoConfig:
    return HpoConfig(metric_name="val/accuracy", callbacks_to_exclude="UselessCallback")


@pytest.fixture()
def mock_progress_update_callback() -> MagicMock:
    return MagicMock()


class UsefullCallback:
    pass


class UselessCallback:
    pass


@pytest.fixture()
def mock_callback() -> list:
    return [UsefullCallback(), UselessCallback()]


def test_execute_hpo(
    mock_engine: MagicMock,
    hpo_config: HpoConfig,
    engine_work_dir: Path,
    mock_run_hpo_loop: MagicMock,
    mock_thread: MagicMock,
    mock_hpo_configurator: HPOConfigurator,
    mock_hpo_algo: MagicMock,
    mock_get_best_hpo_weight: MagicMock,
    mock_find_trial_file: MagicMock,
    mock_progress_update_callback: MagicMock,
    mock_callback: list,
):
    hpo_config.progress_update_callback = mock_progress_update_callback
    best_config, best_hpo_weight = execute_hpo(
        engine=mock_engine,
        max_epochs=10,
        hpo_config=hpo_config,
        callbacks=mock_callback,
    )

    # check hpo workdir exists
    assert (engine_work_dir / "hpo").exists()
    assert (engine_work_dir / "hpo" / "best_hp.json").exists()
    # check a case where progress_update_callback exists
    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs["target"] == _update_hpo_progress
    assert mock_thread.call_args.kwargs["daemon"] is True
    mock_thread.return_value.start.assert_called_once()
    # check whether run_hpo_loop is called well
    mock_run_hpo_loop.assert_called_once()
    assert mock_run_hpo_loop.call_args.args[0] == mock_hpo_algo
    # check UselessCallback is excluded
    for callback in mock_run_hpo_loop.call_args.args[1].keywords["callbacks"]:
        assert not isinstance(callback, UselessCallback)
    # check origincal callback lists isn't changed.
    assert len(mock_callback) == 2
    # print_result is called after HPO is done
    mock_hpo_algo.print_result.assert_called_once()
    # best_config and best_hpo_weight are returned well
    assert best_config == "best_config"
    assert best_hpo_weight is not None
    model_weight = torch.load(best_hpo_weight)
    assert model_weight["callbacks"]["ModelCheckpoint"]["best_model_path"] == best_hpo_weight
    for key in ["kth_best_model_path", "best_k_models", "dirpath"]:
        assert key not in model_weight["callbacks"]["ModelCheckpoint"]


class TestHPOConfigurator:
    def test_init(self, mock_engine: MagicMock, hpo_config: HpoConfig):
        HPOConfigurator(mock_engine, 10, hpo_config)

    def test_hpo_config(
        self,
        mock_engine: MagicMock,
        hpo_config: HpoConfig,
        dataset_size: int,
        default_lr: float,
        default_bs: int,
    ):
        max_epochs = 10
        hpo_configurator = HPOConfigurator(mock_engine, max_epochs, hpo_config=hpo_config)
        hpo_config = hpo_configurator.hpo_config

        # check default hpo config is set well
        assert hpo_config["save_path"] == str(mock_engine.work_dir / "hpo")
        assert hpo_config["num_full_iterations"] == max_epochs
        assert hpo_config["full_dataset_size"] == dataset_size
        # check search_space is set automatically
        assert hpo_config["search_space"] is not None
        # check max range of batch size isn't bigger than dataset size
        assert hpo_config["search_space"][HPO_NAME_MAP["bs"]]["max"] <= dataset_size
        # check current hyper parameter will be tested first
        assert hpo_config["prior_hyper_parameters"] == {HPO_NAME_MAP["lr"]: default_lr, HPO_NAME_MAP["bs"]: default_bs}

    def test_get_default_search_space(self, mock_engine: MagicMock, hpo_config: HpoConfig):
        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)
        search_sapce = hpo_configurator._get_default_search_space()

        for hp_name in HPO_NAME_MAP.values():
            assert hp_name in search_sapce

    def test_get_default_search_space_bs1(self, mock_engine: MagicMock, hpo_config: HpoConfig):
        """Check batch sizes search space is set as [1, 2] if default bs is 1."""
        mock_engine.datamodule.train_subset.batch_size = 1
        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)
        search_sapce = hpo_configurator._get_default_search_space()

        for hp_name in HPO_NAME_MAP.values():
            assert hp_name in search_sapce
        assert search_sapce[HPO_NAME_MAP["bs"]]["min"] == 1
        assert search_sapce[HPO_NAME_MAP["bs"]]["max"] == 2

    def test_align_lr_bs_name(self, mock_engine: MagicMock, hpo_config: HpoConfig, dataset_size):
        """Check learning rate and batch size names are aligned well."""
        search_space = {
            "model.optimizer.lr": {
                "type": "loguniform",
                "min": 0.0001,
                "max": 0.1,
            },
            "data.train_subset.batch_size": {
                "type": "quniform",
                "min": 2,
                "max": 512,
                "step": 1,
            },
        }
        hpo_config.search_space = search_space

        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)

        for new_name in HPO_NAME_MAP.values():
            assert new_name in hpo_configurator.hpo_config["search_space"]
        # check max range of batch size isn't bigger than dataset size
        assert hpo_configurator.hpo_config["search_space"][HPO_NAME_MAP["bs"]]["max"] <= dataset_size

    def test_align_scheduler_callable_support_hpo_name(self, mock_engine: MagicMock, hpo_config: HpoConfig):
        """Check scheduler name is aligned well if class of scheduler is SchedulerCallableSupportHPO."""
        mock_engine.model.scheduler_callable = MagicMock(spec=SchedulerCallableSupportHPO)
        mock_engine.model.scheduler_callable.factor = 0.001
        mock_engine.model.scheduler_callable.scheduler_kwargs = {"factor": 0.001}
        search_space = {
            "model.scheduler.factor": {
                "type": "loguniform",
                "min": 0.0001,
                "max": 0.1,
            },
        }
        hpo_config.search_space = search_space

        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)

        assert "model.scheduler_callable.scheduler_kwargs.factor" in hpo_configurator.hpo_config["search_space"]

    def test_align_linear_warmup_scheduler_callable_name(self, mock_engine: MagicMock, hpo_config: HpoConfig):
        """Check scheduler name is aligned well if class of scheduler is LinearWarmupSchedulerCallable."""
        scheduler_callable = MagicMock(spec=LinearWarmupSchedulerCallable)
        scheduler_callable.num_warmup_steps = 0.001
        main_scheduler_callable = MagicMock()
        main_scheduler_callable.factor = 0.001
        main_scheduler_callable.scheduler_kwargs = {"factor": 0.001}
        scheduler_callable.main_scheduler_callable = main_scheduler_callable
        mock_engine.model.scheduler_callable = scheduler_callable
        search_space = {
            "model.scheduler.num_warmup_steps": {
                "type": "loguniform",
                "min": 0.0001,
                "max": 0.1,
            },
            "model.scheduler.main_scheduler_callable.factor": {
                "type": "loguniform",
                "min": 0.0001,
                "max": 0.1,
            },
        }
        hpo_config.search_space = search_space

        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)

        assert "model.scheduler_callable.num_warmup_steps" in hpo_configurator.hpo_config["search_space"]
        assert (
            "model.scheduler_callable.main_scheduler_callable.scheduler_kwargs.factor"
            in hpo_configurator.hpo_config["search_space"]
        )

    def test_remove_wrong_search_space(self, mock_engine: MagicMock, hpo_config: HpoConfig):
        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)
        wrong_search_space = {
            "wrong_choice": {
                "type": "choice",
                "choice_list": [],  # choice shouldn't be empty
            },
            "wrong_quniform": {
                "type": "quniform",
                "min": 2,
                "max": 3,  # max should be larger than min + step
                "step": 2,
            },
        }
        hpo_configurator._remove_wrong_search_space(wrong_search_space)
        assert wrong_search_space == {}

    def test_search_space_file(self, mock_engine: MagicMock, hpo_config: HpoConfig, dataset_size, tmp_path):
        """Check search space is set well if search space file is given."""
        search_space = {
            "model.optimizer.lr": {
                "type": "loguniform",
                "min": 0.0001,
                "max": 0.1,
            },
        }

        search_space_file = tmp_path / "search_space.yaml"
        with search_space_file.open("w") as f:
            yaml.dump(search_space, f)

        hpo_config.search_space = str(search_space_file)

        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)

        assert hpo_configurator.hpo_config["search_space"][HPO_NAME_MAP["lr"]] == search_space["model.optimizer.lr"]
        assert len(hpo_configurator.hpo_config["search_space"].keys()) == 1

    def test_get_hpo_algo(self, mocker, mock_engine: MagicMock, hpo_config: HpoConfig):
        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)
        mock_hyper_band = mocker.patch.object(target_file, "HyperBand")
        hpo_configurator.get_hpo_algo()

        mock_hyper_band.assert_called_once()
        assert mock_hyper_band.call_args.kwargs == hpo_configurator.hpo_config

    @pytest.fixture()
    def max_bs_for_memory(self, default_bs) -> int:
        return default_bs + 2

    @pytest.fixture()
    def mock_adapt_batch_size(self, mocker, max_bs_for_memory) -> MagicMock:
        def func(engine, not_increase: bool = True, *args, **kwargs) -> None:
            origin_bs = engine.datamodule.train_subset.batch_size
            if not not_increase:
                engine.datamodule.train_subset.batch_size = max_bs_for_memory
                engine.model.optimizer_callable.optimizer_kwargs["lr"] *= sqrt(max_bs_for_memory / origin_bs)

        return mocker.patch.object(target_file, "adapt_batch_size", side_effect=func)

    @pytest.mark.parametrize("adapt_mode", ["Safe", "Full"])
    def test_adapt_bs_search_space_max_val(
        self,
        mock_engine: MagicMock,
        hpo_config: HpoConfig,
        mock_adapt_batch_size: MagicMock,
        max_bs_for_memory: int,
        default_bs: int,
        adapt_mode: str,
    ):
        origin_lr = mock_engine.model.optimizer_callable.optimizer_kwargs["lr"]
        hpo_config.adapt_bs_search_space_max_val = adapt_mode
        expected_bs = default_bs * 2 if adapt_mode == "Safe" else max_bs_for_memory

        hpo_configurator = HPOConfigurator(mock_engine, 10, hpo_config)

        assert (
            hpo_configurator.hpo_config["search_space"]["datamodule.train_subset.batch_size"]["max"] == expected_bs
        )  # check batch size is adapted
        mock_adapt_batch_size.assert_called_once()
        assert mock_adapt_batch_size.call_args.args[1] == (adapt_mode != "Full")
        assert mock_engine.model.optimizer_callable.optimizer_kwargs["lr"] == origin_lr  # check lr isn't changed

    @pytest.mark.parametrize("adapt_mode", ["Safe", "Full"])
    def test_adapt_bs_search_space_max_val_wo_bs(
        self,
        mock_engine: MagicMock,
        hpo_config: HpoConfig,
        mock_adapt_batch_size: MagicMock,
        adapt_mode: str,
    ):
        search_space = {
            "model.optimizer.lr": {
                "type": "loguniform",
                "min": 0.0001,
                "max": 0.1,
            },
        }
        hpo_config.search_space = search_space
        hpo_config.adapt_bs_search_space_max_val = adapt_mode

        HPOConfigurator(mock_engine, 10, hpo_config)
        # check _adapt_batch_size_search_space isn't called if search space doesn't include batch size
        mock_adapt_batch_size.assert_not_called()


def test_update_hpo_progress(mocker, mock_progress_update_callback: MagicMock):
    mock_hpo_algo = MagicMock()
    mock_hpo_algo.is_done.side_effect = [False, False, False, True]
    progress_arr = [0.3, 0.6, 1]
    mock_hpo_algo.get_progress.side_effect = progress_arr
    mocker.patch.object(target_file, "time")

    _update_hpo_progress(mock_progress_update_callback, mock_hpo_algo)

    mock_progress_update_callback.assert_called()
    for i in range(3):
        assert mock_progress_update_callback.call_args_list[i].args[0] == pytest.approx(progress_arr[i] * 100)


def test_adjust_train_args():
    new_train_args = _adjust_train_args(
        {
            "self": "self",
            "run_hpo": "run_hpo",
            "kwargs": {
                "kwargs_1": "kwargs_1",
                "kwargs_2": "kwargs_2",
            },
        },
    )

    assert "self" not in new_train_args
    assert "run_hpo" not in new_train_args
    assert "kwargs" not in new_train_args
    assert "kwargs_1" in new_train_args
    assert "kwargs_2" in new_train_args


@pytest.fixture()
def mock_hpo_workdir(tmp_path: Path) -> Path:
    (tmp_path / "1.ckpt").touch()
    sub_dir = tmp_path / "a"
    sub_dir.mkdir()
    (sub_dir / "2.ckpt").touch()
    return tmp_path


def test_remove_unused_model_weights(mock_hpo_workdir: Path):
    best_weight = mock_hpo_workdir / "3.ckpt"
    best_weight.touch()

    _remove_unused_model_weights(mock_hpo_workdir, best_weight)

    ckpt_files = list(mock_hpo_workdir.rglob("*.ckpt"))
    assert len(ckpt_files) == 1
    assert ckpt_files[0] == best_weight
