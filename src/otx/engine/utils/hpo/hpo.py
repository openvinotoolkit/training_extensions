# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Components to execute HPO."""

from __future__ import annotations

import time
import json
import logging
import yaml
from threading import Thread
from pathlib import Path
from functools import partial
from typing import TYPE_CHECKING, Any, Callable
from tempfile import TemporaryDirectory

import torch
from lightning import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from otx.algo.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling
from otx.utils.utils import get_using_comma_seperated_key, set_using_comma_seperated_key, get_decimal_point
from otx.hpo import run_hpo_loop, HyperBand, TrialStatus

if TYPE_CHECKING:
    from lightning import Trainer, LightningModule

    from otx.engine.engine import Engine
    from otx.hpo.hpo_base import HpoBase

logger = logging.getLogger(__name__)

AVAILABLE_HP_NAME_MAP = {
    "data.config.train_subset.batch_size" : "datamodule.config.train_subset.batch_size",
    "optimizer" : "optimizer.keywords",
    "scheduler" : "scheduler.keywords",
}


def execute_hpo(
    engine: Engine,
    max_epochs: int,
    hpo_time_ratio: int = 4,
    hpo_cfg_file: str | Path | None = None,
    progress_update_callback: Callable[[int | float], None] | None = None,
    **train_args
) -> None:
    hpo_workdir = Path(engine.work_dir) / "hpo"
    hpo_workdir.mkdir(exist_ok=True)
    hpo_configurator = HPOConfigurator(
        engine,
        max_epochs,
        hpo_time_ratio,
        hpo_workdir,
        hpo_cfg_file
    )
    hpo_algo = hpo_configurator.get_hpo_algo()

    if progress_update_callback is not None:
        Thread(target=_update_hpo_progress, args=[progress_update_callback, hpo_algo], daemon=True).start()

    run_hpo_loop(
        hpo_algo,
        partial(
            run_hpo_trial,
            hpo_workdir=hpo_workdir,
            engine=engine,
            max_epochs=max_epochs,
            **train_args,
        ),
        "gpu" if torch.cuda.is_available() else "cpu",
    )

    best_trial = hpo_algo.get_best_config()
    if best_trial is None:
        best_config = None
        best_hpo_weight = None
    else:
        best_config = best_trial["configuration"]
        if (trial_file := _find_trial_file(hpo_workdir, best_trial["id"])) is not None:
            best_hpo_weight = _get_best_hpo_weight(_get_hpo_weight_dir(hpo_workdir, best_trial["id"]), trial_file)

    hpo_algo.print_result()
    _remove_unused_model_weights(hpo_workdir, best_hpo_weight)

    return best_config, best_hpo_weight


class HPOConfigurator:
    def __init__(
        self,
        engine: Engine,
        max_epoch: int,
        hpo_time_ratio: int,
        hpo_workdir: Path | None = None,
        hpo_cfg_file: str | Path | None = None,
    ):
        self._engine = engine
        self._max_epoch = max_epoch
        self._hpo_time_ratio = hpo_time_ratio
        self._hpo_workdir = hpo_workdir if hpo_workdir is not None else Path(engine.work_dir) / "hpo"
        self._hpo_cfg_file: Path | None = Path(hpo_cfg_file) if isinstance(hpo_cfg_file, str) else hpo_cfg_file
        self._hpo_config: dict[str, Any] | None = None

    @property
    def hpo_config(self) -> dict[str, Any]:
        if self._hpo_config is None:
            if self._hpo_cfg_file is None:
                hpo_config = {}
            elif not self._hpo_cfg_file.exists():
                hpo_config = {}
                logger.warning("HPO configuration file doesn't exist.")
            else:
                with self._hpo_cfg_file.open("r") as f:
                    hpo_config = yaml.safe_load(f)

            train_dataset_size = len(self._engine.datamodule.subsets['train'])
            val_dataset_size = len(self._engine.datamodule.subsets['val'])

            hpo_config["save_path"] = str(self._hpo_workdir)
            hpo_config["num_full_iterations"] = self._max_epoch
            hpo_config["full_dataset_size"] = train_dataset_size
            hpo_config["non_pure_train_ratio"] = val_dataset_size / (train_dataset_size + val_dataset_size)
            hpo_config["expected_time_ratio"] = self._hpo_time_ratio
            hpo_config["asynchronous_bracket"] = True
            hpo_config["asynchronous_sha"] = torch.cuda.device_count() != 1,

            if "search_space" not in hpo_config:  # optimize lr and bs as default
                hpo_config["search_space"] = self._get_default_search_space(train_dataset_size)
            else:
                self._align_hp_name(hpo_config["search_space"])
            self._remove_wrong_search_space(hpo_config["search_space"])

            if "prior_hyper_parameters" not in hpo_config:  # default hp is tried at first
                hpo_config["prior_hyper_parameters"] = {
                    hp : get_using_comma_seperated_key(hp, self._engine) for hp in hpo_config["search_space"].keys()
                }

            self._hpo_config = hpo_config

        return self._hpo_config

    def _get_default_search_space(self, train_dataset_size: int) -> dict[str, Any]:
        """Set learning rate and batch size as search space."""
        search_space = {}

        cur_lr = self._engine.optimizer.keywords["lr"]
        min_lr = cur_lr / 10
        search_space["optimizer.keywords.lr"] = {
            "type" : "qloguniform",
            "min": min_lr,
            "max": min(cur_lr * 10, 0.1),
            "step": 10 ** -get_decimal_point(min_lr)
        }

        cur_bs = self._engine.datamodule.config.train_subset.batch_size
        search_space["datamodule.config.train_subset.batch_size"] = {
            "type" : "qloguniform",
            "min": cur_bs // 2,
            "max": min(cur_bs * 2, train_dataset_size),
            "step": 2,
        }

        return search_space

    @staticmethod
    def _remove_wrong_search_space(search_space: dict[str, dict[str, Any]]):
        for hp_name, config in list(search_space.items()):
            if config["type"] == "choice":
                if not config["choice_list"]:
                    search_space.pop(hp_name)
                    logger.warning(f"choice_list is empty. {hp_name} is excluded from HPO serach space.")
            elif config["max"] < config["min"] + config.get("step", 0):
                search_space.pop(hp_name)
                if "step" in config:
                    reason_to_exclude = "max is smaller than sum of min and step"
                else:
                    reason_to_exclude = "max is smaller than min"
                logger.warning(f"{reason_to_exclude}. {hp_name} is excluded from HPO serach space.")

    @staticmethod
    def _align_hp_name(search_space: dict[str, Any]) -> None:
        for hp_name in list(search_space.keys()):
            for valid_hp in AVAILABLE_HP_NAME_MAP:
                if valid_hp in hp_name:
                    search_space[AVAILABLE_HP_NAME_MAP[hp_name]] = search_space.pop(hp_name)
                    break
            else:
                error_msg = (
                    "Given hyper parameter can't be optimized by HPO. "
                    f"Please choose one from {','.join(AVAILABLE_HP_NAME_MAP)}."
                )
                raise ValueError(error_msg)

    def get_hpo_algo(self):
        if not self.hpo_config["search_space"]:
            logger.warning("There is no hyper parameter to optimize.")
            return None
        return HyperBand(**self.hpo_config)


def _update_hpo_progress(progress_update_callback, hpo_algo: HpoBase):
    """Function for a thread to report a HPO progress regularly.

    Args:
        hpo_algo (HpoBase): HPO algorithm class
    """

    while True:
        if hpo_algo.is_done():
            break
        progress_update_callback(hpo_algo.get_progress() * 100)
        time.sleep(1)


def run_hpo_trial(
    hp_config: dict[str, Any],
    report_func: Callable,
    hpo_workdir: Path,
    engine: Engine,
    callbacks: list[Callback] | Callback | None = None,
    **train_args,
) -> None:
    trial_id = hp_config["id"]
    hpo_weight_dir = _get_hpo_weight_dir(hpo_workdir, trial_id)

    _set_trial_hyper_parameter(hp_config["configuration"], engine, train_args)

    if (checkpoint := _find_last_weight(hpo_weight_dir)) is not None:
        engine.checkpoint = checkpoint
        train_args["resume"] = True

    callbacks = _register_hpo_callback(report_func, callbacks)
    _set_to_validate_every_epoch(callbacks, train_args)

    with TemporaryDirectory(prefix="OTX-HPO-") as temp_dir:
        _change_work_dir(callbacks, engine, temp_dir)
        engine.train(callbacks=callbacks, **train_args)

        _keep_best_and_last_weight(Path(temp_dir), hpo_workdir, trial_id)

    report_func(0, 0, done=True)


def _get_hpo_weight_dir(hpo_workdir: Path, trial_id: str) -> Path:
    hpo_weight_dir: Path = hpo_workdir / "weight" / trial_id
    if not hpo_weight_dir.exists():
        hpo_weight_dir.mkdir(parents=True)
    return hpo_weight_dir


def _set_trial_hyper_parameter(hyper_parameter: dict[str, Any], engine: Engine, train_args: dict[str, Any]) -> None:
    train_args["max_epochs"] = round(hyper_parameter.pop("iterations"))
    update_hyper_parameter(engine, hyper_parameter)


def _find_last_weight(weight_dir: Path) -> Path | None:
    return _find_file_recursively(weight_dir, "last.ckpt")

    
def _find_file_recursively(directory: Path, file_name: str) -> Path | None:
    if found_file := list(directory.rglob(file_name)):
        return found_file[0]
    return None


def _register_hpo_callback(report_func: Callable, callbacks: list[Callback] | Callback | None) -> list[Callback]:
    if isinstance(callbacks, Callback):
        callbacks = [callbacks]
    elif callbacks is None:
        callbacks = []
    callbacks.append(HPOCallback(report_func, get_metric(callbacks)))
    return callbacks


class HPOCallback(Callback):
    """Timer for logging iteration time for train, val, and test phases."""

    def __init__(self, report_func: Callable[[float | int, float | int], TrialStatus], metric: str):
        super().__init__()
        self._report_func = report_func
        self.metric = metric

    def on_train_epoch_end(self, trainer: Trainer, pl_module_: LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        score = trainer.callback_metrics.get(self.metric)
        if (score := trainer.callback_metrics.get(self.metric)) is not None:
            if self._report_func(score=score.item(), progress=epoch) == TrialStatus.STOP:
                trainer.should_stop = True


def _set_to_validate_every_epoch(callbacks: list[Callback], train_args: dict[str, Any]) -> None:
    for callback in callbacks:
        if isinstance(callback, AdaptiveTrainScheduling):
            callback.max_interval = 1
            break
    else:        
        train_args["check_val_every_n_epoch"] = 1


def _change_work_dir(callbacks: list[Callback], engine: Engine, work_dir: str) -> None:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            callback.dirpath = work_dir
            break
    engine.work_dir = work_dir


def _keep_best_and_last_weight(trial_work_dir: Path, hpo_workdir: Path, trial_id: str):
    last_weight = _find_last_weight(trial_work_dir)
    if (trial_file := _find_trial_file(hpo_workdir, trial_id)) is not None:
        best_weight = _get_best_hpo_weight(trial_work_dir, trial_file)
    else:
        best_weight = None

    for ckpt_file in [best_weight, last_weight]:
        if ckpt_file is not None:
            ckpt_file.replace(hpo_workdir / "weight" / trial_id / ckpt_file.name)


def _find_trial_file(hpo_dir: Path, trial_id: str) -> Path | None:
    return _find_file_recursively(hpo_dir, f"{trial_id}.json")


def _get_best_hpo_weight(weight_dir: Path, trial_file: Path) -> Path | None:
    """Get best model weight path of the HPO trial.

    Args:
        weight_dir (Path): directory where model weeights are saved.
        trial_file (Path): json format trial file which stores trial record.

    Returns:
        Optional[str]: best HPO model weight
    """
    if not trial_file.exists():
        return None

    with trial_file.open("r") as f:
        trial_output = json.load(f)

    best_epochs = []
    best_score = None
    for eph, score in trial_output["score"].items():
        eph = str(int(eph) - 1)  # lightning uses index starting from 0
        if best_score is None:
            best_score = score
            best_epochs.append(eph)
        elif best_score < score:
            best_score = score
            best_epochs = [eph]
        elif best_score == score:
            best_epochs.append(eph)

    for best_epoch in best_epochs:
        if (best_weight_path := _find_file_recursively(weight_dir, f"epoch_*{best_epoch}.ckpt")) is not None:
            return best_weight_path

    return None


def _remove_unused_model_weights(hpo_workdir: Path, best_hpo_weight: Path | None = None):
    for weight in hpo_workdir.rglob("*.ckpt"):
        if weight != best_hpo_weight:
            weight.unlink()


def get_metric(callbacks: list[Callback]) -> str:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback.monitor
    error_msg = "Failed to find a metric. There is no ModelCheckpoint in callback list."
    raise RuntimeError(error_msg)


def update_hyper_parameter(engine: Engine, hyper_parameter: dict[str, Any]) -> None:
    for key, val in hyper_parameter.items():
        set_using_comma_seperated_key(key, val, engine)
