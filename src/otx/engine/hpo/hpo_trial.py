# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Components to run HPO trial."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable

from lightning import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from otx.algo.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling
from otx.hpo import TrialStatus
from otx.utils.utils import find_file_recursively, remove_matched_files, set_using_dot_delimited_key

from .utils import find_trial_file, get_best_hpo_weight, get_hpo_weight_dir, get_metric

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

    from otx.engine.engine import Engine


def update_hyper_parameter(engine: Engine, hyper_parameter: dict[str, Any]) -> None:
    """Update hyper parameter in the engine."""
    for key, val in hyper_parameter.items():
        set_using_dot_delimited_key(key, val, engine)


class HPOCallback(Callback):
    """HPO callback class which reports a score to HPO algorithm every epoch."""

    def __init__(self, report_func: Callable[[float | int, float | int], TrialStatus], metric: str) -> None:
        super().__init__()
        self._report_func = report_func
        self.metric = metric

    def on_train_epoch_end(self, trainer: Trainer, pl_module_: LightningModule) -> None:
        """Report scores if score exists at the end of each epoch."""
        score = trainer.callback_metrics.get(self.metric)
        if score is not None and self._report_func(score.item(), trainer.current_epoch + 1) == TrialStatus.STOP:
            trainer.should_stop = True


class HPOInitWeightCallback(Callback):
    """Callbacks to save a HPO initial model weight. The weight is used for all trials."""

    def __init__(self, save_path: Path) -> None:
        super().__init__()
        self._save_path = save_path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Save a model weight to save_path."""
        trainer.save_checkpoint(self._save_path)


def run_hpo_trial(
    hp_config: dict[str, Any],
    report_func: Callable[[int | float, int | float, bool], None],
    hpo_workdir: Path,
    engine: Engine,
    callbacks: list[Callback] | Callback | None = None,
    metric_name: str | None = None,
    **train_args,
) -> None:
    """Run HPO trial. After it's done, best weight and last weight are saved for later use.

    Args:
        hp_config (dict[str, Any]): trial's hyper parameter.
        report_func (Callable): function to report score.
        hpo_workdir (Path): HPO work directory.
        engine (Engine): engine instance.
        callbacks (list[Callback] | Callback | None, optional): callbacks used during training. Defaults to None.
        metric_name (str | None, optional):
            metric name to determine trial performance. If it's None, get it from ModelCheckpoint callback.
        train_args: Arugments for 'engine.train'.
    """
    trial_id = hp_config["id"]
    hpo_weight_dir = get_hpo_weight_dir(hpo_workdir, trial_id)

    _set_trial_hyper_parameter(hp_config["configuration"], engine, train_args)

    if (checkpoint := _find_last_weight(hpo_weight_dir)) is not None:
        train_args["checkpoint"] = checkpoint
        train_args["resume"] = True

    callbacks = _register_hpo_callback(report_func, callbacks, engine, metric_name)
    _set_to_validate_every_epoch(callbacks, train_args)

    if train_args.get("checkpoint") is None:
        hpo_initial_weight = _get_hpo_initial_weight(hpo_workdir)
        if hpo_initial_weight.exists():
            train_args["checkpoint"] = hpo_initial_weight
        else:
            callbacks = _register_init_weight_callback(callbacks, hpo_initial_weight)

    with TemporaryDirectory(prefix="OTX-HPO-") as temp_dir:
        _change_work_dir(temp_dir, callbacks, engine)
        engine.train(callbacks=callbacks, **train_args)

        _keep_best_and_last_weight(Path(temp_dir), hpo_workdir, trial_id)

    report_func(0, 0, done=True)  # type: ignore[call-arg]


def _set_trial_hyper_parameter(hyper_parameter: dict[str, Any], engine: Engine, train_args: dict[str, Any]) -> None:
    train_args["max_epochs"] = round(hyper_parameter.pop("iterations"))
    update_hyper_parameter(engine, hyper_parameter)


def _find_last_weight(weight_dir: Path) -> Path | None:
    return find_file_recursively(weight_dir, "last.ckpt")


def _register_hpo_callback(
    report_func: Callable,
    callbacks: list[Callback] | Callback | None = None,
    engine: Engine | None = None,
    metric_name: str | None = None,
) -> list[Callback]:
    if isinstance(callbacks, Callback):
        callbacks = [callbacks]
    elif callbacks is None:
        callbacks = [] if engine is None else engine._cache.args.get("callbacks", [])  # noqa: SLF001
    callbacks.append(HPOCallback(report_func, get_metric(callbacks) if metric_name is None else metric_name))
    return callbacks


def _set_to_validate_every_epoch(callbacks: list[Callback], train_args: dict[str, Any]) -> None:
    for callback in callbacks:
        if isinstance(callback, AdaptiveTrainScheduling):
            callback.max_interval = 1
            break
    else:
        train_args["check_val_every_n_epoch"] = 1


def _get_hpo_initial_weight(hpo_workdir: Path) -> Path:
    return hpo_workdir / "hpo_initial_weight.ckpt"


def _register_init_weight_callback(callbacks: list[Callback], save_path: Path) -> list[Callback]:
    callbacks.append(HPOInitWeightCallback(save_path))
    return callbacks


def _change_work_dir(work_dir: str, callbacks: list[Callback], engine: Engine) -> None:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            callback.dirpath = work_dir
            break
    engine.work_dir = work_dir


def _keep_best_and_last_weight(trial_work_dir: Path, hpo_workdir: Path, trial_id: str) -> None:
    weight_dir = get_hpo_weight_dir(hpo_workdir, trial_id)
    _move_all_ckpt(trial_work_dir, weight_dir)
    if (trial_file := find_trial_file(hpo_workdir, trial_id)) is not None:
        best_weight = get_best_hpo_weight(weight_dir, trial_file)
        remove_matched_files(weight_dir, "epoch_*.ckpt", best_weight)


def _move_all_ckpt(src: Path, dest: Path) -> None:
    for ckpt_file in src.rglob("*.ckpt"):
        shutil.copy(ckpt_file, dest)
        ckpt_file.unlink()
