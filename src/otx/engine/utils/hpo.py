# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Components to execute HPO."""

from __future__ import annotations

import json
import logging
import yaml
from pathlib import Path
from functools import partial
from typing import TYPE_CHECKING, Any, Callable
from tempfile import TemporaryDirectory

import torch
from lightning import Callback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from otx.algo.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling
from otx.core.types.task import OTXTaskType
from otx.utils.utils import get_using_comma_seperated_key, set_using_comma_seperated_key, get_decimal_point
from otx.hpo import run_hpo_loop, HyperBand, TrialStatus

if TYPE_CHECKING:
    from lightning import Trainer, LightningModule

    from otx.engine.engine import Engine

logger = logging.getLogger(__name__)

TASK_METRIC = {
    OTXTaskType.MULTI_CLASS_CLS: "val/accuracy",
    # OTXTaskType.MULTI_LABEL_CLS: "",
    # OTXTaskType.H_LABEL_CLS: "",
    OTXTaskType.DETECTION: "val/map_50",
    # OTXTaskType.ROTATED_DETECTION: "",
    # OTXTaskType.INSTANCE_SEGMENTATION: "",
    OTXTaskType.SEMANTIC_SEGMENTATION: "val/mIoU",
    OTXTaskType.ACTION_CLASSIFICATION: "accuracy",
    # OTXTaskType.ACTION_DETECTION: "",
    # OTXTaskType.VISUAL_PROMPTING: "",
    # OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: "",  # noqa: E501
}

HP_AVAILABLE_TO_OPTIMIZE = (
    "datamodule.config.train_subset.batch_size",
    "optimizer.keywords",
    "scheduler.keywords",
)


def execute_hpo(
    engine: Engine,
    max_epochs: int,
    hpo_time_ratio: int = 4,
    hpo_cfg_file: str | Path | None = None,
    **train_args
):
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
    best_config = hpo_algo.get_best_config()
    if best_config is not None:
        print(best_config["config"])
    hpo_algo.print_result()

    return best_config


def run_hpo_trial(
    hp_config: dict[str, Any],
    report_func: Callable,
    hpo_workdir: Path,
    engine: Engine,
    callbacks: list[Callback] | Callback | None = None,
    **train_args,
):
    train_args["check_val_every_n_epoch"] = 1
    for callback in callbacks:
        if isinstance(callback, AdaptiveTrainScheduling):
            callback.max_interval = 1
        
    train_args["max_epochs"] = round(hp_config["configuration"].pop("iterations"))
    for key, val in hp_config["configuration"].items():
        set_using_comma_seperated_key(key, val, engine)

    trial_id = hp_config["id"]
    hpo_weight_dir: Path = hpo_workdir / "weight" / trial_id
    hpo_weight_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_checkpoint(hpo_weight_dir)
    if checkpoint is not None:
        engine.checkpoint = checkpoint
        train_args["resume"] = True

    hpo_callback = HPOCallback(report_func, TASK_METRIC[engine.task])
    if isinstance(callbacks, list):
        callbacks.append(hpo_callback)
    elif isinstance(callbacks, Callback):
        callbacks = [callbacks, hpo_callback]
    elif callbacks is None:
        callbacks = hpo_callback
    

    with TemporaryDirectory(prefix="OTX-HPO-") as temp_dir:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = temp_dir
                break

        engine.work_dir = temp_dir
        engine.train(callbacks=callbacks, **train_args)
        finilize_hpo_trial(Path(temp_dir), hpo_weight_dir)

        # remove unnecessary weight file
        best_weight = get_best_hpo_weight(hpo_workdir, trial_id)
        if best_weight is not None:
            for model_weight in hpo_weight_dir.glob("epoch_*.ckpt"):
                if model_weight != best_weight:
                    model_weight.unlink()

    report_func(0, 0, done=True)


def find_checkpoint(weight_dir: Path) -> Path | None:
    if last_ckpt := list(weight_dir.glob("last.ckpt")):
        return last_ckpt[0]
    return None

    
def finilize_hpo_trial(work_dir: Path, dest: Path) -> None:
    for ckpt_file in work_dir.rglob("*.ckpt"):
        ckpt_file.replace(dest / ckpt_file.name)



class HPOCallback(Callback):
    """Timer for logging iteration time for train, val, and test phases."""

    def __init__(self, report_func, metric: str):
        super().__init__()
        self._report_func = report_func
        self.metric = metric

    def on_train_epoch_end(self, trainer: Trainer, pl_module_: LightningModule) -> None:
        logs = trainer.callback_metrics
        score = logs.get(self.metric)
        epoch = trainer.current_epoch + 1
        print("#"*100, f"In hpo callback : {logs}")
        if score is not None:
            score = score.item()
            print(f"In hpo callback : {score} /  {epoch}")
            if self._report_func(score=score, progress=epoch) == TrialStatus.STOP:
                trainer.should_stop = True


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
        if self._hpo_config is not None:
            return self._hpo_config

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
        hpo_config["metric"] = TASK_METRIC[self._engine.task]
        hpo_config["expected_time_ratio"] = self._hpo_time_ratio
        hpo_config["asynchronous_bracket"] = True
        hpo_config["asynchronous_sha"] = torch.cuda.device_count() != 1,

        if "search_space" not in hpo_config:  # optimize lr and bs as default
            hpo_config["search_space"] = {}

            cur_lr = self._engine.optimizer.keywords["lr"]
            min_lr = cur_lr / 10
            max_lr = min(cur_lr * 10, 0.1)
            if max_lr > min_lr:
                hpo_config["search_space"]["optimizer.keywords.lr"] = {
                    "param_type" : "qloguniform",
                    "min": min_lr,
                    "max": max_lr,
                    "step": 10 ** -get_decimal_point(min_lr)
                }

            cur_bs = self._engine.datamodule.config.train_subset.batch_size
            min_bs = cur_bs // 2
            max_bs = min(cur_bs * 2, train_dataset_size)
            step = 2
            if max_bs >= min_bs + step:
                hpo_config["search_space"]["datamodule.config.train_subset.batch_size"] = {
                    "param_type" : "qloguniform",
                    "min": min_bs,
                    "max": max_bs,
                    "step": step,
                }
            else:
                for key in hpo_config["search_space"].keys():
                    self._check_hp_valid(key)
        if "prior_hyper_parameters" not in hpo_config:  # default hp is tried at first
            hpo_config["prior_hyper_parameters"] = {
                hp : get_using_comma_seperated_key(hp, self._engine) for hp in hpo_config["search_space"].keys()
            }

        self._hpo_config = hpo_config
        return self._hpo_config

    @staticmethod
    def _check_hp_valid(hp_name: str) -> None:
        for valid_hp in HP_AVAILABLE_TO_OPTIMIZE:
            if valid_hp in hp_name:
                return
        error_msg = f"Given hyper parameter can't be optimized by HPO. Please choose one from {','.join(HP_AVAILABLE_TO_OPTIMIZE)}."
        raise ValueError(error_msg)

    def get_hpo_algo(self):
        if not self.hpo_config["search_space"]:
            logger.warning("There is no hyper parameter to optimize.")
            return None
        return HyperBand(**self.hpo_config)


def get_best_hpo_weight(hpo_dir: Path, trial_id: str) -> Path | None:
    """Get best model weight path of the HPO trial.

    Args:
        hpo_dir (Union[str, Path]): HPO work directory path
        trial_id (Union[str, Path]): trial id

    Returns:
        Optional[str]: best HPO model weight
    """
    trial_output_files = list(hpo_dir.rglob(f"{trial_id}.json"))
    if not trial_output_files:
        return None

    with trial_output_files[0].open("r") as f:
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

    best_weight = None
    for best_epoch in best_epochs:
        best_weight_path = list(hpo_dir.glob(f"weight/{trial_id}/epoch_*{best_epoch}.ckpt"))
        if best_weight_path:
            best_weight = best_weight_path[0]

    return best_weight
