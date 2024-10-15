# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Components to run HPO."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from copy import copy
from functools import partial
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
import yaml
from lightning import Callback

from otx.core.config.hpo import HpoConfig
from otx.core.optimizer.callable import OptimizerCallableSupportHPO
from otx.core.schedulers import LinearWarmupSchedulerCallable, SchedulerCallableSupportHPO
from otx.core.types.device import DeviceType
from otx.core.types.task import OTXTaskType
from otx.engine.adaptive_bs import adapt_batch_size
from otx.hpo import HyperBand, run_hpo_loop
from otx.utils.utils import (
    get_decimal_point,
    get_using_dot_delimited_key,
    is_xpu_available,
    remove_matched_files,
)

from .hpo_trial import run_hpo_trial
from .utils import find_trial_file, get_best_hpo_weight, get_callable_args_name, get_hpo_weight_dir, get_metric

if TYPE_CHECKING:
    from lightning.pytorch.cli import OptimizerCallable

    from otx.engine.engine import Engine
    from otx.hpo.hpo_base import HpoBase

logger = logging.getLogger(__name__)


def execute_hpo(
    engine: Engine,
    max_epochs: int,
    hpo_config: HpoConfig,
    callbacks: list[Callback] | Callback | None = None,
    **train_args,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Execute HPO.

    Args:
        engine (Engine): engine instnace.
        max_epochs (int): max epochs to train.
        hpo_config (HpoConfig): Configuration for HPO.
        callbacks (list[Callback] | Callback | None, optional): callbacks used during training. Defaults to None.

    Returns:
        tuple[dict[str, Any] | None, Path | None]:
            best hyper parameters and model weight trained with best hyper parameters. If it doesn't exist,
            return None.
    """
    if engine.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:  # type: ignore[has-type]
        msg = "Zero shot visual prompting task doesn't support HPO."
        raise RuntimeError(msg)
    if "anomaly.padim" in str(type(engine.model)).lower():
        msg = "Padim doesn't need HPO."
        raise RuntimeError(msg)

    engine.model.patch_optimizer_and_scheduler_for_hpo()

    hpo_workdir = Path(engine.work_dir) / "hpo"
    hpo_workdir.mkdir(parents=True, exist_ok=True)
    hpo_configurator = HPOConfigurator(
        engine=engine,
        max_epochs=max_epochs,
        hpo_config=hpo_config,
        hpo_workdir=hpo_workdir,
        callbacks=callbacks,
        train_args=train_args,
    )
    if (
        train_args.get("adaptive_bs", None) == "Full"
        and "datamodule.train_subset.batch_size" in hpo_configurator.hpo_config["search_space"]
    ):
        logger.info("Because adaptive_bs is set as Full, batch size is excluded from HPO.")
        hpo_configurator.hpo_config["search_space"].pop("datamodule.train_subset.batch_size")

    if (hpo_algo := hpo_configurator.get_hpo_algo()) is None:
        logger.warning("HPO is skipped.")
        return None, None

    if hpo_config.progress_update_callback is not None:
        Thread(target=_update_hpo_progress, args=[hpo_config.progress_update_callback, hpo_algo], daemon=True).start()

    if hpo_config.callbacks_to_exclude is not None and callbacks is not None:
        if isinstance(hpo_config.callbacks_to_exclude, str):
            hpo_config.callbacks_to_exclude = [hpo_config.callbacks_to_exclude]
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]

        callbacks = copy(callbacks)
        callback_names = [callback.__class__.__name__ for callback in callbacks]
        callback_idx_to_exclude = [
            callback_names.index(cb_name) for cb_name in hpo_config.callbacks_to_exclude if cb_name in callback_names
        ]
        sorted(callback_idx_to_exclude, reverse=True)
        for idx in callback_idx_to_exclude:
            callbacks.pop(idx)

    run_hpo_loop(
        hpo_algo,
        partial(
            run_hpo_trial,
            hpo_workdir=hpo_workdir,
            engine=engine,
            max_epochs=max_epochs,
            callbacks=callbacks,
            metric_name=hpo_config.metric_name,
            **_adjust_train_args(train_args),
        ),
        _get_resource_type() if engine.device.accelerator == DeviceType.auto else engine.device.accelerator,  # type: ignore[arg-type]
        num_parallel_trial=hpo_configurator.hpo_config["num_workers"],
    )

    best_trial = hpo_algo.get_best_config()

    best_config = None
    best_hpo_weight = None
    if best_trial is not None:
        best_config = best_trial["configuration"]
        if (trial_file := find_trial_file(hpo_workdir, best_trial["id"])) is not None:
            best_hpo_weight = get_best_hpo_weight(get_hpo_weight_dir(hpo_workdir, best_trial["id"]), trial_file)
            if best_hpo_weight is not None:
                _update_model_ckpt_callback(best_hpo_weight)

    hpo_algo.print_result()
    _remove_unused_model_weights(hpo_workdir, best_hpo_weight)

    if best_config is not None:
        with (hpo_workdir / "best_hp.json").open("w") as f:
            json.dump(best_config, f)

    return best_config, best_hpo_weight


class HPOConfigurator:
    """HPO configurator. Prepare a configuration and provide an HPO algorithm based on the configuration.

    Args:
        engine (Engine): engine instance.
        max_epochs (int): max epochs to train.
        hpo_config (HpoConfig): Configuration for HPO.
        hpo_workdir (Path | None, optional): HPO work directory. Defaults to None.
        callbacks (list[Callback] | Callback | None, optional): Callbacks used during training. Defaults to None.
        train_args (dict | None, optional): Additional train arguments. It's used for adapt_batch_size_search_space.
    """

    def __init__(
        self,
        engine: Engine,
        max_epochs: int,
        hpo_config: HpoConfig,
        hpo_workdir: Path | None = None,
        callbacks: list[Callback] | Callback | None = None,
        train_args: dict | None = None,
    ) -> None:
        self._engine = engine
        self._max_epochs = max_epochs
        self._hpo_workdir = hpo_workdir if hpo_workdir is not None else Path(engine.work_dir) / "hpo"
        self._callbacks = callbacks
        self._train_args = train_args if train_args is not None else {}
        self.hpo_config: dict[str, Any] = hpo_config  # type: ignore[assignment]

    @property
    def hpo_config(self) -> dict[str, Any]:
        """Configuration for HPO algorithm."""
        return self._hpo_config

    @hpo_config.setter
    def hpo_config(self, hpo_config: HpoConfig) -> None:
        train_dataset_size = len(
            self._engine.datamodule.subsets[self._engine.datamodule.train_subset.subset_name],
        )

        if hpo_config.metric_name is None:
            if self._callbacks is None:
                msg = (
                    "HPOConfigurator can't find the metric because callback doesn't exist. "
                    "Please set hpo_config.metric_name."
                )
                raise RuntimeError(msg)
            hpo_config.metric_name = get_metric(self._callbacks)

        if "loss" in hpo_config.metric_name and hpo_config.mode == "max":
            logger.warning(
                f"Because metric for HPO is {hpo_config.metric_name}, hpo_config.mode is changed from max to min.",
            )
            hpo_config.mode = "min"

        self._hpo_config: dict[str, Any] = {  # default setting
            "save_path": str(self._hpo_workdir),
            "num_full_iterations": self._max_epochs,
            "full_dataset_size": train_dataset_size,
        }

        hb_arg_names = get_callable_args_name(HyperBand)
        self._hpo_config.update(
            {
                key: val
                for key, val in dataclasses.asdict(hpo_config).items()
                if val is not None and key in hb_arg_names
            },
        )

        if "search_space" not in self._hpo_config:
            self._hpo_config["search_space"] = self._get_default_search_space()
        else:
            self._align_search_space()

        if hpo_config.adapt_bs_search_space_max_val != "None":
            if "datamodule.train_subset.batch_size" not in self._hpo_config["search_space"]:
                logger.warning("Batch size isn't included for HPO. 'adapt_batch_size_search_space' is ignored.")
            else:
                self._adapt_batch_size_search_space(hpo_config.adapt_bs_search_space_max_val)

        if (  # align batch size to train set size
            "datamodule.train_subset.batch_size" in self._hpo_config["search_space"]
            and self._hpo_config["search_space"]["datamodule.train_subset.batch_size"]["max"] > train_dataset_size
        ):
            logger.info(
                "Max value of batch size in HPO search space is lower than train dataset size. "
                "Decrease it to train dataset size.",
            )
            self._hpo_config["search_space"]["datamodule.train_subset.batch_size"]["max"] = train_dataset_size

        self._remove_wrong_search_space(self._hpo_config["search_space"])

        if "prior_hyper_parameters" not in self._hpo_config:  # default hyper parameters are tried first
            self._hpo_config["prior_hyper_parameters"] = {
                hp: get_using_dot_delimited_key(hp, self._engine)
                for hp in self._hpo_config["search_space"].keys()  # noqa: SIM118
            }

    def _get_default_search_space(self) -> dict[str, Any]:
        """Set learning rate and batch size as search space."""
        search_space = {}

        search_space["model.optimizer_callable.optimizer_kwargs.lr"] = self._make_lr_search_space(
            self._engine.model.optimizer_callable,
        )

        cur_bs = self._engine.datamodule.train_subset.batch_size
        search_space["datamodule.train_subset.batch_size"] = {
            "type": "qloguniform",
            "min": cur_bs // 2 if cur_bs != 1 else 1,
            "max": cur_bs * 2 if cur_bs != 1 else 2,
            "step": 2,
        }

        return search_space

    @staticmethod
    def _make_lr_search_space(optimizer_callable: OptimizerCallable) -> dict[str, Any]:
        if not isinstance(optimizer_callable, OptimizerCallableSupportHPO):
            raise TypeError(optimizer_callable)

        cur_lr = optimizer_callable.lr  # type: ignore[attr-defined]
        min_lr = cur_lr / 10
        return {
            "type": "qloguniform",
            "min": min_lr,
            "max": min(cur_lr * 10, 0.1),
            "step": 10 ** -get_decimal_point(min_lr),
        }

    def _align_search_space(self) -> None:
        if isinstance(self._hpo_config["search_space"], (str, Path)):
            search_space_file = Path(self._hpo_config["search_space"])
            if not search_space_file.exists():
                msg = f"{search_space_file} is set to HPO search_space, but it doesn't exist."
                raise FileExistsError(msg)
            with search_space_file.open("r") as f:
                self._hpo_config["search_space"] = yaml.safe_load(f)
        elif not isinstance(self._hpo_config["search_space"], dict):
            msg = "HpoConfig.search_space should be str or dict type."
            raise TypeError(msg)
        self._align_hp_name(self._hpo_config["search_space"])  # type: ignore[arg-type]

    def _align_hp_name(self, search_space: dict[str, Any]) -> None:
        available_hp_name_map: dict[str, Callable[[str], None]] = {
            "data.train_subset.batch_size": lambda hp_name: self._replace_hp_name(
                hp_name,
                "data.train_subset.batch_size",
                "datamodule.train_subset.batch_size",
            ),
            "optimizer": lambda hp_name: self._replace_hp_name(
                hp_name,
                "optimizer",
                "optimizer_callable.optimizer_kwargs",
            ),
            "scheduler": self._align_scheduler_name,
        }

        for hp_name in list(search_space.keys()):
            for valid_hp in available_hp_name_map:
                if valid_hp in hp_name:
                    available_hp_name_map[valid_hp](hp_name)
                    break
            else:
                error_msg = (
                    "Given hyper parameter can't be optimized by HPO. "
                    f"Please choose one from {','.join(available_hp_name_map)}."
                )
                raise ValueError(error_msg)

    def _align_scheduler_name(self, hp_name: str) -> None:
        if isinstance(self._engine.model.scheduler_callable, LinearWarmupSchedulerCallable):
            if "main_scheduler_callable" in hp_name:
                self._replace_hp_name(
                    hp_name,
                    "scheduler.main_scheduler_callable",
                    "scheduler_callable.main_scheduler_callable.scheduler_kwargs",
                )
            else:
                self._replace_hp_name(hp_name, "scheduler", "scheduler_callable")
        elif isinstance(self._engine.model.scheduler_callable, SchedulerCallableSupportHPO):
            self._replace_hp_name(hp_name, "scheduler", "scheduler_callable.scheduler_kwargs")

    def _replace_hp_name(self, ori_hp_name: str, old: str, new: str) -> None:
        new_hp_name = ori_hp_name.replace(old, new)
        self._hpo_config["search_space"][new_hp_name] = self._hpo_config["search_space"].pop(ori_hp_name)

    def _adapt_batch_size_search_space(self, adapt_mode: Literal["Safe", "Full"]) -> None:
        origin_bs = self._engine.datamodule.train_subset.batch_size
        origin_lr = self._engine.model.optimizer_callable.optimizer_kwargs["lr"]  # type: ignore[attr-defined]

        self._engine.datamodule.train_subset.batch_size = \
            self._hpo_config["search_space"]["datamodule.train_subset.batch_size"]["max"]  # fmt: off

        adapt_batch_size(
            self._engine,
            adapt_mode != "Full",
            self._callbacks,
            **self._train_args,
        )

        adapted_bs = self._engine.datamodule.train_subset.batch_size

        self._engine.datamodule.train_subset.batch_size = origin_bs
        self._engine.model.optimizer_callable.optimizer_kwargs["lr"] = origin_lr  # type: ignore[attr-defined]
        self._hpo_config["search_space"]["datamodule.train_subset.batch_size"]["max"] = adapted_bs
        logger.info(f"Max value of batch size search space : {origin_bs} -> {adapted_bs}")

    @staticmethod
    def _remove_wrong_search_space(search_space: dict[str, dict[str, Any]]) -> None:
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

    def get_hpo_algo(self) -> HpoBase | None:
        """Get HPO algorithm based on prepared configuration."""
        if not self.hpo_config["search_space"]:
            logger.warning("There is no hyper parameter to optimize.")
            return None
        return HyperBand(**self.hpo_config)


def _update_hpo_progress(progress_update_callback: Callable[[int | float], None], hpo_algo: HpoBase) -> None:
    while not hpo_algo.is_done():
        progress_update_callback(hpo_algo.get_progress() * 100)
        time.sleep(1)


def _adjust_train_args(train_args: dict[str, Any]) -> dict[str, Any]:
    train_args.update(train_args.pop("kwargs", {}))
    train_args.pop("self", None)
    train_args.pop("run_hpo", None)
    train_args.pop("adaptive_bs", None)

    return train_args


def _remove_unused_model_weights(hpo_workdir: Path, best_hpo_weight: Path | None = None) -> None:
    remove_matched_files(hpo_workdir, "*.ckpt", best_hpo_weight)


def _get_resource_type() -> Literal[DeviceType.cpu, DeviceType.gpu, DeviceType.xpu]:
    if torch.cuda.is_available():
        return DeviceType.gpu
    if is_xpu_available():
        return DeviceType.xpu
    return DeviceType.cpu


def _update_model_ckpt_callback(best_hpo_weight: Path) -> None:
    """Update values in ModelCheckpoint callback.

    Some values of ModelCheckpoint callback have HPO temporary directory value,
    which can make error when best_hpo_weight is resumed.
    To prevent it, change `best_model_path` value and remove unnecessary values.

    Args:
        best_hpo_weight (Path): Best HPO model weight.
    """
    best_weight = torch.load(best_hpo_weight)
    for key, val in list(best_weight["callbacks"].items()):
        if "ModelCheckpoint" in key:
            val["best_model_path"] = best_hpo_weight
            val.pop("kth_best_model_path", None)
            val.pop("best_k_models", None)
            val.pop("dirpath", None)
            break
    else:
        return
    torch.save(best_weight, best_hpo_weight)
