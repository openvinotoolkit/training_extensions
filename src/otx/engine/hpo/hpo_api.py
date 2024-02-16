# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Components to run HPO."""

from __future__ import annotations

import dataclasses
import logging
import time
from functools import partial
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable

import torch
from lightning.pytorch.cli import OptimizerCallable

from otx.core.config.hpo import HpoConfig
from otx.core.types.task import OTXTaskType
from otx.hpo import HyperBand, run_hpo_loop
from otx.utils.utils import get_decimal_point, get_using_dot_delimited_key, remove_matched_files

from .hpo_trial import run_hpo_trial
from .utils import find_trial_file, get_best_hpo_weight, get_hpo_weight_dir

if TYPE_CHECKING:
    from otx.engine.engine import Engine
    from otx.hpo.hpo_base import HpoBase

logger = logging.getLogger(__name__)

AVAILABLE_HP_NAME_MAP = {
    "data.config.train_subset.batch_size": "datamodule.config.train_subset.batch_size",
    "optimizer": "optimizer.keywords",
    "scheduler": "scheduler.keywords",
}


def execute_hpo(
    engine: Engine,
    max_epochs: int,
    hpo_config: HpoConfig | None = None,
    progress_update_callback: Callable[[int | float], None] | None = None,
    **train_args,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Execute HPO.

    Args:
        engine (Engine): engine instnace.
        max_epochs (int): max epochs to train.
        hpo_config (HpoConfig | None, optional): Configuration for HPO.
        progress_update_callback (Callable[[int | float], None] | None, optional):
            callback to update progress. If it's given, it's called with progress every second. Defaults to None.

    Returns:
        tuple[dict[str, Any] | None, Path | None]:
            best hyper parameters and model weight trained with best hyper parameters. If it doesn't exist,
            return None.
    """
    if engine.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:  # type: ignore[has-type]
        logger.warning("Zero shot visual prompting task doesn't support HPO.")
        return None, None

    hpo_workdir = Path(engine.work_dir) / "hpo"
    hpo_workdir.mkdir(exist_ok=True)
    hpo_configurator = HPOConfigurator(
        engine,
        max_epochs,
        hpo_workdir,
        hpo_config,
    )
    if (hpo_algo := hpo_configurator.get_hpo_algo()) is None:
        logger.warning("HPO is skipped.")
        return None, None

    if progress_update_callback is not None:
        Thread(target=_update_hpo_progress, args=[progress_update_callback, hpo_algo], daemon=True).start()

    run_hpo_loop(
        hpo_algo,
        partial(
            run_hpo_trial,
            hpo_workdir=hpo_workdir,
            engine=engine,
            max_epochs=max_epochs,
            **_adjust_train_args(train_args),
        ),
        "gpu" if torch.cuda.is_available() else "cpu",
    )

    best_trial = hpo_algo.get_best_config()
    if best_trial is None:
        best_config = None
        best_hpo_weight = None
    else:
        best_config = best_trial["configuration"]
        if (trial_file := find_trial_file(hpo_workdir, best_trial["id"])) is not None:
            best_hpo_weight = get_best_hpo_weight(get_hpo_weight_dir(hpo_workdir, best_trial["id"]), trial_file)

    hpo_algo.print_result()
    _remove_unused_model_weights(hpo_workdir, best_hpo_weight)

    return best_config, best_hpo_weight


class HPOConfigurator:
    """HPO configurator. Prepare a configuration and provide an HPO algorithm based on the configuration.

    Args:
        engine (Engine): engine instance.
        max_epoch (int): max epochs to train.
        hpo_workdir (Path | None, optional): HPO work directory. Defaults to None.
        hpo_config (HpoConfig | None, optional): Configuration for HPO.
    """

    def __init__(
        self,
        engine: Engine,
        max_epoch: int,
        hpo_workdir: Path | None = None,
        hpo_config: HpoConfig | None = None,
    ) -> None:
        self._engine = engine
        self._max_epoch = max_epoch
        self._hpo_workdir = hpo_workdir if hpo_workdir is not None else Path(engine.work_dir) / "hpo"
        self.hpo_config: dict[str, Any] = hpo_config  # type: ignore[assignment]

    @property
    def hpo_config(self) -> dict[str, Any]:
        """Configuration for HPO algorithm."""
        return self._hpo_config

    @hpo_config.setter
    def hpo_config(self, hpo_config: HpoConfig | None) -> None:
        train_dataset_size = len(self._engine.datamodule.subsets["train"])
        val_dataset_size = len(self._engine.datamodule.subsets["val"])

        self._hpo_config: dict[str, Any] = {  # default setting
            "save_path": str(self._hpo_workdir),
            "num_full_iterations": self._max_epoch,
            "full_dataset_size": train_dataset_size,
            "non_pure_train_ratio": val_dataset_size / (train_dataset_size + val_dataset_size),
            "asynchronous_bracket": True,
            "asynchronous_sha": (torch.cuda.device_count() != 1),
        }

        if hpo_config is not None:
            self._hpo_config.update(
                {key: val for key, val in dataclasses.asdict(hpo_config).items() if val is not None},
            )

        if "search_space" not in self._hpo_config:
            self._hpo_config["search_space"] = self._get_default_search_space()
        else:
            self._align_hp_name(self._hpo_config["search_space"])

        if (  # align batch size to train set size
            "datamodule.config.train_subset.batch_size" in self._hpo_config["search_space"]
            and self._hpo_config["search_space"]["datamodule.config.train_subset.batch_size"]["max"]
            > train_dataset_size
        ):
            logger.info(
                "Max value of batch size in HPO search space is lower than train dataset size. "
                "Decrease it to train dataset size.",
            )
            self._hpo_config["search_space"]["datamodule.config.train_subset.batch_size"]["max"] = train_dataset_size

        self._remove_wrong_search_space(self._hpo_config["search_space"])

        if "prior_hyper_parameters" not in self._hpo_config:  # default hyper parameters are tried first
            self._hpo_config["prior_hyper_parameters"] = {
                hp: get_using_dot_delimited_key(hp, self._engine)
                for hp in self._hpo_config["search_space"].keys()  # noqa: SIM118
            }

    def _get_default_search_space(self) -> dict[str, Any]:
        """Set learning rate and batch size as search space."""
        search_space = {}

        if isinstance(self._engine.optimizer, list):
            for i, optimizer in enumerate(self._engine.optimizer):
                search_space[f"optimizer.{i}.keywords.lr"] = self._make_lr_search_space(optimizer)
        elif isinstance(self._engine.optimizer, OptimizerCallable):
            search_space["optimizer.keywords.lr"] = self._make_lr_search_space(self._engine.optimizer)

        cur_bs = self._engine.datamodule.config.train_subset.batch_size
        search_space["datamodule.config.train_subset.batch_size"] = {
            "type": "qloguniform",
            "min": cur_bs // 2,
            "max": cur_bs * 2,
            "step": 2,
        }

        return search_space

    @staticmethod
    def _make_lr_search_space(optimizer: OptimizerCallable) -> dict[str, Any]:
        cur_lr = optimizer.keywords["lr"]  # type: ignore[union-attr]
        min_lr = cur_lr / 10
        return {
            "type": "qloguniform",
            "min": min_lr,
            "max": min(cur_lr * 10, 0.1),
            "step": 10 ** -get_decimal_point(min_lr),
        }

    @staticmethod
    def _align_hp_name(search_space: dict[str, Any]) -> None:
        for hp_name in list(search_space.keys()):
            for valid_hp in AVAILABLE_HP_NAME_MAP:
                if valid_hp in hp_name:
                    new_hp_name = hp_name.replace(valid_hp, AVAILABLE_HP_NAME_MAP[valid_hp])
                    search_space[new_hp_name] = search_space.pop(hp_name)
                    break
            else:
                error_msg = (
                    "Given hyper parameter can't be optimized by HPO. "
                    f"Please choose one from {','.join(AVAILABLE_HP_NAME_MAP)}."
                )
                raise ValueError(error_msg)

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

    return train_args


def _remove_unused_model_weights(hpo_workdir: Path, best_hpo_weight: Path | None = None) -> None:
    remove_matched_files(hpo_workdir, "*.ckpt", best_hpo_weight)
