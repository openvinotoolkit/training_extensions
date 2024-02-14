# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Components to run HPO."""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable

import torch
import yaml

from otx.hpo import HyperBand, run_hpo_loop
from otx.utils.utils import get_decimal_point, get_using_comma_seperated_key

from .hpo_trial import run_hpo_trial
from .utils import find_trial_file, get_best_hpo_weight, get_hpo_weight_dir, remove_unused_files

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
    hpo_time_ratio: int = 4,
    hpo_cfg_file: str | Path | None = None,
    progress_update_callback: Callable[[int | float], None] | None = None,
    **train_args,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Execute HPO.

    Args:
        engine (Engine): engine instnace.
        max_epochs (int): max epochs to train.
        hpo_time_ratio (int, optional): time ratio to use for HPO compared to training time. Defaults to 4.
        hpo_cfg_file (str | Path | None, optional):
            HPO configuration file. If it isn't given, default setting will be used.
        progress_update_callback (Callable[[int | float], None] | None, optional):
            callback to update progress. If it's given, it's called with progress every second. Defaults to None.

    Returns:
        tuple[dict[str, Any] | None, Path | None]:
            best hyper parameters and model weight trained with best hyper parameters. If it doesn't exist,
            return None.
    """
    hpo_workdir = Path(engine.work_dir) / "hpo"
    hpo_workdir.mkdir(exist_ok=True)
    hpo_configurator = HPOConfigurator(
        engine,
        max_epochs,
        hpo_time_ratio,
        hpo_workdir,
        hpo_cfg_file,
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
        hpo_time_ratio (int): time ratio to use for HPO compared to training time. Defaults to 4.
        hpo_workdir (Path | None, optional): HPO work directory. Defaults to None.
        hpo_cfg_file (str | Path | None, optional):
            HPO configuration file. If it isn't given, default setting will be used. Defaults to None.
    """

    def __init__(
        self,
        engine: Engine,
        max_epoch: int,
        hpo_time_ratio: int = 4,
        hpo_workdir: Path | None = None,
        hpo_cfg_file: str | Path | None = None,
    ) -> None:
        self._engine = engine
        self._max_epoch = max_epoch
        self._hpo_time_ratio = hpo_time_ratio
        self._hpo_workdir = hpo_workdir if hpo_workdir is not None else Path(engine.work_dir) / "hpo"
        self._hpo_cfg_file: Path | None = Path(hpo_cfg_file) if isinstance(hpo_cfg_file, str) else hpo_cfg_file
        self._hpo_config: dict[str, Any] | None = None

    @property
    def hpo_config(self) -> dict[str, Any]:
        """Configuration for HPO algorithm."""
        if self._hpo_config is None:
            train_dataset_size = len(self._engine.datamodule.subsets["train"])
            val_dataset_size = len(self._engine.datamodule.subsets["val"])

            hpo_config = {
                "save_path" : str(self._hpo_workdir),
                "num_full_iterations" : self._max_epoch,
                "full_dataset_size" : train_dataset_size,
                "non_pure_train_ratio" : val_dataset_size / (train_dataset_size + val_dataset_size),
                "expected_time_ratio" : self._hpo_time_ratio,
                "asynchronous_bracket" : True,
                "asynchronous_sha" : (torch.cuda.device_count() != 1),
            }

            if self._hpo_cfg_file is not None:
                if not self._hpo_cfg_file.exists():
                    logger.warning("HPO configuration file doesn't exist.")
                else:
                    with self._hpo_cfg_file.open("r") as f:
                        hpo_config.update(yaml.safe_load(f))

            if "search_space" not in hpo_config:
                hpo_config["search_space"] = self._get_default_search_space()
            else:
                self._align_hp_name(hpo_config["search_space"])

            if (  # align batch size to train set size
                "datamodule.config.train_subset.batch_size" in hpo_config["search_space"] and
                hpo_config["search_space"]["datamodule.config.train_subset.batch_size"]["max"] > train_dataset_size
            ):
                hpo_config["search_space"]["datamodule.config.train_subset.batch_size"]["max"] = train_dataset_size

            self._remove_wrong_search_space(hpo_config["search_space"])

            if "prior_hyper_parameters" not in hpo_config:  # default hyper parameters are tried first
                hpo_config["prior_hyper_parameters"] = {
                    hp: get_using_comma_seperated_key(hp, self._engine) for hp in hpo_config["search_space"].keys()
                }

            self._hpo_config = hpo_config

        return self._hpo_config

    def _get_default_search_space(self) -> dict[str, Any]:
        """Set learning rate and batch size as search space."""
        search_space = {}

        cur_lr = self._engine.optimizer.keywords["lr"]
        min_lr = cur_lr / 10
        search_space["optimizer.keywords.lr"] = {
            "type": "qloguniform",
            "min": min_lr,
            "max": min(cur_lr * 10, 0.1),
            "step": 10 ** -get_decimal_point(min_lr),
        }

        cur_bs = self._engine.datamodule.config.train_subset.batch_size
        search_space["datamodule.config.train_subset.batch_size"] = {
            "type": "qloguniform",
            "min": cur_bs // 2,
            "max": cur_bs * 2,
            "step": 2,
        }

        return search_space

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

    def get_hpo_algo(self) -> HpoBase:
        """Get HPO algorithm based on prepared configuration."""
        if not self.hpo_config["search_space"]:
            logger.warning("There is no hyper parameter to optimize.")
            return None
        return HyperBand(**self.hpo_config)


def _update_hpo_progress(progress_update_callback: Callable[[int | float], None], hpo_algo: HpoBase) -> None:
    while not hpo_algo.is_done():
        progress_update_callback(hpo_algo.get_progress() * 100)
        time.sleep(1)


def _remove_unused_model_weights(hpo_workdir: Path, best_hpo_weight: Path | None = None) -> None:
    remove_unused_files(hpo_workdir, "*.ckpt", best_hpo_weight)
