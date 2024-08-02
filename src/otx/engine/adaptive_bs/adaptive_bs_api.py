# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Algorithm to find a proper batch size which is fit to current GPU device."""

from __future__ import annotations

import logging
import os
from functools import partial
from math import sqrt
from typing import TYPE_CHECKING, Any

from lightning import Callback
from lightning.pytorch.loggers.logger import DummyLogger
from torch.cuda import is_available as is_cuda_available

from otx.core.types.task import OTXTaskType
from otx.utils.utils import is_xpu_available

from .bs_search_algo import BsSearchAlgo

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

    from otx.engine.engine import Engine

logger = logging.getLogger(__name__)


def adapt_batch_size(
    engine: Engine,
    not_increase: bool = True,
    callbacks: list[Callback] | Callback | None = None,
    **train_args,
) -> None:
    """Change the actual batch size depending on the current GPU status.

    If not_increase is True, check current batch size is available to GPU and if not, decrease batch size.
    If not_increase is False, increase batch size to use most of GPU memory.

    Args:
        engine (Engine): engine instnace.
        not_increase (bool) : Whether adapting batch size to larger value than default value or not.
        callbacks (list[Callback] | Callback | None, optional): callbacks used during training. Defaults to None.
    """
    if not (is_cuda_available() or is_xpu_available()):
        msg = "Adaptive batch size supports CUDA or XPU."
        raise RuntimeError(msg)
    if engine.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:  # type: ignore[has-type]
        msg = "Zero shot visual prompting task doesn't support adaptive batch size."
        raise RuntimeError(msg)

    engine.model.patch_optimizer_and_scheduler_for_hpo()
    default_bs = engine.datamodule.train_subset.batch_size

    if "ADAPTIVE_BS_FOR_DIST" in os.environ:  # main process of distributed training already executes adapt_batch_size
        new_batch_size = int(os.environ["ADAPTIVE_BS_FOR_DIST"])
        if default_bs != new_batch_size:
            _apply_new_batch_size(engine, new_batch_size)
        return

    train_func = partial(_train_model, engine=engine, callbacks=callbacks, **_adjust_train_args(train_args))
    bs_search_algo = BsSearchAlgo(
        train_func=train_func,
        default_bs=default_bs,
        max_bs=(len(engine.datamodule.subsets[engine.datamodule.train_subset.subset_name]) // engine.device.devices),
    )
    if not_increase:
        new_batch_size = bs_search_algo.auto_decrease_batch_size()
    else:
        new_batch_size = bs_search_algo.find_big_enough_batch_size()

    if engine.device.devices != 1:
        os.environ["ADAPTIVE_BS_FOR_DIST"] = str(new_batch_size)

    if default_bs != new_batch_size:
        origin_lr = engine.model.optimizer_callable.optimizer_kwargs["lr"]  # type: ignore[attr-defined]
        _apply_new_batch_size(engine, new_batch_size)
        msg = (
            "Adapting batch size is done.\n"
            f"Batch size is adapted : {default_bs} -> {new_batch_size}\n"
            f"learning rate is adapted : {origin_lr} -> {engine.model.optimizer_callable.optimizer_kwargs['lr']}"  # type: ignore[attr-defined]
        )
        logger.info(msg)
    else:
        logger.info("Adapting batch size is done. Batch size isn't changed.")


def _adjust_train_args(train_args: dict[str, Any]) -> dict[str, Any]:
    train_args.update(train_args.pop("kwargs", {}))
    train_args.pop("self", None)
    train_args.pop("run_hpo", None)
    train_args.pop("adaptive_bs")
    return train_args


def _train_model(bs: int, engine: Engine, callbacks: list[Callback] | Callback | None = None, **train_args) -> None:
    if bs <= 0:
        msg = f"Batch size should be greater than 0, but {bs} is given."
        raise ValueError(msg)
    if engine.device.devices != 1:  # TODO(Eunwoo): Need to change after device api is updated
        engine._cache.update(devices=1)  # noqa: SLF001

    engine.datamodule.train_subset.batch_size = bs
    engine.train(callbacks=_register_callback(callbacks), **train_args)


def _register_callback(callbacks: list[Callback] | Callback | None = None) -> list[Callback]:
    if isinstance(callbacks, Callback):
        callbacks = [callbacks]
    elif callbacks is None:
        callbacks = []
    callbacks.append(BatchSizeFinder())
    return callbacks


class BatchSizeFinder(Callback):
    """This callback makes trainer run specified iteration and exit.

    Args:
        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs, however in practice a few are needed.
    """

    def __init__(
        self,
        steps_per_trial: int = 3,
    ) -> None:
        self._steps_per_trial = steps_per_trial

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None) -> None:
        """Check current stage is fit."""
        if stage != "fit":
            msg = "Adaptive batch size supports only training."
            raise RuntimeError(msg)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run steps_per_trial iterations and exit."""
        _scale_batch_reset_params(trainer, self._steps_per_trial)
        _try_loop_run(trainer)


def _try_loop_run(trainer: Trainer) -> None:
    loop = trainer._active_loop  # noqa: SLF001
    if loop is None:
        msg = "There is no active loop."
        raise RuntimeError(msg)
    loop.restarting = False
    loop.run()


def _scale_batch_reset_params(trainer: Trainer, steps_per_trial: int) -> None:
    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []

    loop = trainer._active_loop  # noqa: SLF001
    if loop is None:
        msg = "There is no active loop."
        raise RuntimeError(msg)
    if trainer.fit_loop.epoch_loop.max_steps == -1:  # epoch based loop
        trainer.fit_loop.max_epochs = 1
        trainer.limit_train_batches = steps_per_trial
    else:  # iter based loop
        trainer.fit_loop.epoch_loop.max_steps = steps_per_trial
        trainer.limit_train_batches = 1.0
    if trainer.limit_val_batches != 0:
        trainer.limit_val_batches = steps_per_trial


def _apply_new_batch_size(engine: Engine, new_batch_size: int) -> None:
    origin_bs = engine.datamodule.train_subset.batch_size
    if new_batch_size == origin_bs:
        return
    engine.datamodule.train_subset.batch_size = new_batch_size
    engine.model.optimizer_callable.optimizer_kwargs["lr"] *= sqrt(new_batch_size / origin_bs)  # type: ignore[attr-defined]
