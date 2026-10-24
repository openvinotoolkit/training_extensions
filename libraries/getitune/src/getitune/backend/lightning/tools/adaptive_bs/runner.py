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
from torch.cuda import is_available as is_cuda_available

from getitune.backend.lightning.callbacks import BatchSizeFinder
from getitune.utils.device import is_xpu_available

from .algorithm import BsSearchAlgo

if TYPE_CHECKING:
    from getitune.backend.lightning.engine import LightningEngine

logger = logging.getLogger(__name__)


def adapt_batch_size(
    engine: LightningEngine,
    not_increase: bool = True,
    **train_args,
) -> None:
    """Change the actual batch size depending on the current GPU status.

    If not_increase is True, check current batch size is available to GPU and if not, decrease batch size.
    If not_increase is False, increase batch size to use most of GPU memory.

    Args:
        engine (LightningEngine): engine instnace.
        not_increase (bool) : Whether adapting batch size to larger value than default value or not.
    """
    if not (is_cuda_available() or is_xpu_available()):
        msg = "Adaptive batch size supports only CUDA or XPU."
        logger.warning(msg)
        return

    engine.model.patch_optimizer_and_scheduler_for_adaptive_bs()
    default_bs = engine.datamodule.train_subset.batch_size

    if "ADAPTIVE_BS_FOR_DIST" in os.environ:  # main process of distributed training already executes adapt_batch_size
        new_batch_size = int(os.environ["ADAPTIVE_BS_FOR_DIST"])
        if default_bs != new_batch_size:
            _apply_new_batch_size(engine, new_batch_size)
        return

    train_func = partial(_train_model, engine=engine, **_adjust_train_args(train_args))
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
    train_args.pop("adaptive_bs", None)
    train_args.pop("callbacks", None)
    return train_args


def _train_model(bs: int, engine: LightningEngine, **train_args) -> None:
    if bs <= 0:
        msg = f"Batch size should be greater than 0, but {bs} is given."
        raise ValueError(msg)
    if engine.device.devices != 1:  # TODO(Eunwoo): Need to change after device api is updated
        engine._cache.update(devices=1)  # noqa: SLF001

    engine.datamodule.train_subset.batch_size = bs
    engine.datamodule.val_subset.batch_size = bs
    engine.datamodule.test_subset.batch_size = bs
    train_args["adaptive_bs"] = "None"
    logger.info("Running training trial with bs = %s ...", bs)
    engine.train(callbacks=_register_callback(), **train_args)


def _register_callback(callbacks: list[Callback] | Callback | None = None) -> list[Callback]:
    if isinstance(callbacks, Callback):
        callbacks = [callbacks]
    elif callbacks is None:
        callbacks = []
    callbacks.append(BatchSizeFinder())
    return callbacks


def _apply_new_batch_size(engine: LightningEngine, new_batch_size: int) -> None:
    origin_bs = engine.datamodule.train_subset.batch_size
    engine.datamodule.train_subset.batch_size = new_batch_size
    engine.datamodule.val_subset.batch_size = new_batch_size
    engine.datamodule.test_subset.batch_size = new_batch_size
    new_lr = engine.model.optimizer_callable.optimizer_kwargs["lr"] * sqrt(new_batch_size / origin_bs)  # type: ignore[attr-defined]
    logger.info(
        "new batch size = %s with learning rate = %s is set for training and validation.", new_batch_size, new_lr
    )
    engine.model.optimizer_callable.optimizer_kwargs["lr"] = new_lr  # type: ignore[attr-defined]
