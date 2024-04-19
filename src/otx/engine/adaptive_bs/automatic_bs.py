"""Algorithm to find a proper batch size which is fit to current GPU device for tasks using mmcv."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from math import sqrt
from typing import TYPE_CHECKING, Any

from torch.cuda import is_available as cuda_available

from otx.utils.utils import is_xpu_available
from .bs_search_algo import BsSearchAlgo


if TYPE_CHECKING:
    from lightning import Callback

    from otx.engine.engine import Engine

logger = logging.getLogger(__name__)


def adapt_batch_size(
    engine: Engine,
    not_increase: bool = True,
    callbacks: list[Callback] | Callback | None = None,
    **train_args,
) -> None:
    """Decrease batch size if default batch size isn't fit to current GPU device.

    This function just setup for single iteration training to reduce time for adapting.
    The core part of adapting batch size is done in adapt_batch_size in the torch.utils package.

    Args:
        train_func (Callable): The function to train a model.
            Only cfg, dataset and meta are passed to the function when invoking it.
        model (torch.nn.Module): Model to train.
        datasets (List): List of datasets.
        cfg (OTXConfig): Configuration of a training.
        distributed (bool): whether distributed training or not.
        is_nncf (bool): Whether nncf or not.
        meta (Optional[Dict[str, Any]]): meta information.
        not_increase (bool) : Whether adapting batch size to larger value than default value or not.
        model_builder (Optional[Callable]):
            Function for building a model. If it exsits, a model build from model_builder is used instead of the model
            in the argument. It's required for nncf because nncf changes model , which prevent model from pickling.
    """

    if not (cuda_available() or is_xpu_available()):
        msg = "Adaptive batch size supports CUDA or XPU."
        raise RuntimeError(msg)
    engine.model.patch_optimizer_and_scheduler_for_hpo()

    default_bs = engine.datamodule.config.train_subset.batch_size
    bs_search_algo = BsSearchAlgo(
        engine=engine,
        default_bs=default_bs,
        callbacks=callbacks,
        max_bs=len(engine.datamodule.subsets[engine.datamodule.config.train_subset.subset_name]),
        **_adjust_train_args(train_args),
    )
    if not_increase:
        new_batch_size = bs_search_algo.auto_decrease_batch_size()
    else:
        new_batch_size = bs_search_algo.find_big_enough_batch_size()

    if default_bs != new_batch_size:
        engine.datamodule.config.train_subset.batch_size = new_batch_size
        logger.warning("Adapting batch size is done.")
        logger.warning(f"Batch size is adapted : {default_bs} -> {new_batch_size}")

        bs_change_ratio = new_batch_size / default_bs
        origin_lr = engine.model.optimizer_callable.optimizer_kwargs["lr"]
        engine.model.optimizer_callable.optimizer_kwargs["lr"] *= sqrt(bs_change_ratio)  # Using root scale instead of linear scale
        logger.warning(f"learning rate is adapted : {origin_lr} -> {engine.model.optimizer_callable.optimizer_kwargs['lr']}")
    else:
        logger.warning("Adapting batch size is done. Batch size isn't changed.")


def _adjust_train_args(train_args: dict[str, Any]) -> dict[str, Any]:
    train_args.update(train_args.pop("kwargs", {}))
    train_args.pop("self", None)
    train_args.pop("run_hpo", None)
    train_args.pop("adapt_batch_size", None)

    return train_args