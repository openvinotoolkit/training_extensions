"""Algorithm to find a proper batch size which is fit to current GPU device for tasks using mmcv."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from math import sqrt
from typing import TYPE_CHECKING

from torch.cuda import is_available as cuda_available

from otx.utils.utils import is_xpu_available
from .bs_search_algo import BsSearchAlgo


if TYPE_CHECKING:
    from lightning import Trainer

    from otx.core.model.module.base import OTXLitModule
    from otx.core.data.module import OTXDataModule

logger = logging.getLogger(__name__)


def adapt_batch_size(
    trainer: Trainer,
    model: OTXLitModule,
    datamodule: OTXDataModule,
    not_increase: bool = True,
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

    default_bs = datamodule.config.train_subset.batch_size
    bs_search_algo = BsSearchAlgo(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        default_bs=default_bs,
        max_bs=len(datamodule.subsets[datamodule.config.train_subset.subset_name]),
    )
    if not_increase:
        new_batch_size = bs_search_algo.auto_decrease_batch_size()
    else:
        new_batch_size = bs_search_algo.find_big_enough_batch_size()

    if default_bs != new_batch_size:
        datamodule.config.train_subset.batch_size = new_batch_size
        logger.warning("Adapting batch size is done.")
        logger.warning(f"Batch size is adapted : {default_bs} -> {new_batch_size}")

        bs_change_ratio = new_batch_size / default_bs
        if isinstance(model.optimizer, list):
            for i, opt in enumerate(model.optimizer):
                origin_lr = opt.keywords["lr"]
                opt.keywords["lr"] *= sqrt(bs_change_ratio)  # Using root scale instead of linear scale
                logger.warning(f"learning rate of optimizer[{i}] is adapted : {origin_lr} -> {opt.keywords['lr']}")
        else:
            origin_lr = model.optimizer.keywords["lr"]
            model.optimizer.keywords["lr"] *= sqrt(bs_change_ratio)  # Using root scale instead of linear scale
            logger.warning(f"learning rate is adapted : {origin_lr} -> {model.optimizer.keywords['lr']}")
    else:
        logger.warning("Adapting batch size is done. Batch size isn't changed.")
