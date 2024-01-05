# MIT License

# Copyright (c) 2023 Intel Corporation
# Copyright (c) 2021 ashleve

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This source code is borrowed from https://github.com/ashleve/lightning-hydra-template

"""Engine component to training pipeline."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import hydra

from otx.core.config import TrainConfig
from otx.core.model.entity.base import OTXModel

if TYPE_CHECKING:
    from lightning import Callback, Trainer
    from lightning.pytorch.loggers import Logger

    from otx.core.model.module.base import OTXLitModule


def train(
    cfg: TrainConfig,
    otx_model: OTXModel | None = None,
) -> tuple[Trainer, dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during training.

    Args:
        cfg: A DictConfig configuration composed by Hydra.
        otx_model: If it is not `None`, the given OTX model will be overrided.

    Returns:
        A tuple with Pytorch Lightning Trainer and Python dict of metrics
    """
    import torch
    from lightning import seed_everything

    from otx.core.data.module import OTXDataModule
    from otx.core.engine.utils.instantiators import (
        instantiate_callbacks,
        instantiate_loggers,
    )
    from otx.core.engine.utils.logging_utils import log_hyperparameters

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data}>")
    datamodule = OTXDataModule(task=cfg.base.task, config=cfg.data)
    log.info(f"Instantiating model <{cfg.model}>")
    model: OTXLitModule = hydra.utils.instantiate(cfg.model)
    model.meta_info = datamodule.meta_info

    if otx_model is not None:
        if not isinstance(otx_model, OTXModel):
            raise TypeError(otx_model)
        model.model = otx_model
        msg = f"Overriding to this OTX model <{otx_model.__class__.__name__}>"
        log.info(msg)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.callbacks)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.train:
        log.info("Starting training!")
        if cfg.resume:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
        else:
            # load weight to finetune the model
            if cfg.checkpoint is not None:
                loaded_checkpoint = torch.load(cfg.checkpoint)
                model.load_state_dict(loaded_checkpoint["state_dict"])
            # train
            trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.test:
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    log.info(test_metrics)

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return trainer, metric_dict
