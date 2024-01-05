# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine component to testing pipeline."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import hydra

from otx.core.config import TrainConfig

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


def test(cfg: TrainConfig) -> tuple[Trainer, dict[str, Any]]:
    """Tests the model.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with Pytorch Lightning Trainer and Python dict of metrics
    """
    from otx.core.data.module import OTXDataModule

    log.info(f"Instantiating datamodule <{cfg.data}>")
    datamodule = OTXDataModule(task=cfg.base.task, config=cfg.data)

    log.info(f"Instantiating model <{cfg.model}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.meta_info = datamodule.meta_info

    log.info(f"Instantiating trainer <{cfg.trainer}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Starting testing!")
    if cfg.checkpoint is None:
        msg = "Checkpoint was not found! Could you please specify 'checkpoint'?"
        raise ValueError(msg)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.checkpoint)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**test_metrics}

    return trainer, metric_dict
