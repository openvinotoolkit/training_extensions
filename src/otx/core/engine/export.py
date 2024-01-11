# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine component to testing pipeline."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import hydra
import torch

from otx.core.config import TrainConfig

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


def export(cfg: TrainConfig) -> str:
    """Exports the model.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with Pytorch Lightning Trainer and Python dict of metrics
    """
    from otx.core.data.module import OTXDataModule

    log.info(f"Instantiating datamodule <{cfg.data}>")
    datamodule = OTXDataModule(task=cfg.base.task, config=cfg.data)

    log.info(f"Instantiating model <{cfg.model}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.meta_info = datamodule.meta_info
    model.load_state_dict(torch.load(cfg.checkpoint)["state_dict"])

    log.info("Starting exporting!")
    model.eval()
    model.export(cfg.base.output_dir)

    return ""
