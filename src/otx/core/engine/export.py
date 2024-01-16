# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine component to testing pipeline."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

import hydra
import torch

from otx.core.config import TrainConfig

if TYPE_CHECKING:
    from lightning import LightningModule


def export(cfg: TrainConfig) -> str:
    """Exports the model.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with Pytorch Lightning Trainer and Python dict of metrics
    """
    log.info(f"Instantiating model <{cfg.model}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.checkpoint)["state_dict"])

    log.info("Starting exporting!")
    model.eval()
    model.export(cfg.base.output_dir)

    return ""
