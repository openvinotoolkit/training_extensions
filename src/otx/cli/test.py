# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for testing."""
# ruff: noqa

from __future__ import annotations

from typing import TYPE_CHECKING
import hydra
import logging as log
from hydra import compose, initialize
from jsonargparse import ArgumentParser
from otx.core.model.entity.base import OTXModel
from otx.cli.utils.hydra import configure_hydra_outputs

if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands


def otx_test(overrides: list[str]) -> None:
    """Main entry point for testing.

    :param overrides: Override List values.
    :return: Optional[float] with optimized metric value.
    """
    from otx.core.config import register_configs

    # This should be in front of hydra.initialize()
    register_configs()

    with initialize(config_path="../config", version_base="1.3", job_name="otx_test"):
        cfg = compose(config_name="test", overrides=overrides, return_hydra_config=True)
        configure_hydra_outputs(cfg)

        # test the model
        from otx.core.data.module import OTXDataModule

        log.info(f"Instantiating datamodule <{cfg.data}>")
        datamodule = OTXDataModule(task=cfg.base.task, config=cfg.data)

        log.info(f"Instantiating model <{cfg.model}>")
        model: OTXModel = hydra.utils.instantiate(cfg.model.otx_model)
        optimizer = hydra.utils.instantiate(cfg.model.optimizer)
        scheduler = hydra.utils.instantiate(cfg.model.scheduler)

        from otx.engine import Engine

        trainer_kwargs = {**cfg.trainer}
        engine = Engine(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            datamodule=datamodule,
            checkpoint=cfg.checkpoint,
            device=trainer_kwargs.pop("accelerator", "auto"),
        )
        engine.test()
