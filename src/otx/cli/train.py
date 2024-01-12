# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for training."""
# ruff: noqa

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any
import hydra
from hydra import compose, initialize
from jsonargparse import ArgumentParser
from otx.core.model.entity.base import OTXModel
from otx.cli.utils.hydra import configure_hydra_outputs


if TYPE_CHECKING:
    from jsonargparse._actions import _ActionSubCommands
    from lightning import Callback
    from lightning.pytorch.loggers import Logger


def add_train_parser(subcommands_action: _ActionSubCommands) -> None:
    """Add subparser for train command.

    Args:
        subcommands_action (_ActionSubCommands): Sub-Command in CLI.

    Returns:
        None
    """
    parser = ArgumentParser()
    parser.add_argument("overrides", help="overrides values", default=[], nargs="+")
    subcommands_action.add_subcommand("train", parser, help="Training subcommand for OTX")


def otx_train(overrides: list[str]) -> dict[str, Any]:
    """Main entry point for training.

    :param overrides: Override List values.
    :return: Metrics values obtained from the model trainer.
    """
    from otx.core.config import register_configs

    # This should be in front of hydra.initialize()
    register_configs()

    with initialize(config_path="../config", version_base="1.3", job_name="otx_train"):
        cfg = compose(config_name="train", overrides=overrides, return_hydra_config=True)
        configure_hydra_outputs(cfg)

        # train the model
        from lightning import seed_everything

        from otx.core.data.module import OTXDataModule
        from otx.core.utils.instantiators import (
            instantiate_callbacks,
            instantiate_loggers,
        )

        if cfg.seed is not None:
            seed_everything(cfg.seed, workers=True)

        log.info(f"Instantiating datamodule <{cfg.data}>")
        datamodule = OTXDataModule(task=cfg.base.task, config=cfg.data)
        log.info(f"Instantiating model <{cfg.model}>")
        model: OTXModel = hydra.utils.instantiate(cfg.model.otx_model)
        optimizer = hydra.utils.instantiate(cfg.model.optimizer)
        scheduler = hydra.utils.instantiate(cfg.model.scheduler)

        log.info("Instantiating callbacks...")
        callbacks: list[Callback] = instantiate_callbacks(cfg.callbacks)

        log.info("Instantiating loggers...")
        logger: list[Logger] = instantiate_loggers(cfg.logger)

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

        train_metrics = {}

        trainer_kwargs.pop("_target_", None)
        if cfg.train:
            train_metrics = engine.train(
                callbacks=callbacks,
                logger=logger,
                resume=cfg.resume,
                **trainer_kwargs,
            )

        test_metrics = {}
        if cfg.test:
            test_metrics = engine.test()

        return {**train_metrics, **test_metrics}
