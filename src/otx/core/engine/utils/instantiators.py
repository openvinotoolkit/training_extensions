# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Instantiator functions for OTX engine components."""

from __future__ import annotations

from lightning.pytorch.cli import instantiate_class
from lightning import Callback
from lightning.pytorch.loggers import Logger

from . import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: list) -> list[Callback]:
    """Instantiate a list of callbacks based on the provided configuration.

    Args:
        callbacks_cfg (list): A list of callback configurations.

    Returns:
        list[Callback]: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for cb_conf in callbacks_cfg:
        if isinstance(cb_conf, dict) and "class_path" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf['class_path']}>")
            callbacks.append(instantiate_class(args=(), init=cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: dict | None) -> list[Logger]:
    """Instantiate loggers based on the provided logger configuration.

    Args:
        logger_cfg (dict | None): The logger configuration.

    Returns:
        list[Logger]: The list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if isinstance(logger_cfg, dict) and "class_path" in logger_cfg:
        log.info(f"Instantiating logger <{logger_cfg['class_path']}>")
        logger.append(instantiate_class(args=(), init=logger_cfg))

    return logger
