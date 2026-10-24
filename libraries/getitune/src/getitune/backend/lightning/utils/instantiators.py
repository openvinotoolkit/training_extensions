# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Instantiator functions for getitune engine components."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.utilities import rank_zero_only

if TYPE_CHECKING:
    from lightning import Callback
    from lightning.pytorch.loggers import Logger


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    Args:
        name: The name of the logger, defaults to ``__name__``.

    Returns:
        A logger object.
    """
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_pylogger(__name__)


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
            log.info("Instantiating callback <%s>", cb_conf["class_path"])
            callbacks.append(instantiate_class(args=(), init=cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: list | None) -> list[Logger]:
    """Instantiate loggers based on the provided logger configuration.

    Args:
        logger_cfg (list | None): The logger configuration.

    Returns:
        list[Logger]: The list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    for lg_conf in logger_cfg:
        if isinstance(lg_conf, dict) and "class_path" in lg_conf:
            log.info("Instantiating logger <%s>", lg_conf["class_path"])
            logger.append(instantiate_class(args=(), init=lg_conf))

    return logger


def partial_instantiate_class(init: list | dict | None) -> list[partial] | None:
    """Partially instantiates a class with the given initialization arguments.

    Copy from lightning.pytorch.cli.instantiate_class and modify it to use partial.

    Args:
        init (list | dict | None): A dictionary containing the initialization arguments.
            It should have the following each keys:
            - "init_args" (dict): A dictionary of keyword arguments to be passed to the class constructor.
            - "class_path" (str): The fully qualified path of the class to be instantiated.

    Returns:
        list[partial] | None: A partial object representing the partially instantiated class.
    """
    if not init:
        return None
    if not isinstance(init, list):
        init = [init]
    items: list[partial] = []
    for item in init:
        kwargs = item.get("init_args", {})
        class_module, class_name = item["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
        items.append(partial(args_class, **kwargs))
    return items
