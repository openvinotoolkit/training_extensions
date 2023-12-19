# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Instantiator functions for OTX engine components."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from lightning.pytorch.cli import instantiate_class

from . import pylogger

if TYPE_CHECKING:
    from lightning import Callback
    from lightning.pytorch.loggers import Logger

    from otx.core.model.module.base import OTXLitModule

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


def partial_instantiate_class(init: dict) -> partial:
    """Partially instantiates a class with the given initialization arguments.

    Copy from lightning.pytorch.cli.instantiate_class and modify it to use partial.

    Args:
        init (dict): A dictionary containing the initialization arguments.
            It should have the following keys:
            - "init_args" (dict): A dictionary of keyword arguments to be passed to the class constructor.
            - "class_path" (str): The fully qualified path of the class to be instantiated.

    Returns:
        partial: A partial object representing the partially instantiated class.
    """
    kwargs = init.get("init_args", {})
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return partial(args_class, **kwargs)


def instantiate_model(model_cfg: dict | None) -> OTXLitModule | None:
    """Instantiate a module based on the provided module configuration.

    Args:
        module_cfg (dict | None): The module configuration.

    Returns:
        LightningModule | OTXDataModule | None: The instantiated module.
    """
    if not model_cfg:
        log.warning("No model configs found! Skipping...")
        return None

    model_args = model_cfg.get("init_args", {})
    model_args["otx_model"] = instantiate_class(args=(), init=model_args["otx_model"])
    model_args["optimizer"] = partial_instantiate_class(init=model_args["optimizer"])
    model_args["scheduler"] = partial_instantiate_class(init=model_args["scheduler"])
    model_cfg["init_args"] = model_args

    if isinstance(model_cfg, dict) and "class_path" in model_cfg:
        log.info(f"Instantiating module <{model_cfg['class_path']}>")
        return instantiate_class(args=(), init=model_cfg)

    return None
