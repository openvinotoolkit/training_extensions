# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utils used for MMConfigs."""

import inspect
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def read_mmconfig(model_name: str) -> DictConfig:
    """Read MMConfig.

    It try to read MMConfig from the yaml file which exists in
    `<Directory path of __file__ who calls this function>/mmconfigs/<model_name>.yaml`
    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])

    if module is None or module.__file__ is None:
        msg = "Cannot get valid model from stack"
        raise RuntimeError(msg)

    root_dir = Path().parent / "mmconfigs"
    fpath = root_dir / f"{model_name}.yaml"

    if not fpath.exists():
        raise FileNotFoundError

    return OmegaConf.load(fpath)
