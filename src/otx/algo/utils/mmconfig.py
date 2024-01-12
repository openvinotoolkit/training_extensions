# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utils used for MMConfigs."""

import inspect
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def read_mmconfig(model_name: str, subdir_name: str = ".") -> DictConfig:
    """Read MMConfig.

    It try to read MMConfig from the yaml file which exists in
    `<Directory path of __file__ who calls this function>/mmconfigs/<subdir_name>/<model_name>.yaml`

    For example, if this function is called in `otx/algo/action_classification/x3d.py`,
    `otx/algo/action_classification/mmconfigs/x3d.yaml` will be read.
    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])

    if module is None or (module_file_path := module.__file__) is None:
        msg = (
            "Cannot get Cannot get a valid module from Python function stack. "
            "Please refer to this function docstring to see how to use correctly."
        )
        raise RuntimeError(msg)

    root_dir = Path(module_file_path).parent / "mmconfigs" / subdir_name
    yaml_fpath = root_dir / f"{model_name}.yaml"

    if not yaml_fpath.exists():
        msg = f"mmconfig file for {model_name} is not found in {yaml_fpath}"
        raise FileNotFoundError(msg)

    return OmegaConf.load(yaml_fpath)
