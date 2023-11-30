# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Functions related to Hydra."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.core.utils import _save_config, configure_log

if TYPE_CHECKING:
    from omegaconf import DictConfig


def configure_hydra_outputs(cfg: DictConfig) -> None:
    """Configures the outputs for Hydra.

    Copied some of the log and output configuration code from run_job fucntion of 'hydra.core.utils'.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        None
    """
    output_dir = Path(cfg.base.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure Logger & logging outputs
    configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)

    # Configure .hydra output config files
    if cfg.hydra.output_subdir is not None:
        hydra_output = Path(cfg.hydra.runtime.output_dir) / Path(cfg.hydra.output_subdir)
        task_cfg = copy.deepcopy(cfg)
        # hydra_cfg = copy.deepcopy(HydraConfig.instance().cfg)
        _save_config(task_cfg, "config.yaml", hydra_output)
        # _save_config(hydra_cfg, "hydra.yaml", hydra_output)
        _save_config(cfg.hydra.overrides.task, "overrides.yaml", hydra_output)
