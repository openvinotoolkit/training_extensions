# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os
import time

from mmcv import Config, ConfigDict, build_from_cfg

from otx.algorithms.common.adapters.mmcv.hooks.workflow_hook import (
    WorkflowHook,
    build_workflow_hook,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.common.utils.logger import config_logger, get_logger

from .registry import STAGES
from .stage import get_available_types
from .workflow import Workflow

# from collections import defaultdict


logger = get_logger()


def __build_stage(config, common_cfg=None, index=0):
    logger.info("build_stage()")
    logger.debug(f"[args] config = {config}, common_cfg = {common_cfg}, index={index}")
    config.type = config.type if "type" in config.keys() else "Stage"  # TODO: tmp workaround code for competability
    config.common_cfg = common_cfg
    config.index = index
    return build_from_cfg(config, STAGES)


def __build_workflow(config):
    logger.info("build_workflow()")
    logger.debug(f"[args] config = {config}")

    whooks = []
    whooks_cfg = config.get("workflow_hooks", [])
    for whook_cfg in whooks_cfg:
        if isinstance(whook_cfg, WorkflowHook):
            whooks.append(whook_cfg)
        else:
            whook = build_workflow_hook(whook_cfg.copy())
            whooks.append(whook)

    output_path = config.get("output_path", "logs")
    folder_name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    config.output_path = os.path.join(output_path, folder_name)
    os.makedirs(config.output_path, exist_ok=True)

    # create symbolic link to the output path
    symlink_dst = os.path.join(output_path, "latest")
    if os.path.exists(symlink_dst):
        os.unlink(symlink_dst)
    os.symlink(folder_name, symlink_dst, True)

    log_level = config.get("log_level", "INFO")
    config_logger(os.path.join(config.output_path, "app.log"), level=log_level)

    if not hasattr(config, "gpu_ids"):
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        logger.info(f"CUDA_VISIBLE_DEVICES = {gpu_ids}")
        if gpu_ids is not None:
            if isinstance(gpu_ids, str):
                config.gpu_ids = range(len(gpu_ids.split(",")))
            else:
                raise ValueError(f"not supported type for gpu_ids: {type(gpu_ids)}")
        else:
            config.gpu_ids = range(1)

    common_cfg = copy.deepcopy(config)
    common_cfg.pop("stages")
    if len(whooks_cfg) > 0:
        common_cfg.pop("workflow_hooks")

    stages = [__build_stage(stage_cfg.copy(), common_cfg, index=i) for i, stage_cfg in enumerate(config.stages)]
    return Workflow(stages, whooks)


def build(config, mode=None, stage_type=None, common_cfg=None):
    logger.info("called build_recipe()")
    logger.debug(f"[args] config = {config}")

    if not isinstance(config, Config):
        if isinstance(config, str):
            if os.path.exists(config):
                config = MPAConfig.fromfile(config)
            else:
                logger.error(f"cannot find configuration file {config}")
                raise ValueError(f"cannot find configuration file {config}")

    if hasattr(config, "stages"):
        # build as workflow
        return __build_workflow(config)
    else:
        # build as stage
        if not hasattr(config, "type"):
            logger.info("seems to be passed stage yaml...")
            supported_stage_types = get_available_types()
            if stage_type in supported_stage_types:
                cfg_dict = ConfigDict(
                    dict(
                        type=stage_type,
                        name=f"{stage_type}-{mode}",
                        mode=mode,
                        config=config,
                        index=0,
                    )
                )
            else:
                msg = f"type {stage_type} is not in {supported_stage_types}"
                logger.error(msg)
                raise RuntimeError(msg)
        return __build_stage(cfg_dict, common_cfg=common_cfg)
