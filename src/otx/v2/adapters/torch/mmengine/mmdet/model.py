"""OTX adapters.torch.mmengine.mmpretrain Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from copy import deepcopy
from pathlib import Path

import torch
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
from mmengine.runner.checkpoint import load_checkpoint

from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path
from otx.v2.api.utils.logger import LEVEL, get_logger

logger = get_logger()

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/instance_segmentation/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


def get_model(
    model: str | (Config | dict),
    num_classes: int | None = None,
    channel_last: bool = False,
) -> torch.nn.Module:
    """Return a PyTorch model for training.

    Args:
        model (str, Config, dict): The model to use for training. Can be a string representing the model name,
            a Config object, or a dictionary.
        num_classes (int, None, optional): The number of classes in the dataset. Defaults to None.
        channel_last (bool): Whether to use channel last memory format. Defaults to False.

    Returns:
        torch.nn.Module: The PyTorch model for training.
    """
    model_name: str | None = None
    if isinstance(model, dict):
        model_cfg = Config(cfg_dict=model)
    elif isinstance(model, str):
        if Path(model).is_file():
            model_cfg = Config.fromfile(filename=model)
        else:
            model_name = model
            model_cfg = Config({"model": {}})
    else:
        model_cfg = model

    if model_cfg.get("model", None) is None:
        model_cfg = Config({"model": Config.to_dict(model_cfg)})
    checkpoint = model_cfg.get("load_from", None)
    model = Config(model_cfg.get("model"))
    if hasattr(model, "name"):
        model_name = model.pop("name")

    if isinstance(model_name, str) and model_name in MODEL_CONFIGS:
        base_model = Config.fromfile(filename=MODEL_CONFIGS[model_name])
        checkpoint = base_model.get("load_from", None)
        base_model = base_model.get("model", base_model)
        model_cfg.model = Config(cfg_dict=Config.merge_cfg_dict(base_model, model_cfg.model))

    model_cfg["model"]["_scope_"] = "mmdet"
    if num_classes is not None:
        head_names = ("mask_head", "bbox_head", "segm_head")
        if "roi_head" in model_cfg.model:
            for head_name in head_names:
                if head_name in model_cfg.model.roi_head:
                    if isinstance(model_cfg.model.roi_head[head_name], list):
                        for head in model_cfg.model.roi_head[head_name]:
                            head.num_classes = num_classes
                    else:
                        model_cfg.model.roi_head[head_name].num_classes = num_classes
        else:
            # For other architectures (including SSD)
            for head_name in head_names:
                if head_name in model_cfg.model:
                    model_cfg.model[head_name].num_classes = num_classes

    torch_model = MODELS.build(model_cfg.model)
    mm_logger = MMLogger.get_current_instance()
    mm_logger.setLevel("WARNING")
    torch_model.init_weights()
    mm_logger.setLevel(LEVEL)

    if checkpoint:
        load_checkpoint(torch_model, checkpoint)
        model_cfg.load_from = checkpoint

    if channel_last and isinstance(torch_model, torch.nn.Module):
        torch_model = torch_model.to(memory_format=torch.channels_last)

    # For compatibility with mmdet api.
    torch_model.cfg = model_cfg
    # For compatibility with other tasks.
    torch_model._config = model_cfg  # noqa: SLF001

    source_config = deepcopy(model_cfg.model)
    source_config.name = model_name
    torch_model.config_dict = Config.to_dict(source_config)

    return torch_model


def list_models(pattern: str | None = None) -> list[str]:
    """Returns a list of available models for training.

    Args:
        pattern (Optional[str]): A string pattern to filter the list of available models. Defaults to None.

    Returns:
        List[str]: A sorted list of available models for pretraining.
    """
    model_list = []
    model_list.extend(list(MODEL_CONFIGS.keys()))

    if pattern is not None:
        # Always match keys with any postfix.
        return sorted(set(fnmatch.filter(model_list, pattern + "*")))
    return sorted(model_list)
