"""OTX adapters.torch.mmengine.mmaction Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from pathlib import Path

import torch
from mmpretrain import get_model as get_mmpretrain_model
from mmpretrain import list_models as list_mmpretrain_model
from mmpretrain.models import build_backbone, build_neck

from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path
from otx.v2.api.utils.logger import get_logger

logger = get_logger()

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/action_classification/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


def get_model(
    model: str | (Config | dict),
    pretrained: str | bool = False,
    num_classes: int | None = None,
    channel_last: bool = False,
    **kwargs,
) -> torch.nn.Module:
    """Return a PyTorch model for training.

    Args:
        model (Union[str, Config, Dict]): The model to use for pretraining. Can be a string representing the model name,
            a Config object, or a dictionary.
        pretrained (Union[str, bool], optional): Whether to use a pretrained model. Defaults to False.
        num_classes (Optional[int], optional): The number of classes in the dataset. Defaults to None.
        channel_last (bool, optional): Whether to use channel last memory format. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        torch.nn.Module: The PyTorch model for pretraining.
    """
    model_name = None
    if isinstance(model, dict):
        model = Config(cfg_dict=model)
    elif isinstance(model, str):
        if Path(model).is_file():
            model = Config.fromfile(filename=model)
        else:
            model_name = model
    if isinstance(model, Config):
        if hasattr(model, "model"):
            model = Config(model.get("model"))
        if hasattr(model, "name"):
            model_name = model.pop("name")
    if isinstance(model_name, str) and model_name in MODEL_CONFIGS:
        base_model = Config.fromfile(filename=MODEL_CONFIGS[model_name])
        base_model = base_model.get("model", base_model)
        if isinstance(model, str):
            model = {}
        model = Config(cfg_dict=Config.merge_cfg_dict(base_model, model))

    if isinstance(model, Config):
        if not hasattr(model, "model"):
            model["_scope_"] = "mmaction"
            model = Config(cfg_dict={"model": model})
        else:
            model["model"]["_scope_"] = "mmaction"
        if num_classes is not None:
            cls_head = model.model.get("cls_head", {})
            if cls_head and hasattr(cls_head, "num_classes"):
                model["model"]["cls_head"]["num_classes"] = num_classes

    model = get_mmpretrain_model(model, pretrained=pretrained, **kwargs)

    if channel_last and isinstance(model, torch.nn.Module):
        model = model.to(memory_format=torch.channels_last)
    return model


def list_models(pattern: str | None = None, **kwargs) -> list[str]:
    """Returns a list of available models for training.

    Args:
        pattern (Optional[str]): A string pattern to filter the list of available models. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the underlying model listing functions.

    Returns:
        List[str]: A sorted list of available models for pretraining.
    """
    # First, make sure it's a model from mmpretrain.
    model_list = list_mmpretrain_model(pattern=pattern, **kwargs)
    # Add OTX Custom models
    model_list.extend(list(MODEL_CONFIGS.keys()))

    if pattern is not None:
        # Always match keys with any postfix.
        model_list = set(fnmatch.filter(model_list, pattern + "*"))

    return sorted(model_list)
