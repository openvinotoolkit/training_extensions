"""OTX adapters.torch.mmengine.mmaction Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from pathlib import Path
import warnings

import torch
from mmaction.apis import init_recognizer
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
    **kwargs,
) -> torch.nn.Module:
    """Return a PyTorch model for training.

    Args:
        model (Union[str, Config, Dict]): The model to use for pretraining. Can be a string representing the model name,
            a Config object, or a dictionary.
        pretrained (Union[str, bool], optional): Whether to use a pretrained model. Defaults to False.
        num_classes (Optional[int], optional): The number of classes in the dataset. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        torch.nn.Module: The PyTorch model for pretraining.
    """
    model_name = None 
    if pretrained is True and 'load_from' in model:
        pretrained = model.load_from
    
    if pretrained is True:
        warnings.warn('Unable to find pre-defined checkpoint of the model.')
        pretrained = None
    elif pretrained is False:
        pretrained = None

    if isinstance(model, dict):
        model = Config(cfg_dict=model)
    elif isinstance(model, str):
        if Path(model).is_file():
            model = Config.fromfile(filename=model)
        else:
            model_name = model
    if isinstance(model, Config):
        if hasattr(model, "name"):
            model_name = model.pop("name")
    if isinstance(model_name, str) and model_name in MODEL_CONFIGS:
        base_model = Config.fromfile(filename=MODEL_CONFIGS[model_name])
        if isinstance(model, str):
            model = {}
        model = Config(cfg_dict=Config.merge_cfg_dict(base_model, model))
    
    if isinstance(model, Config):
        if num_classes is not None:
            cls_head = model.model.get("cls_head", {})
            if cls_head and hasattr(cls_head, "num_classes"):
                model["model"]["cls_head"]["num_classes"] = num_classes
    if pretrained:
        from mmengine.runner import load_checkpoint
        checkpoint = load_checkpoint(model, pretrained, map_location='cpu')
    else:
        checkpoint = None
    
    model = init_recognizer(model, checkpoint, 'cpu', **kwargs)

    return model


def list_models(pattern: str | None = None, **kwargs) -> list[str]:
    """Returns a list of available models for training.

    Args:
        pattern (Optional[str]): A string pattern to filter the list of available models. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the underlying model listing functions.

    Returns:
        List[str]: A sorted list of available models for pretraining.
    """
    model_list = [] 
    # Add OTX Custom models
    model_list.extend(list(MODEL_CONFIGS.keys()))

    if pattern is not None:
        # Always match keys with any postfix.
        model_list = set(fnmatch.filter(model_list, pattern + "*"))

    return sorted(model_list)
