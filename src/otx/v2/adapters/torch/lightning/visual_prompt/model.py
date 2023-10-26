"""OTX adapters.torch.lightning.visual_prompt Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path

from .modules import MODELS

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/visual_prompt/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


def get_model(
    model: dict | (DictConfig | str) | None = None,
    checkpoint: str | None = None,
    **kwargs,
) -> torch.nn.Module:
    """Return a torch.nn.Module object based on the provided model configuration or anomalib model api.

    Args:
        model (Optional[Union[Dict[str, Any], DictConfig, str]]): The model configuration. Can be a dictionary,
            a DictConfig object, or a path to a YAML file containing the configuration.
        checkpoint (Optional[str]): The path to a checkpoint file to load weights from.
        **kwargs: Additional keyword arguments to pass to the `anomalib_get_model` function.

    Returns:
        torch.nn.Module: The model object.

    """
    kwargs = kwargs or {}
    if isinstance(model, str):
        if model in MODEL_CONFIGS:
            model = MODEL_CONFIGS[model]
        if Path(model).is_file():
            model = OmegaConf.load(model)
    if not model.get("model", False):
        model = DictConfig(content={"model": model})

    state_dict = None
    if checkpoint is not None:
        model["checkpoint"] = checkpoint
        state_dict = torch.load(checkpoint)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    if isinstance(model, dict):
        model = OmegaConf.create(model)
    if model.model.name.startswith("otx"):
        model.model.name = "_".join(model.model.name.split("_")[1:])

    model_class = MODELS.get(model.model.name)
    if model_class is None:
        msg = f"Current selected model {model.model.name} is not implemented."
        raise NotImplementedError(
            msg,
        )
    return model_class(config=model, state_dict=state_dict)



def list_models(pattern: str | None = None) -> list[str]:
    """Return a list of available model names.

    Args:
        pattern (Optional[str]): A pattern to filter the model names. Defaults to None.

    Returns:
        List[str]: A sorted list of available model names.
    """
    model_list = list(MODEL_CONFIGS.keys())

    if pattern is not None:
        # Always match keys with any postfix.
        model_list = list(set(fnmatch.filter(model_list, pattern + "*")))

    return sorted(model_list)
