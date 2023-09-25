"""OTX adapters.torch.anomalib Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from anomalib.models import get_model as anomalib_get_model
from omegaconf import DictConfig, OmegaConf

from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/anomaly_classification/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


def get_model(
    model: Optional[Union[Dict[str, Any], DictConfig, str]] = None,
    checkpoint: Optional[str] = None,
    **kwargs,
) -> torch.nn.Module:
    """_summary_.

    Args:
        config (Optional[Union[Dict[str, Any], DictConfig]], optional): _description_. Defaults to None.
        {"model": {"name"}}
        checkpoint (Optional[str], optional): _description_. Defaults to None.

    Returns:
        AnomalyModule: _description_
    """
    kwargs = kwargs or {}
    if isinstance(model, str):
        if model in MODEL_CONFIGS:
            model = MODEL_CONFIGS[model]
        model = OmegaConf.load(model)
    if not model.get("model", False):
        model = DictConfig(content={"model": model})
    if checkpoint is not None:
        model["init_weights"] = checkpoint
    if isinstance(model, dict):
        model = OmegaConf.create(model)
    return anomalib_get_model(config=model)


def list_models(pattern: Optional[str] = None) -> List[str]:
    model_list = list(MODEL_CONFIGS.keys())

    if pattern is not None:
        # Always match keys with any postfix.
        model_list = list(set(fnmatch.filter(model_list, pattern + "*")))

    return sorted(model_list)


if __name__ == "__main__":
    model_config = {
        "model": {
            "name": "padim",
            "backbone": "resnet18",
            "pre_trained": True,
            "layers": ["layer1", "layer2", "layer3"],
            "normalization_method": "min_max",
            "input_size": [256, 256],
        },
    }
    model = get_model(model_config)
