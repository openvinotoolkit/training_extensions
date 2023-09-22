"""OTX adapters.torch.mmengine.mmpretrain Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from mmpretrain import get_model as get_mmpretrain_model
from mmpretrain import list_models as list_mmpretrain_model
from mmpretrain.models import build_backbone, build_neck

from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path
from otx.v2.api.utils.logger import get_logger

logger = get_logger()

TRANSFORMER_BACKBONES = ["VisionTransformer", "T2T_ViT", "Conformer"]
MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/classification/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


def configure_in_channels(config: Config, input_shape: List[int] = [3, 224, 224]) -> Config:
    # COPY from otx.algorithms.classification.adapters.mmpretrain.configurer.ClassificationConfigurer::configure_in_channel
    configure_required = False
    wrap_model = hasattr(config, "model")
    model_config = config.get("model") if wrap_model else config
    if model_config.get("neck") is not None:
        if model_config["neck"].get("in_channels") is not None and model_config["neck"]["in_channels"] <= 0:
            configure_required = True
    if not configure_required and model_config.get("head") is not None:
        if model_config["head"].get("in_channels") is not None and model_config["head"]["in_channels"] <= 0:
            configure_required = True
    if not configure_required:
        return config

    layer = build_backbone(model_config["backbone"])
    layer.eval()
    if hasattr(layer, "input_shapes"):
        input_shape = next(iter(layer.input_shapes.values()))
        input_shape = input_shape[1:]
        if any(i < 0 for i in input_shape):
            input_shape = [3, 244, 244]
    output = layer(torch.rand([1, *list(input_shape)]))
    if isinstance(output, (tuple, list)):
        output = output[-1]

    if layer.__class__.__name__ in TRANSFORMER_BACKBONES and isinstance(output, (tuple, list)):
        # mmpretrain.VisionTransformer outputs Tuple[List[...]] and the last index of List is the final logit.
        _, output = output

    in_channels = output.shape[1]
    if model_config.get("neck") is not None and model_config["neck"].get("in_channels") is not None:
        logger.info(
            f"'in_channels' config in model.neck is updated from "
            f"{model_config['neck']['in_channels']} to {in_channels}",
        )
        model_config["neck"].in_channels = in_channels
        logger.debug(f"input shape for neck {input_shape}")

        layer = build_neck(model_config["neck"])
        layer.eval()
        output = layer(torch.rand(output.shape))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        in_channels = output.shape[1]
    if model_config.get("head") is not None and model_config["head"].get("in_channels") is not None:
        logger.info(
            f"'in_channels' config in model.head is updated from "
            f"{model_config['head']['in_channels']} to {in_channels}",
        )
        model_config["head"]["in_channels"] = in_channels
    if wrap_model:
        config["model"] = model_config
    return config


def get_model(
    model: Union[str, Config, Dict],
    pretrained: Union[str, bool] = False,
    num_classes: Optional[int] = None,
    channel_last: bool = False,
    return_dict: bool = False,
    **kwargs,
) -> torch.nn.Module:
    model_name = None
    if isinstance(model, dict):
        model = Config(cfg_dict=model)
    elif isinstance(model, str):
        if Path(model).is_file():
            model = Config.fromfile(filename=model)
        else:
            model_name = model
    if isinstance(model, (dict, Config)):
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
        model = configure_in_channels(model)  # TODO: more clearly
        if not hasattr(model, "model"):
            model["_scope_"] = "mmpretrain"
            model = Config(cfg_dict={"model": model})
        else:
            model["model"]["_scope_"] = "mmpretrain"
        if num_classes is not None:
            head = model.model.get("head", {})
            if head and hasattr(head, "num_classes"):
                model["model"]["head"]["num_classes"] = num_classes
        if return_dict:
            return model.model._cfg_dict.to_dict()

    model = get_mmpretrain_model(model, pretrained=pretrained, **kwargs)

    if channel_last and isinstance(model, torch.nn.Module):
        model = model.to(memory_format=torch.channels_last)
    return model


def list_models(pattern: Optional[str] = None, **kwargs) -> List[str]:
    # First, make sure it's a model from mmpretrain.
    model_list = list_mmpretrain_model(pattern=pattern, **kwargs)
    # Add OTX Custom models
    model_list.extend(list(MODEL_CONFIGS.keys()))

    if pattern is not None:
        # Always match keys with any postfix.
        model_list = set(fnmatch.filter(model_list, pattern + "*"))

    return sorted(model_list)


if __name__ == "__main__":
    model_list = list_models("otx*")
    model = get_model(model_list[0])
