"""OTX adapters.torch.mmengine.mmpretrain Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
from copy import deepcopy
from pathlib import Path

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


def configure_in_channels(config: Config, input_shape: list[int] | None = None) -> Config:
    """Configure the 'in_channels' parameter for the model's neck and head based on the output shape of the backbone.

    Args:
        config (Config): The configuration object for the model.
        input_shape (Optional[List[int]], optional): The input shape of the model. Defaults to None.

    Returns:
        Config: The updated configuration object.
    """
    configure_required = False
    wrap_model = hasattr(config, "model")
    model_config = config.get("model") if wrap_model else config
    if (
        model_config.get("neck") is not None
        and model_config["neck"].get("in_channels") is not None
        and model_config["neck"]["in_channels"] <= 0
    ):
        configure_required = True
    if (
        not configure_required
        and model_config.get("head") is not None
        and model_config["head"].get("in_channels") is not None
        and model_config["head"]["in_channels"] <= 0
    ):
        configure_required = True
    if not configure_required:
        return config

    layer = build_backbone(model_config["backbone"])
    layer.eval()
    _input_shape = [3, 224, 224] if input_shape is None else input_shape
    if hasattr(layer, "input_shapes"):
        _input_shape = next(iter(layer.input_shapes.values()))
        _input_shape = _input_shape[1:]
        if any(i < 0 for i in _input_shape):
            _input_shape = [3, 244, 244]
    output = layer(torch.rand([1, *list(_input_shape)]))
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
        model = configure_in_channels(model)
        if not hasattr(model, "model"):
            model["_scope_"] = "mmpretrain"
            model = Config(cfg_dict={"model": model})
        else:
            model["model"]["_scope_"] = "mmpretrain"
        if num_classes is not None:
            head = model.model.get("head", {})
            if head and hasattr(head, "num_classes"):
                model["model"]["head"]["num_classes"] = num_classes

    torch_model = get_mmpretrain_model(model, pretrained=pretrained, **kwargs)

    source_config = deepcopy(torch_model._config.model)  # noqa: SLF001
    source_config.name = model_name
    torch_model.config_dict = Config.to_dict(source_config)

    if channel_last and isinstance(torch_model, torch.nn.Module):
        torch_model = torch_model.to(memory_format=torch.channels_last)
    return torch_model


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
