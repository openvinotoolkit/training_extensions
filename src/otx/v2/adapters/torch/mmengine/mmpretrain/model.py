from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from mmpretrain import get_model as get_mmpretrain_model
from mmpretrain.models import build_backbone, build_neck
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig as Config
from otx.v2.api.utils.logger import get_logger

logger = get_logger()

TRANSFORMER_BACKBONES = ["VisionTransformer", "T2T_ViT", "Conformer"]


def configure_in_channels(config, input_shape=[3, 224, 224]):
    # COPY from otx.algorithms.classification.adapters.mmpretrain.configurer.ClassificationConfigurer::configure_in_channel
    configure_required = False
    wrap_model = hasattr(config, "model")
    model_config = config.pop("model") if wrap_model else config
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
        input_shape = next(iter(getattr(layer, "input_shapes").values()))
        input_shape = input_shape[1:]
        if any(i < 0 for i in input_shape):
            input_shape = [3, 244, 244]
    output = layer(torch.rand([1] + list(input_shape)))
    if isinstance(output, (tuple, list)):
        output = output[-1]

    if layer.__class__.__name__ in TRANSFORMER_BACKBONES and isinstance(output, (tuple, list)):
        # mmpretrain.VisionTransformer outputs Tuple[List[...]] and the last index of List is the final logit.
        _, output = output

    in_channels = output.shape[1]
    if model_config.get("neck") is not None:
        if model_config["neck"].get("in_channels") is not None:
            logger.info(
                f"'in_channels' config in model.neck is updated from "
                f"{model_config['neck']['in_channels']} to {in_channels}"
            )
            model_config["neck"].in_channels = in_channels
            logger.debug(f"input shape for neck {input_shape}")

            layer = build_neck(model_config["neck"])
            layer.eval()
            output = layer(torch.rand(output.shape))
            if isinstance(output, (tuple, list)):
                output = output[-1]
            in_channels = output.shape[1]
    if model_config.get("head") is not None:
        if model_config["head"].get("in_channels") is not None:
            logger.info(
                f"'in_channels' config in model.head is updated from "
                f"{model_config['head']['in_channels']} to {in_channels}"
            )
            model_config["head"]["in_channels"] = in_channels
    if wrap_model:
        config["model"] = model_config
    return config


def get_model(
    model: Union[str, Config, Dict],
    pretrained: Union[str, bool] = False,
    num_classes: int = 1000,
    channel_last: bool = False,
    **kwargs,
):
    if isinstance(model, dict):
        model = Config(cfg_dict=model)
    elif isinstance(model, str) and Path(model).is_file():
        model = Config.fromfile(filename=model)
    if isinstance(model, Config):
        model = configure_in_channels(model)
    if not hasattr(model, "model"):
        model = Config(cfg_dict={"model": model})
    model["model"]["_scope_"] = "mmpretrain"
    model = get_mmpretrain_model(model, pretrained, **kwargs)
    if num_classes is not None:
        model.head.num_classes = num_classes

    if channel_last:
        model = model.to(memory_format=torch.channels_last)
    return model
