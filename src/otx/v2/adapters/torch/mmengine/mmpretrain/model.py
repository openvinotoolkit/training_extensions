from typing import Any, Dict, Optional, Union

import torch
from mmpretrain import get_model as get_mmpretrain_model
from mmpretrain.models import build_backbone, build_classifier, build_neck
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig
from otx.v2.api.utils.logger import get_logger

from mmengine.runner import load_checkpoint

logger = get_logger()

TRANSFORMER_BACKBONES = ["VisionTransformer", "T2T_ViT", "Conformer"]


def configure_in_channels(model_config, input_shape=[3, 224, 224]):
    # COPY from otx.algorithms.classification.adapters.mmpretrain.configurer.ClassificationConfigurer::configure_in_channel
    configure_required = False
    if model_config.get("neck") is not None:
        if model_config["neck"].get("in_channels") is not None and model_config["neck"]["in_channels"] <= 0:
            configure_required = True
    if not configure_required and model_config.get("head") is not None:
        if model_config["head"].get("in_channels") is not None and model_config["head"]["in_channels"] <= 0:
            configure_required = True
    if not configure_required:
        return model_config

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
    return model_config


def get_model(
    config: Union[Dict, str],
    checkpoint: Optional[str] = None,
    num_classes: int = 1000,
    channel_last: bool = False,
) -> torch.nn.Module:
    if isinstance(config, str):
        config = CustomConfig.fromfile(filename=config)
    elif isinstance(config, dict):
        config = CustomConfig(cfg_dict=config)
    model_config = config.get("model", config)
    model_config = configure_in_channels(model_config)
    # Update num_classes
    model_config["head"]["num_classes"] = num_classes
    model_config.pop("task", None)
    model = build_classifier(model_config)
    device = config.get("device", "cpu")
    model = model.to(device)

    # Checkpoint: 1) parameter 2) config.load_from
    if checkpoint is None:
        checkpoint = model_config.pop("load_from", None)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location=device)
    # model_config["load_from"] = checkpoint
    model._build_config = model_config

    if channel_last:
        model = model.to(memory_format=torch.channels_last)
    return model
