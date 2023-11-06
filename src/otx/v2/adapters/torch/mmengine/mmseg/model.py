"""OTX adapters.torch.mmengine.mmseg Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from otx.v2.adapters.torch.mmengine.mmseg.registry import MODELS
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path
from otx.v2.api.utils.logger import get_logger

logger = get_logger()

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/segmentation/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


def replace_num_classes(d: Config | dict, num_classes: int) -> None:
    """Recursively replaces the value of 'num_classes' in a nested dictionary with the given num_classes.

    Args:
        d (dict): The dictionary to be modified.
        num_classes (int): the given num_classes.

    Returns:
        None
    """
    for k, v in d.items():
        if k == "num_classes":
            d[k] = num_classes
        elif isinstance(v, dict):
            replace_num_classes(v, num_classes)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    replace_num_classes(item, num_classes)


def get_model(
    model: str | (Config | dict),
    pretrained: str | bool | None = False,
    num_classes: int | None = None,
    device: str | torch.device | None = None,
    url_mapping: tuple[str, str] | None = None,
    **kwargs,
) -> torch.nn.Module:
    """Return a PyTorch model for training.

    Args:
        model (Union[str, Config, Dict]): The model to use for pretraining. Can be a string representing the model name,
            a Config object, or a dictionary.
        pretrained (Union[str, bool], optional): Whether to use a pretrained model. Defaults to False.
        num_classes (Optional[int], optional): The number of classes in the dataset. Defaults to None.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        url_mapping (Tuple[str, str], optional): The mapping of pretrained
            checkpoint link. For example, load checkpoint from a local dir
            instead of download by ``('https://.*/', './checkpoint')``.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        torch.nn.Module: The PyTorch model for pretraining.
    """
    config = Config(cfg_dict={})
    if isinstance(model, dict):
        config = Config(cfg_dict=model)
    elif isinstance(model, str):
        if Path(model).is_file():
            config = Config.fromfile(filename=model)
        elif model in MODEL_CONFIGS:
            config = Config.fromfile(filename=MODEL_CONFIGS[model])
    else:
        msg = f"model must be a name, a path or a Config object, but got {type(model)}"
        raise TypeError(msg)

    if num_classes is not None:
        replace_num_classes(config, num_classes)

    metainfo = None
    if pretrained is True and "load_from" in config:
        pretrained = config.load_from

    if pretrained is True:
        warnings.warn("Unable to find pre-defined checkpoint of the model.", stacklevel=2)
        pretrained = None
    elif pretrained is False:
        pretrained = None

    if kwargs:
        config.merge_from_dict({"model": kwargs})
    config.model.setdefault("data_preprocessor", config.get("data_preprocessor", None))

    if not hasattr(config, "model"):
        config["_scope_"] = "mmseg"
    else:
        config["model"]["_scope_"] = "mmseg"

    from mmengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("mmseg"):
        seg_model = MODELS.build(config.model)

    dataset_meta = {}
    if pretrained:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        from mmengine.runner import load_checkpoint

        if url_mapping is not None:
            pretrained = re.sub(url_mapping[0], url_mapping[1], pretrained)
        checkpoint = load_checkpoint(seg_model, pretrained, map_location="cpu")
        if "dataset_meta" in checkpoint.get("meta", {}):
            # mmseg 1.x
            dataset_meta = checkpoint["meta"]["dataset_meta"]
        elif "CLASSES" in checkpoint.get("meta", {}):
            # mmseg 0.x
            dataset_meta = {"classes": checkpoint["meta"]["CLASSES"]}

    if device is not None:
        seg_model.to(device)

    seg_model._dataset_meta = dataset_meta  # noqa: SLF001
    seg_model._config = config  # noqa: SLF001
    seg_model._metainfo = metainfo  # noqa: SLF001
    seg_model.eval()
    return seg_model


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
