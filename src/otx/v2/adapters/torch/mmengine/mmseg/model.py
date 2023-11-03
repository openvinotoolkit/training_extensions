"""OTX adapters.torch.mmengine.mmseg Model APIs."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Union

import torch

from otx.v2.adapters.torch.mmengine.mmseg.registry import MODELS
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.importing import get_files_dict, get_otx_root_path
from otx.v2.api.utils.logger import get_logger

logger = get_logger()

MODEL_CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/segmentation/models"
MODEL_CONFIGS = get_files_dict(MODEL_CONFIG_PATH)


# def test_replace_num_classes():
#     # Define a nested dictionary with 'num_classes' values to be replaced
#     d = {
#         'num_classes': 10,
#         'layers': [
#             {
#                 'num_classes': 20,
#                 'filters': 32
#             },
#             {
#                 'num_classes': 30,
#                 'filters': 64
#             }
#         ]
#     }

#     # Call the function to replace 'num_classes' values with 5
#     replace_num_classes(d, 5)

#     # Check that all 'num_classes' values have been replaced with 5
#     assert d['num_classes'] == 5
#     assert d['layers'][0]['num_classes'] == 5
#     assert d['layers'][1]['num_classes'] == 5


def replace_num_classes(d: Union[Config | dict], num_classes: int) -> None:
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
    pretrained: str | bool = False,
    num_classes: int | None = None,
    device=None,
    url_mapping: tuple[str, str] = None,
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
    config = Config(cfg_dict={})
    if isinstance(model, dict):
        config = Config(cfg_dict=model)
    elif isinstance(model, str):
        if Path(model).is_file():
            config = Config.fromfile(filename=model)
        elif model in MODEL_CONFIGS:
            config = Config.fromfile(filename=MODEL_CONFIGS[model])
    else:
        raise TypeError("model must be a name, a path or a Config object, " f"but got {type(model)}")

    if num_classes is not None:
        replace_num_classes(config, num_classes)

    metainfo = None
    if pretrained is True and "load_from" in config:
        pretrained = config.load_from

    if pretrained is True:
        warnings.warn("Unable to find pre-defined checkpoint of the model.")
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
        model = MODELS.build(config.model)

    dataset_meta = {}
    if pretrained:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        from mmengine.runner import load_checkpoint

        if url_mapping is not None:
            pretrained = re.sub(url_mapping[0], url_mapping[1], pretrained)
        checkpoint = load_checkpoint(model, pretrained, map_location="cpu")
        # TODO: need to check this part for mmseg
        if "dataset_meta" in checkpoint.get("meta", {}):
            # mmpretrain 1.x
            dataset_meta = checkpoint["meta"]["dataset_meta"]
        elif "CLASSES" in checkpoint.get("meta", {}):
            # mmcls 0.x
            dataset_meta = {"classes": checkpoint["meta"]["CLASSES"]}

    if device is not None:
        model.to(device)

    model._dataset_meta = dataset_meta  # save the dataset meta
    model._config = config  # save the config in the model
    model._metainfo = metainfo  # save the metainfo in the model
    model.eval()
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
    # model_list = list_mmseg_model(pattern=pattern, **kwargs)
    # # Add OTX Custom models
    # model_list.extend(list(MODEL_CONFIGS.keys()))

    # if pattern is not None:
    #     # Always match keys with any postfix.
    #     model_list = set(fnmatch.filter(model_list, pattern + "*"))

    return []
