"""Builder Class for training template.

For user's various use cases and convenient CLI,
It is an internal Builder class used in the otx build command
that enables the configuration of the basic workspace of OTX
and supports the replacement of the backbone of the model.
"""
# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import inspect
from pathlib import Path
from typing import Any, Dict, Union

import mmcv
import torch
from mmcv.utils import Registry, build_from_cfg
from torch import nn

from otx.algorithms import TRANSFORMER_BACKBONES
from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.api.entities.model_template import TaskType
from otx.cli.utils.importing import (
    get_backbone_list,
    get_backbone_registry,
    get_module_args,
)

# pylint: disable=too-many-locals, too-many-statements, too-many-branches


def get_backbone_out_channels(backbone: nn.Module):
    """Get output channels of backbone using fake data."""
    out_channels = []
    input_size = backbone.input_size if hasattr(backbone, "input_size") else 64
    fake_data = torch.rand(2, 3, input_size, input_size)
    outputs = backbone(fake_data)
    for out in outputs:
        out_channels.append(out.shape[1])
    return out_channels


def update_backbone_args(backbone_config: dict, registry: Registry, backend: str):
    """Update Backbone required arguments.

    This function checks the init parameters of the corresponding backbone function (or class)
    and identifies the required arguments.
    Also, it distinguishes the argment needed for the build to add convenience to the user.
    """
    backbone_module = registry.get(backbone_config["type"])
    if not backbone_module:
        raise ValueError(f"{backbone_config['type']} is not supported backbone")
    required_args, default_args = get_module_args(backbone_module)
    for arg_key, default_value in default_args.items():
        if arg_key not in backbone_config:
            backbone_config[arg_key] = default_value

    missing_args = []
    for arg in required_args:
        if arg not in backbone_config:
            missing_args.append(arg)
    if len(missing_args) > 0:
        print(
            f"[*] {backbone_config['type']} requires the argument : {missing_args}"
            f"\n[*] Please refer to {inspect.getfile(backbone_module)}"
        )
    if "out_indices" in backbone_config:
        backbone_config["use_out_indices"] = True
    else:
        backbone_config["use_out_indices"] = False

    updated_missing_args = []
    backbone_type = backbone_config["type"]
    backbone_list = get_backbone_list(backend)
    if backbone_type not in backbone_list:
        return missing_args
    backbone_data = backbone_list[backbone_type]
    # Patch missing_args
    for arg in missing_args:
        if "options" in backbone_data and arg in backbone_data["options"]:
            backbone_config[arg] = backbone_data["options"][arg][0]
            print(
                f"[*] '{arg}' can choose between: {backbone_data['options'][arg]}"
                f"\n[*] '{arg}' default value: {backbone_config[arg]}"
            )
        else:
            backbone_config[arg] = "!!!!!!!!!!!INPUT_HERE!!!!!!!!!!!"
            updated_missing_args.append(arg)
    return updated_missing_args


def update_channels(model_config: OTXConfig, out_channels: Any):
    """Update in_channel of head or neck."""
    if hasattr(model_config.model, "neck") and model_config.model.neck:
        if model_config.model.neck.get("type", None) == "GlobalAveragePooling":
            model_config.model.neck.pop("in_channels", None)
        else:
            print(f"\tUpdate model.neck.in_channels: {out_channels}")
            model_config.model.neck.in_channels = out_channels

    elif hasattr(model_config.model, "decode_head"):
        head_in_index = model_config.model.decode_head.get("in_index", None)
        if head_in_index and len(out_channels) != len(head_in_index):
            updated_in_index = list(range(len(out_channels)))
            print(f"\tUpdate model.decode_head.in_index: {updated_in_index}")
            model_config.model.decode_head.in_index = updated_in_index
        print(f"\tUpdate model.decode_head.in_channels: {out_channels}")
        model_config.model.decode_head.in_channels = out_channels

    elif hasattr(model_config.model, "head"):
        print(f"\tUpdate model.head.in_channels: {out_channels}")
        model_config.model.head.in_channels = out_channels
    else:
        raise NotImplementedError("This architecture currently does not support public backbone.")


class Builder:
    """Class that implements a model templates registry."""

    def build_backbone_config(self, backbone_type: str, output_path: Union[Path, str]):
        """Build Backbone configs from backbone type.

        This is a function that makes the configuration
        of the usable backbone found by the user through otx find.
        backbone_type: The type of backbone want to get - {backend.backbone_type} (str)
        output_path: new backbone configuration file output path (Union[Path, str])
        """
        print(f"[*] Backbone Config: {backbone_type}")
        output_path = output_path if isinstance(output_path, Path) else Path(output_path)

        backend, backbone_class = Registry.split_scope_key(backbone_type)
        backbone_config: Dict[str, Any] = dict(type=backbone_type)
        if backbone_class == "MMOVBackbone":
            backend = f"omz.{backend}"
            backbone_config["verify_shape"] = False
        backbone_registry, _ = get_backbone_registry(backend)
        missing_args = update_backbone_args(backbone_config, backbone_registry, backend)
        if str(output_path).endswith((".yml", ".yaml", ".json")):
            mmcv.dump({"backbone": backbone_config}, str(output_path.absolute()))
            print(f"[*] Save backbone configuration: {str(output_path.absolute())}")
        else:
            raise ValueError("The backbone config support file format is as follows: (.yml, .yaml, .json)")
        return missing_args

    def merge_backbone(
        self,
        model_config_path: Union[Path, str],
        backbone_config_path: Union[Path, str],
        output_path: Union[Path, str] = None,
    ):
        """Build model & update backbone configs.

        This is a function that updates the existing model to be able to build
        through the backbone configuration file or backbone type.
        model_config_path: model configuration file path (Union[Path, str])
        backbone_config_path: backbone configuration file path (Union[Path, str])
        output_path: new model.py output path (Union[Path, str])
        """
        print(f"[*] Update {model_config_path} with {backbone_config_path}")
        model_config_path = model_config_path if isinstance(model_config_path, Path) else Path(model_config_path)
        backbone_config_path = (
            backbone_config_path if isinstance(backbone_config_path, Path) else Path(backbone_config_path)
        )

        # Get Model config from model config file
        if model_config_path.exists():
            model_config = OTXConfig.fromfile(str(model_config_path))
            print(f"\tTarget Model: {model_config.model.type}")
        else:
            raise ValueError(f"[*] The model is not properly defined or not found: {model_config_path}")

        # Get Backbone config from config file
        if backbone_config_path.exists():
            backbone_config = mmcv.load(str(backbone_config_path))
        else:
            raise ValueError(f"[*] The backbone is not found: {str(backbone_config_path)}")

        if "backbone" in backbone_config:
            backbone_config = backbone_config["backbone"]

        # Get Backbone configuration
        backend, backbone_class = Registry.split_scope_key(backbone_config["type"])
        backend = f"omz.{backend}" if backbone_class == "MMOVBackbone" else backend
        print(f"\tTarget Backbone: {backbone_config['type']}")
        otx_registry, custom_imports = get_backbone_registry(backend)

        # Update out_indices of backbone
        if backbone_config["use_out_indices"]:
            model_in_indices = []
            if "backbone" in model_config.model:
                model_in_indices = model_config.model.backbone.get("out_indices", [])
            backbone_out_indices = backbone_config.get("out_indices", None)
            if not backbone_out_indices and model_in_indices:
                # Check out_indices vs num_stage
                backbone_config["out_indices"] = model_in_indices
        backbone_config.pop("use_out_indices", None)

        # Build Backbone
        backbone = build_from_cfg(backbone_config, otx_registry, None)
        if model_config.model.get("task", None) == str(TaskType.CLASSIFICATION).lower():
            # Update model layer's in/out configuration in ClsStage.configure_model
            out_channels = -1
            if hasattr(model_config.model, "head"):
                model_config.model.head.in_channels = -1
            # TODO: This is a hard coded part of the Transformer backbone and needs to be refactored.
            if backend == "mmcls" and backbone_class in TRANSFORMER_BACKBONES:
                if hasattr(model_config.model, "neck"):
                    model_config.model.neck = None
                if hasattr(model_config.model, "head"):
                    model_config.model.head["type"] = "VisionTransformerClsHead"
        else:
            # Need to update in/out channel configuration here
            out_channels = get_backbone_out_channels(backbone)
        update_channels(model_config, out_channels)

        # Update Model Configuration
        if backend in ("torchvision"):
            backbone_config["init_cfg"] = {"Pretrained": True}
        print(f"\tBackbone config: {backbone_config}")
        model_config.model.backbone = backbone_config
        model_config.load_from = None

        if custom_imports:
            model_config["custom_imports"] = dict(imports=custom_imports, allow_failed_imports=False)

        # Dump or create model config file
        if output_path is None:
            output_path = model_config_path
        model_config.dump(str(output_path))
        print(f"[*] Save model configuration: {str(output_path)}")
