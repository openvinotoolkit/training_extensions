"""Builder Class for training template."""

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
import os
import shutil

import mmcv
import torch
from mmcv.utils import Registry, build_from_cfg
from mpa.utils.config_utils import MPAConfig

from otx.cli.registry import Registry as OTXRegistry
from otx.cli.utils.importing import get_backbone_registry

DEFAULT_MODEL_TEMPLATE_ID = {
    "classification": "Custom_Image_Classification_EfficinetNet-B0",
    "detection": "Custom_Object_Detection_Gen3_ATSS",
    "segmentation": "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR",
}


# pylint: disable=too-many-locals


def get_backbone_out_channels(backbone):
    """Get output channels of backbone using fake data."""
    out_channels = []
    input_size = backbone.input_size if hasattr(backbone, "input_size") else 64
    fake_data = torch.rand(1, 3, input_size, input_size)
    outputs = backbone(fake_data)
    for out in outputs:
        out_channels.append(out.shape[1])
    return out_channels


def update_backbone_args(backbone_config, registry, skip_missing=False):
    """Update Backbone required arguments."""
    backbone_function = registry.get(backbone_config["type"])
    required_args, missing_args = [], []
    args_signature = inspect.signature(backbone_function)
    for arg_key, arg_value in args_signature.parameters.items():
        if arg_value.default is inspect.Parameter.empty:
            # TODO: How to get argument type (or hint or format)
            required_args.append(arg_key)
            continue
        # Update Backbone config to defaults
        backbone_config[arg_key] = arg_value.default

    # Get args from parents
    if len(backbone_function.__bases__):
        parent_args_signature = inspect.signature(backbone_function.__bases__[0])
        for arg_key, arg_value in parent_args_signature.parameters.items():
            if arg_value.default is inspect.Parameter.empty and arg_key not in required_args:
                required_args.append(arg_key)
                continue
            # Update Backbone config to defaults
            if arg_key not in backbone_config and arg_key in ("out_indices"):
                backbone_config[arg_key] = arg_value.default

    for arg in required_args:
        if arg not in backbone_config and arg not in ("args", "kwargs", "self"):
            missing_args.append(arg)
    if len(missing_args) > 0 and not skip_missing:
        raise ValueError(
            f"[otx build] {backbone_config['type']} requires the argument : {missing_args}"
            f"\n[otx build] Please refer to {inspect.getfile(backbone_function)}"
        )
    if len(missing_args) > 0 and skip_missing:
        for arg in missing_args:
            backbone_config[arg] = "!!!!!!!!!!!INPUT_HERE!!!!!!!!!!!"
        print(
            f"[otx build] {backbone_config['type']} requires the argument : {missing_args}"
            f"\n[otx build] Please refer to {inspect.getfile(backbone_function)}"
        )


def update_in_channel(model_config, out_channels):
    """Update in_channel of head or neck."""
    if hasattr(model_config.model, "neck"):
        print(f"\tUpdate model.neck.in_channels: {out_channels}")
        model_config.model.neck.in_channels = out_channels
    elif hasattr(model_config.model, "bbox_head"):
        print(f"\tUpdate model.bbox_head.in_channels: {out_channels}")
        model_config.model.bbox_head.in_channels = out_channels
        if hasattr(model_config.model.bbox_head, "anchor_generator"):
            strides = get_max_strides(out_channels)
            model_config.model.bbox_head.anchor_generator.strides = strides
            print(f"\tUpdate model.bbox_head.anchor_generator.strides: {strides}")
    elif hasattr(model_config.model, "decode_head"):
        print(f"\tUpdate model.decode_head.in_channels: {out_channels}")
        model_config.model.decode_head.in_channels = out_channels
    elif hasattr(model_config.model, "head"):
        print(f"\tUpdate model.head.in_channels: {out_channels}")
        model_config.model.head.in_channels = out_channels


def get_max_strides(in_channels):
    """Get max strides value from in_channels."""
    strides = []
    for channel in in_channels:
        i = 0
        while channel % 2 == 0 and i < 5:
            i += 1
            channel /= 2
        strides.append(2**i)
    return strides


class Builder:
    """Class that implements a model templates registry."""

    def __init__(self):
        pass

    def build_task_config(self, task_type, model_type=None, workspace_path=None):
        """Build OTX workspace with Template configs from task type."""

        # Create OTX-workspace
        if workspace_path is None:
            workspace_path = f"./otx-workspace-{task_type}"
        os.makedirs(workspace_path, exist_ok=True)

        # Load & Save Model Template
        otx_registry = OTXRegistry("otx").filter(task_type=task_type)
        if model_type:
            template = [temp for temp in otx_registry.templates if temp.name.lower() == model_type.lower()]
            if len(template) == 0:
                raise ValueError(
                    f"[otx build] {model_type} is not a type supported by OTX {task_type}."
                    f"\n[otx build] Please refer to 'otx find --template --task_type {task_type}'"
                )
            template = template[0]
        else:
            template = otx_registry.get(DEFAULT_MODEL_TEMPLATE_ID[task_type])
        template_dir = os.path.dirname(template.model_template_path)

        # Copy task configuration file
        task_configuration_path = os.path.join("/".join(template_dir.split("/")[:-1]), "configuration.yaml")
        shutil.copyfile(task_configuration_path, os.path.join(workspace_path, "configuration.yaml"))
        # Load & Save Model Template
        template_config = MPAConfig.fromfile(template.model_template_path)
        template_config.hyper_parameters.base_path = "./configuration.yaml"
        template_config.dump(os.path.join(workspace_path, "template.yaml"))

        # Load & Save Model config
        model_config = MPAConfig.fromfile(os.path.join(template_dir, "model.py"))
        model_config.dump(os.path.join(workspace_path, "model.py"))

        # Copy Data config
        if os.path.exists(os.path.join(template_dir, "data_pipeline.py")):
            data_pipeline_config = MPAConfig.fromfile(os.path.join(template_dir, "data_pipeline.py"))
            data_pipeline_config.dump(os.path.join(workspace_path, "data_pipeline.py"))
        data_subset_format = {"ann-files": None, "data-roots": None}
        data_config = {"data": {subset: data_subset_format.copy() for subset in ("train", "val", "test")}}
        data_config["data"]["unlabeled"] = {"file-list": None, "data-roots": None}
        mmcv.dump(data_config, os.path.join(workspace_path, "data.yaml"))

        print(f"[otx build] Create OTX workspace: {workspace_path}")
        print(f"\tTask Type: {template.task_type}")
        print(f"\tLoad Model Template ID: {template.model_template_id}")
        print(f"\tLoad Model Name: {template.name}")
        print(f"\tYou need to edit that file: {os.path.join(workspace_path, 'data.yaml')}")

    def build_backbone_config(self, backbone_type, output_path):
        """Build Backbone configs from backbone type.

        backbone_type:
        output_path: new model.py output path (os.path)
        """
        print(f"[otx build] Backbone Config: {backbone_type}")

        backend, _ = Registry.split_scope_key(backbone_type)
        backbone_config = {"type": backbone_type}
        otx_registry, _ = get_backbone_registry(backend)
        update_backbone_args(backbone_config, otx_registry, skip_missing=True)
        if output_path.endswith((".yml", ".yaml", ".json")):
            mmcv.dump({"backbone": backbone_config}, output_path)
            print(f"[otx build] Save backbone configuration: {output_path}")

    def build_model_config(self, model_config_path, backbone_config_path, output_path=None):
        """Build model & update backbone configs.

        model_config_path: model.py or model.yaml (os.path)
        backbone_config_path: backbone id (or name?) (os.path)
        output_path: new model.py output path (os.path)
        """
        print(f"[otx build] Model Config with {backbone_config_path}")

        # Get Model config from model config file
        model_in_indices = None
        if os.path.exists(model_config_path):
            model_config = MPAConfig.fromfile(model_config_path)
            print(f"\tTarget Model: {model_config.model.type}")
            if "backbone" in model_config.model:
                model_in_indices = model_config.model.backbone.get("out_indices", None)
        else:
            raise ValueError(f"[otx build] The model is not properly defined or not found: {model_config_path}")

        # Get Backbone config from config file
        if os.path.exists(backbone_config_path):
            backbone_config = mmcv.load(backbone_config_path)
        else:
            raise ValueError(f"[otx build] The backbone is not found: {backbone_config_path}")

        backbone_pretrained = None
        if "model" in backbone_config:
            backbone_config = backbone_config["model"]
            backbone_pretrained = backbone_config.get("pretrained", None)
        if "backbone" in backbone_config:
            backbone_config = backbone_config["backbone"]

        # Get Backbone configuration
        backend, _ = Registry.split_scope_key(backbone_config["type"])
        print(f"\tTarget Backbone: {backbone_config['type']}")
        otx_registry, custom_imports = get_backbone_registry(backend)
        update_backbone_args(backbone_config, otx_registry)
        backbone_out_indices = backbone_config.get("out_indices", None)
        if backbone_out_indices and model_in_indices and len(backbone_out_indices) != len(model_in_indices):
            backbone_out_indices = backbone_out_indices[-len(model_in_indices) :]
        if not backbone_out_indices and model_in_indices:
            # Check out_indices vs num_stage
            backbone_config["out_indices"] = model_in_indices
        print(f"\tBackbone config: {backbone_config}")

        # Build Backbone
        backbone = build_from_cfg(backbone_config, otx_registry, None)
        out_channels = get_backbone_out_channels(backbone)

        # Update Model Configuration
        model_config.model.backbone = backbone_config
        model_config.load_from = None
        if backbone_pretrained:
            model_config.model.pretrained = backbone_pretrained
        if custom_imports:
            model_config["custom_imports"] = dict(imports=custom_imports, allow_failed_imports=False)
        update_in_channel(model_config, out_channels)

        # Dump or create model config file
        if output_path is None:
            base_dir = os.path.abspath(os.path.dirname(backbone_config_path))
            model_file_name = "model.py"
            output_path = os.path.join(base_dir, model_file_name)
        model_config.dump(output_path)
        print(f"[otx build] Save model configuration: {output_path}")
