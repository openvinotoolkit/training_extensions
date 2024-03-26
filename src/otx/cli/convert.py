# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Converter for Geti config."""

import argparse
import json
from pathlib import Path

import yaml

from otx.core.types.task import OTXTaskType
from otx.engine.utils.auto_configurator import AutoConfigurator

TEMPLATE_ID_DICT = {
    "Custom_Object_Detection_Gen3_ATSS": {
        "task": OTXTaskType.DETECTION,
        "model_name": "atss_mobilenetv2",
    },
    "Custom_Image_Classification_DeiT-Tiny": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "otx_deit_tiny",
    },
    "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50": {
        "task": OTXTaskType.INSTANCE_SEGMENTATION,
        "model_name": "maskrcnn_r50",
    },
    "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "litehrnet_18",
    },
}


def convert(geti_config_path: str) -> dict:
    """Convert Geti config to OTX recipe yaml file."""
    with Path(geti_config_path).open() as f:
        geti_config = json.load(f)
        hyperparameters = geti_config["hyperparameters"]
        param_dict = get_params(hyperparameters)
        task_info = TEMPLATE_ID_DICT[geti_config["model_template_id"]]
        if param_dict.get("enable_tiling", None):
            task_info["model_name"] += "_tile"
        default_config = AutoConfigurator(**task_info).config  # type: ignore[arg-type]
        update_params(default_config, param_dict)
        return default_config


def get_params(hyperparameters: dict) -> dict:
    """Get configuraable parameters from Geti config hyperparameters field."""
    param_dict = {}
    for param_name, param_info in hyperparameters.items():
        if isinstance(param_info, dict):
            if "value" in param_info:
                param_dict[param_name] = param_info["value"]
            else:
                param_dict = param_dict | get_params(param_info)

    return param_dict


def update_params(config: dict, param_dict: dict) -> None:  # noqa: C901
    """Update params of OTX recipe from Geit configurable params."""
    unused_params = []
    for param_name, param_value in param_dict.items():
        if param_name == "mem_cache_size":
            config["data"]["config"]["mem_cache_size"] = str(param_value / 1000000000) + "GB"
        elif param_name == "batch_size":
            config["data"]["config"]["train_subset"]["batch_size"] = param_value
        elif param_name == "inference_batch_size":
            config["data"]["config"]["val_subset"]["batch_size"] = param_value
            config["data"]["config"]["test_subset"]["batch_size"] = param_value
        elif param_name == "learning_rate":
            config["optimizer"]["init_args"]["lr"] = param_value
        elif param_name == "learning_rate_warmup_iters":
            scheduler_found = False
            for scheduler in config["scheduler"]:
                if scheduler["class_path"] == "otx.algo.schedulers.LinearWarmupScheduler":
                    scheduler_found = True
                    scheduler["init_args"]["num_warmup_steps"] = param_value
            if not scheduler_found:
                unused_params.append((param_name, param_value))
        elif param_name == "num_iters":
            config["max_epoch"] = param_value
        elif param_name == "num_workers":
            config["data"]["config"]["train_subset"]["num_workers"] = param_value
            config["data"]["config"]["val_subset"]["num_workers"] = param_value
            config["data"]["config"]["test_subset"]["num_workers"] = param_value
        elif param_name == "enable_early_stopping":
            idx = get_callback_idx(config["callbacks"], "lightning.pytorch.callbacks.EarlyStopping")
            if not param_value:
                if idx > 0:
                    config["callbacks"].pop(idx)
                else:
                    unused_params.append((param_name, param_value))
            # Add early stopping hook when enable_early_stopping is True
            # and there's no EarlyStopping callback.
        elif param_name == "early_stop_patience":
            callback_found = False
            for callback in config["callbacks"]:
                if callback["class_path"] == "lightning.pytorch.callbacks.EarlyStopping":
                    callback_found = True
                    callback["init_args"]["patience"] = param_value
                    break
            if not callback_found:
                unused_params.append((param_name, param_value))
        elif param_name == "use_adaptive_interval":
            idx = get_callback_idx(
                config["callbacks"],
                "otx.algo.callbacks.adaptive_train_scheduling.AdaptiveTrainScheduling",
            )
            if not param_value:
                if idx > 0:
                    config["callbacks"].pop(idx)
                else:
                    unused_params.append((param_name, param_value))
            # Add adative scheduling hook when use_adaptive_interval is True
            # and there's no AdaptiveTrainScheduling callback.
        elif param_name == "auto_num_workers":
            config["data"]["config"]["auto_num_workers"] = param_value
        elif param_name == "enable_tiling":
            config["data"]["config"]["tile_config"]["enable_tiler"] = param_value
        elif param_name == "enable_adaptive_params":
            config["data"]["config"]["tile_config"]["enable_adaptive_tiling"] = param_value
        elif param_name == "tile_size":
            config["data"]["config"]["tile_config"]["tile_size"] = (param_value, param_value)
        elif param_name == "tile_overlap":
            config["data"]["config"]["tile_config"]["overlap"] = param_value
        elif param_name == "tile_max_number":
            config["data"]["config"]["tile_config"]["max_num_instance"] = param_value
        elif param_name == "tile_sampling_ratio":
            config["data"]["config"]["tile_config"]["sampling_ratio"] = param_value
        elif param_name == "object_tile_ratio":
            config["data"]["config"]["tile_config"]["object_tile_ratio"] = param_value
        else:
            unused_params.append((param_name, param_value))
    print("Warning: These parameters are not updated")
    for param_name, param_value in unused_params:
        print("\t", param_name, ": ", param_value)


def get_callback_idx(callbacks: list, name: str) -> int:
    """Return required callbacks index from callback list."""
    for idx, callback in enumerate(callbacks):
        if callback["class_path"] == name:
            return idx
    return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Input Geti config")
    args = parser.parse_args()
    default_config = convert(args.config)
    yaml_config = yaml.dump(default_config)
    with Path("converted_config.yaml").open("w") as f:
        f.write(yaml_config)
