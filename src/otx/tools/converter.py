# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Converter for v1 config."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from warnings import warn

from jsonargparse import ArgumentParser, Namespace

from otx.core.config.data import SamplerConfig, SubsetConfig, TileConfig, UnlabeledDataConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel
from otx.core.types import PathLike
from otx.core.types.task import OTXTaskType
from otx.engine import Engine
from otx.engine.utils.auto_configurator import AutoConfigurator

TEMPLATE_ID_DICT = {
    # MULTI_CLASS_CLS
    "Custom_Image_Classification_DeiT-Tiny": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "deit_tiny",
    },
    "Custom_Image_Classification_EfficinetNet-B0": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "efficientnet_b0",
    },
    "Custom_Image_Classification_EfficientNet-V2-S": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "efficientnet_v2",
    },
    "Custom_Image_Classification_MobileNet-V3-large-1x": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "mobilenet_v3_large",
    },
    "Custom_Image_Classification_EfficinetNet-B3": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "tv_efficientnet_b3",
    },
    "Custom_Image_Classification_EfficinetNet-V2-L": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "tv_efficientnet_v2_l",
    },
    "Custom_Image_Classification_MobileNet-V3-small": {
        "task": OTXTaskType.MULTI_CLASS_CLS,
        "model_name": "tv_mobilenet_v3_small",
    },
    # DETECTION
    "Custom_Object_Detection_Gen3_ATSS": {
        "task": OTXTaskType.DETECTION,
        "model_name": "atss_mobilenetv2",
    },
    "Object_Detection_ResNeXt101_ATSS": {
        "task": OTXTaskType.DETECTION,
        "model_name": "atss_resnext101",
    },
    "Custom_Object_Detection_Gen3_SSD": {
        "task": OTXTaskType.DETECTION,
        "model_name": "ssd_mobilenetv2",
    },
    "Object_Detection_YOLOX_X": {
        "task": OTXTaskType.DETECTION,
        "model_name": "yolox_x",
    },
    "Object_Detection_YOLOX_L": {
        "task": OTXTaskType.DETECTION,
        "model_name": "yolox_l",
    },
    "Object_Detection_YOLOX_S": {
        "task": OTXTaskType.DETECTION,
        "model_name": "yolox_s",
    },
    "Custom_Object_Detection_YOLOX": {
        "task": OTXTaskType.DETECTION,
        "model_name": "yolox_tiny",
    },
    "Object_Detection_RTDetr_18": {
        "task": OTXTaskType.DETECTION,
        "model_name": "rtdetr_18",
    },
    "Object_Detection_RTDetr_50": {
        "task": OTXTaskType.DETECTION,
        "model_name": "rtdetr_50",
    },
    "Object_Detection_RTDetr_101": {
        "task": OTXTaskType.DETECTION,
        "model_name": "rtdetr_101",
    },
    "Object_Detection_RTMDet_tiny": {
        "task": OTXTaskType.DETECTION,
        "model_name": "rtmdet_tiny",
    },
    # INSTANCE_SEGMENTATION
    "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50": {
        "task": OTXTaskType.INSTANCE_SEGMENTATION,
        "model_name": "maskrcnn_r50",
    },
    "Custom_Counting_Instance_Segmentation_MaskRCNN_SwinT_FP16": {
        "task": OTXTaskType.INSTANCE_SEGMENTATION,
        "model_name": "maskrcnn_swint",
    },
    "Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B": {
        "task": OTXTaskType.INSTANCE_SEGMENTATION,
        "model_name": "maskrcnn_efficientnetb2b",
    },
    "Custom_Instance_Segmentation_RTMDet_tiny": {
        "task": OTXTaskType.INSTANCE_SEGMENTATION,
        "model_name": "rtmdet_inst_tiny",
    },
    # ROTATED_DETECTION
    "Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50": {
        "task": OTXTaskType.ROTATED_DETECTION,
        "model_name": "maskrcnn_r50",
    },
    "Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B": {
        "task": OTXTaskType.ROTATED_DETECTION,
        "model_name": "maskrcnn_efficientnetb2b",
    },
    # SEMANTIC_SEGMENTATION
    "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "litehrnet_18",
    },
    "Custom_Semantic_Segmentation_Lite-HRNet-18_OCR": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "litehrnet_18",
    },
    "Custom_Semantic_Segmentation_Lite-HRNet-s-mod2_OCR": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "litehrnet_s",
    },
    "Custom_Semantic_Segmentation_Lite-HRNet-x-mod3_OCR": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "litehrnet_x",
    },
    "Custom_Semantic_Segmentation_SegNext_t": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "segnext_t",
    },
    "Custom_Semantic_Segmentation_SegNext_s": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "segnext_s",
    },
    "Custom_Semantic_Segmentation_SegNext_B": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "segnext_b",
    },
    "Custom_Semantic_Segmentation_DINOV2_S": {
        "task": OTXTaskType.SEMANTIC_SEGMENTATION,
        "model_name": "dino_v2",
    },
    # ANOMALY_CLASSIFICATION
    "ote_anomaly_classification_padim": {
        "task": OTXTaskType.ANOMALY_CLASSIFICATION,
        "model_name": "padim",
    },
    "ote_anomaly_classification_stfpm": {
        "task": OTXTaskType.ANOMALY_CLASSIFICATION,
        "model_name": "stfpm",
    },
    # ANOMALY_DETECTION
    "ote_anomaly_detection_padim": {
        "task": OTXTaskType.ANOMALY_DETECTION,
        "model_name": "padim",
    },
    "ote_anomaly_detection_stfpm": {
        "task": OTXTaskType.ANOMALY_DETECTION,
        "model_name": "stfpm",
    },
    # ANOMALY_SEGMENTATION
    "ote_anomaly_segmentation_padim": {
        "task": OTXTaskType.ANOMALY_SEGMENTATION,
        "model_name": "padim",
    },
    "ote_anomaly_segmentation_stfpm": {
        "task": OTXTaskType.ANOMALY_SEGMENTATION,
        "model_name": "stfpm",
    },
}


class ConfigConverter:
    """Convert ModelTemplate for OTX v1 to OTX v2 recipe dictionary.

    This class is used to convert ModelTemplate for OTX v1 to OTX v2 recipe dictionary.

    Example:
        The following examples show how to use the Converter class.
        We expect a config file with ModelTemplate information in json form.

        Convert template.json to dictionary::

            converter = ConfigConverter()
            config = converter.convert("template.json")

        Instantiate an object from the configuration dictionary::

            engine, train_kwargs = converter.instantiate(
                config=config,
                work_dir="otx-workspace",
                data_root="tests/assets/car_tree_bug",
            )

        Train the model::

            engine.train(**train_kwargs)
    """

    @staticmethod
    def convert(config_path: str, task: OTXTaskType | None = None) -> dict:
        """Convert a configuration file to a default configuration dictionary.

        Args:
            config_path (str): The path to the configuration file.
            task (OTXTaskType | None): Value to override the task.

        Returns:
            dict: The default configuration dictionary.

        """
        with Path(config_path).open() as f:
            template_config = json.load(f)

        hyperparameters = template_config["hyperparameters"]
        param_dict = ConfigConverter._get_params(hyperparameters)

        task_info = TEMPLATE_ID_DICT[template_config["model_template_id"]]
        if param_dict.get("enable_tiling", None) and not task_info["model_name"].endswith("_tile"):
            task_info["model_name"] += "_tile"
        if task is not None:
            task_info["task"] = task
        default_config = ConfigConverter._get_default_config(task_info)
        ConfigConverter._update_params(default_config, param_dict)
        if (hpo_time_ratio := template_config.get("hpo_parameters", {}).get("hpo_time_ratio")) is not None:
            default_config["hpo_config.expected_time_ratio"] = hpo_time_ratio
        ConfigConverter._remove_unused_key(default_config)
        return default_config

    @staticmethod
    def _get_default_config(task_info: dict) -> dict:
        """Return default otx conifg for template use."""
        return AutoConfigurator(**task_info).config  # type: ignore[arg-type]

    @staticmethod
    def _get_params(hyperparameters: dict) -> dict:
        """Get configuraable parameters from ModelTemplate config hyperparameters field."""
        param_dict = {}
        for param_name, param_info in hyperparameters.items():
            if isinstance(param_info, dict):
                if "value" in param_info:
                    param_dict[param_name] = param_info["value"]
                else:
                    param_dict = param_dict | ConfigConverter._get_params(param_info)

        return param_dict

    @staticmethod
    def _update_params(config: dict, param_dict: dict) -> None:  # noqa: C901
        """Update params of OTX recipe from Geit configurable params."""
        unused_params = deepcopy(param_dict)

        def update_mem_cache_size(param_value: int) -> None:
            config["data"]["mem_cache_size"] = f"{int(param_value / 1000000)}MB"

        def update_batch_size(param_value: int) -> None:
            config["data"]["train_subset"]["batch_size"] = param_value

        def update_inference_batch_size(param_value: int) -> None:
            config["data"]["val_subset"]["batch_size"] = param_value
            config["data"]["test_subset"]["batch_size"] = param_value

        def update_learning_rate(param_value: float) -> None:
            optimizer = config["model"]["init_args"]["optimizer"]
            if isinstance(optimizer, dict) and "init_args" in optimizer:
                optimizer["init_args"]["lr"] = param_value
            else:
                warn("Warning: learning_rate is not updated", stacklevel=1)

        def update_learning_rate_warmup_iters(param_value: int) -> None:
            scheduler = config["model"]["init_args"]["scheduler"]
            if (
                isinstance(scheduler, dict)
                and "class_path" in scheduler
                and scheduler["class_path"] == "otx.core.schedulers.LinearWarmupSchedulerCallable"
            ):
                scheduler["init_args"]["num_warmup_steps"] = param_value
            else:
                warn("Warning: learning_rate_warmup_iters is not updated", stacklevel=1)

        def update_num_iters(param_value: int) -> None:
            config["max_epochs"] = param_value

        def update_num_workers(param_value: int) -> None:
            config["data"]["train_subset"]["num_workers"] = param_value
            config["data"]["val_subset"]["num_workers"] = param_value
            config["data"]["test_subset"]["num_workers"] = param_value

        def update_enable_early_stopping(param_value: bool) -> None:
            idx = ConfigConverter._get_callback_idx(config["callbacks"], "lightning.pytorch.callbacks.EarlyStopping")
            if not param_value and idx > -1:
                config["callbacks"].pop(idx)

        def update_early_stop_patience(param_value: int) -> None:
            for callback in config["callbacks"]:
                if callback["class_path"] == "lightning.pytorch.callbacks.EarlyStopping":
                    callback["init_args"]["patience"] = param_value
                    break

        def update_use_adaptive_interval(param_value: bool) -> None:
            idx = ConfigConverter._get_callback_idx(
                config["callbacks"],
                "otx.algo.callbacks.adaptive_train_scheduling.AdaptiveTrainScheduling",
            )
            if not param_value and idx > -1:
                config["callbacks"].pop(idx)

        def update_auto_num_workers(param_value: bool) -> None:
            config["data"]["auto_num_workers"] = param_value

        def update_auto_adapt_batch_size(param_value: str) -> None:
            config["adaptive_bs"] = param_value

        def update_enable_tiling(param_value: bool) -> None:
            config["data"]["tile_config"]["enable_tiler"] = param_value
            if param_value:
                config["data"]["tile_config"]["enable_adaptive_tiling"] = param_dict["enable_adaptive_params"]
                config["data"]["tile_config"]["tile_size"] = (
                    param_dict["tile_size"],
                    param_dict["tile_size"],
                )
                config["data"]["tile_config"]["overlap"] = param_dict["tile_overlap"]
                config["data"]["tile_config"]["max_num_instances"] = param_dict["tile_max_number"]
                config["data"]["tile_config"]["sampling_ratio"] = param_dict["tile_sampling_ratio"]
                config["data"]["tile_config"]["object_tile_ratio"] = param_dict["object_tile_ratio"]
                tile_params = [
                    "enable_adaptive_params",
                    "tile_size",
                    "tile_overlap",
                    "tile_max_number",
                    "tile_sampling_ratio",
                    "object_tile_ratio",
                ]
                for tile_param in tile_params:
                    unused_params.pop(tile_param)

        param_update_funcs = {
            "mem_cache_size": update_mem_cache_size,
            "batch_size": update_batch_size,
            "inference_batch_size": update_inference_batch_size,
            "learning_rate": update_learning_rate,
            "learning_rate_warmup_iters": update_learning_rate_warmup_iters,
            "num_iters": update_num_iters,
            "num_workers": update_num_workers,
            "enable_early_stopping": update_enable_early_stopping,
            "early_stop_patience": update_early_stop_patience,
            "use_adaptive_interval": update_use_adaptive_interval,
            "auto_num_workers": update_auto_num_workers,
            "enable_tiling": update_enable_tiling,
            "auto_adapt_batch_size": update_auto_adapt_batch_size,
        }
        for param_name, param_value in param_dict.items():
            update_func = param_update_funcs.get(param_name)
            if update_func:
                update_func(param_value)  # type: ignore[operator]
                unused_params.pop(param_name)

        warn("Warning: These parameters are not updated", stacklevel=1)
        for param_name, param_value in unused_params.items():
            print(f"\t {param_name}: {param_value}")

    @staticmethod
    def _get_callback_idx(callbacks: list, name: str) -> int:
        """Return required callbacks index from callback list."""
        for idx, callback in enumerate(callbacks):
            if callback["class_path"] == name:
                return idx
        return -1

    @staticmethod
    def _remove_unused_key(config: dict) -> None:
        """Remove unused keys from the config dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        config.pop("config")  # Remove config key that for CLI
        config["data"].pop("__path__")  # Remove __path__ key that for CLI overriding

    @staticmethod
    def instantiate(
        config: dict,
        work_dir: PathLike | None = None,
        data_root: PathLike | None = None,
        **kwargs,
    ) -> tuple[Engine, dict[str, Any]]:
        """Instantiate an object from the configuration dictionary.

        Args:
            config (dict): The configuration dictionary.
            work_dir (PathLike): Path to the working directory.
            data_root (PathLike): The root directory for data.

        Returns:
            tuple: A tuple containing the engine and the train kwargs dictionary.
        """
        config.update(kwargs)

        # Instantiate datamodule
        data_config = config.pop("data")
        if data_root is not None:
            data_config["data_root"] = data_root

        train_config = data_config.pop("train_subset")
        val_config = data_config.pop("val_subset")
        test_config = data_config.pop("test_subset")
        unlabeled_config = data_config.pop("unlabeled_subset")
        datamodule = OTXDataModule(
            train_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
            val_subset=SubsetConfig(sampler=SamplerConfig(**val_config.pop("sampler", {})), **val_config),
            test_subset=SubsetConfig(sampler=SamplerConfig(**test_config.pop("sampler", {})), **test_config),
            unlabeled_subset=UnlabeledDataConfig(**unlabeled_config),
            tile_config=TileConfig(**data_config.pop("tile_config", {})),
            **data_config,
        )

        # Update num_classes & Instantiate Model
        model_config = config.pop("model")
        model_config["init_args"]["label_info"] = datamodule.label_info

        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(OTXModel, "model", required=False, fail_untyped=False, skip={"label_info"})
        model = model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

        # Instantiate Engine
        config_work_dir = config.pop("work_dir", config["engine"].pop("work_dir", None))
        config["engine"]["work_dir"] = work_dir if work_dir is not None else config_work_dir
        engine = Engine(
            model=model,
            datamodule=datamodule,
            **config.pop("engine"),
        )

        # Instantiate Engine.train Arguments
        engine_parser = ArgumentParser()
        train_arguments = engine_parser.add_method_arguments(
            Engine,
            "train",
            skip={"accelerator", "devices"},
            fail_untyped=False,
        )
        # Update callbacks & logger dir as engine.work_dir
        for callback in config["callbacks"]:
            if "dirpath" in callback["init_args"]:
                callback["init_args"]["dirpath"] = engine.work_dir
        for logger in config["logger"]:
            if "save_dir" in logger["init_args"]:
                logger["init_args"]["save_dir"] = engine.work_dir
            if "log_dir" in logger["init_args"]:
                logger["init_args"]["log_dir"] = engine.work_dir
        instantiated_kwargs = engine_parser.instantiate_classes(Namespace(**config))

        train_kwargs = {k: v for k, v in instantiated_kwargs.items() if k in train_arguments}

        return engine, train_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Input ModelTemplate config")
    parser.add_argument("-i", "--data_root", help="Input dataset root path")
    parser.add_argument("-o", "--work_dir", help="Input work directory path")
    args = parser.parse_args()
    otx_config = ConfigConverter.convert(config_path=args.config)
    engine, train_kwargs = ConfigConverter.instantiate(
        config=otx_config,
        data_root=args.data_root,
        work_dir=args.work_dir,
    )
    engine.train(**train_kwargs)
