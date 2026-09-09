# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GetiConfigConverter.convert (Geti backend version)."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import patch

import pytest

from app.execution.common.geti_config_converter import GetiConfigConverter, HyperparametersUpdater, TransformsUpdater

EARLY_STOPPING_CLASS_PATH = "getitune.backend.lightning.callbacks.adaptive_early_stopping.EarlyStoppingWithWarmup"

EARLY_STOPPING_CALLBACK = {
    "class_path": EARLY_STOPPING_CLASS_PATH,
    "init_args": {"patience": 10, "monitor": "val/map_50"},
}

CHECKPOINT_CALLBACK = {
    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
    "init_args": {"dirpath": "", "monitor": "val/map_50"},
}
AUG_SCHEDULER_CLASS_PATH = "getitune.backend.lightning.callbacks.aug_scheduler.AugmentationSchedulerCallback"

AUG_SCHEDULER_CALLBACK = {
    "class_path": AUG_SCHEDULER_CLASS_PATH,
    "init_args": {
        "data_aug_switch": {"class_path": "getitune.backend.lightning.callbacks.aug_scheduler.DataAugSwitch"}
    },
}


def _make_getitune_config(**overrides: Any) -> dict:
    """Build a minimal getitune recipe config dict with sane defaults.

    Override any key via keyword arguments.
    """
    cfg: dict[str, Any] = {
        "config": ["some_path"],
        "max_epochs": 200,
        "precision": "16-mixed",
        "model": {
            "class_path": "getitune.backend.lightning.models.detection.atss.ATSS",
            "init_args": {
                "model_name": "atss_mobilenetv2",
                "label_info": 80,
                "optimizer": {
                    "class_path": "torch.optim.SGD",
                    "init_args": {"lr": 0.004, "momentum": 0.9, "weight_decay": 0.0001},
                },
                "scheduler": {
                    "class_path": "getitune.backend.lightning.schedulers.LinearWarmupSchedulerCallable",
                    "init_args": {
                        "num_warmup_steps": 0,
                        "main_scheduler_callable": {
                            "class_path": "lightning.pytorch.cli.ReduceLROnPlateau",
                            "init_args": {
                                "mode": "max",
                                "factor": 0.1,
                                "patience": 4,
                                "monitor": "val/map_50",
                            },
                        },
                    },
                },
            },
        },
        "engine": {"device": "auto"},
        "callbacks": [
            copy.deepcopy(EARLY_STOPPING_CALLBACK),
            copy.deepcopy(CHECKPOINT_CALLBACK),
        ],
        "data": {
            "__path__": "some/path.yaml",
            "input_size": [800, 992],
            "tile_config": {"enable_tiler": False, "enable_adaptive_tiling": False},
            "train_subset": {
                "batch_size": 8,
                "augmentations_cpu": [
                    {"class_path": "getitune.data.augmentation.transforms.RandomIoUCrop"},
                    {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": "$(input_size)"},
                    },
                ],
                "augmentations_gpu": [
                    {"class_path": "kornia.augmentation.RandomHorizontalFlip", "init_args": {"p": 0.5}},
                    {
                        "class_path": "kornia.augmentation.Normalize",
                        "init_args": {
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    },
                ],
            },
            "val_subset": {"batch_size": 8},
            "test_subset": {"batch_size": 8},
        },
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _make_geti_config(
    model_manifest_id: str = "object-detection-atss-mobilenet-v2",
    sub_task_type: str | None = None,
    hyper_parameters: dict | None = None,
    task_level_parameters: dict | None = None,
) -> dict:
    """Build a Geti training configuration dict (the input to convert())."""
    cfg: dict[str, Any] = {
        "model_manifest_id": model_manifest_id,
        "sub_task_type": sub_task_type,
        "hyper_parameters": hyper_parameters or {},
        "task_level_parameters": task_level_parameters or {},
    }
    return cfg


def _make_timm_getitune_config(**overrides: Any) -> dict:
    """Minimal timm_generic.yaml-shaped config (optimizer present, DYNAMIC placeholders)."""
    imagenet_normalize = {
        "class_path": "kornia.augmentation.Normalize",
        "init_args": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    resize = {
        "class_path": "torchvision.transforms.v2.Resize",
        "init_args": {"size": "$(input_size)"},
    }
    cfg: dict[str, Any] = {
        "config": ["some_path"],
        "max_epochs": 90,
        "model": {
            "class_path": (
                "getitune.backend.lightning.models.classification.multiclass_models.timm_model.TimmModelMulticlassCls"
            ),
            "init_args": {
                "label_info": 1000,
                "model_name": "DYNAMIC",
                "optimizer": {
                    "class_path": "getitune.backend.lightning.models.classification.optimizers.timm.TimmOptimizer",
                    "init_args": {"lr": float("nan"), "weight_decay": float("nan")},
                },
            },
        },
        "engine": {"device": "auto"},
        "callbacks": [],
        "data": {
            "__path__": "some/path.yaml",
            "input_size": [224, 224],
            "tile_config": {"enable_tiler": False, "enable_adaptive_tiling": False},
            "train_subset": {
                "batch_size": 32,
                "augmentations_cpu": [copy.deepcopy(resize)],
                "augmentations_gpu": [copy.deepcopy(imagenet_normalize)],
            },
            "val_subset": {
                "batch_size": 32,
                "augmentations_cpu": [copy.deepcopy(resize)],
                "augmentations_gpu": [copy.deepcopy(imagenet_normalize)],
            },
            "test_subset": {
                "batch_size": 32,
                "augmentations_cpu": [copy.deepcopy(resize)],
                "augmentations_gpu": [copy.deepcopy(imagenet_normalize)],
            },
        },
    }
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


class TestGetiConfigConverterConvert:
    """Tests for GetiConfigConverter.convert."""

    def test_convert_returns_config_without_cli_keys(self) -> None:
        """convert() should strip the 'config' and '__path__' keys."""
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config()

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert "config" not in result
        assert "__path__" not in result["data"]

    def test_convert_applies_learning_rate(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(hyper_parameters={"training": {"learning_rate": 0.01}})

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.01

    def test_convert_ultralytics_recipe_produces_backend_tagged_config(self) -> None:
        """Ultralytics recipes are converted via the library-side UltralyticsConfigurator."""
        geti_cfg = _make_geti_config(
            model_manifest_id="object-detection-yolo26-n",
            hyper_parameters={"training": {"learning_rate": 0.002, "batch_size": 4, "max_epochs": 10}},
        )

        result = GetiConfigConverter.convert(geti_cfg)

        assert result["backend"] == "ultralytics"
        assert result["model"]["init_args"]["model_name"] == "yolo26n.yaml"
        assert result["training"]["lr0"] == 0.002
        assert result["training"]["batch"] == 4
        assert result["training"]["epochs"] == 10
        assert result["data"]["train_subset"]["batch_size"] == 4

    def test_convert_multilabel_yolo26_uses_multilabel_recipe(self) -> None:
        """Non-exclusive YOLO26 classification must use the BCE-loss model recipe."""
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-yolo26-m",
            sub_task_type="MULTI_LABEL_CLS",
        )

        result = GetiConfigConverter.convert(geti_cfg)

        assert result["task"] == "MULTI_LABEL_CLS"
        assert result["model"]["class_path"] == (
            "getitune.backend.ultralytics.models.classification.UltralyticsMultiLabelClsModel"
        )

    def test_convert_multiclass_yolo26_keeps_multiclass_recipe(self) -> None:
        """Exclusive YOLO26 classification must retain the standard softmax recipe."""
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-yolo26-m",
            sub_task_type="MULTI_CLASS_CLS",
        )

        result = GetiConfigConverter.convert(geti_cfg)

        assert result["task"] == "MULTI_CLASS_CLS"
        assert result["model"]["class_path"] == (
            "getitune.backend.ultralytics.models.classification.UltralyticsMultiClassClsModel"
        )

    def test_convert_ultralytics_applies_augmentations_via_transforms_updater(self) -> None:
        """Augmentation hyper_parameters should flow through the shared TransformsUpdater for Ultralytics."""
        geti_cfg = _make_geti_config(
            model_manifest_id="object-detection-yolo26-n",
            hyper_parameters={
                "dataset_preparation": {
                    "augmentation": {
                        "iou_random_crop": {"enable": False},
                    }
                }
            },
        )

        result = GetiConfigConverter.convert(geti_cfg)

        assert result["backend"] == "ultralytics"
        # iou_random_crop should have been removed by TransformsUpdater
        cpu_augs = result["data"]["train_subset"]["augmentations_cpu"]
        crop = [a for a in cpu_augs if "RandomIoUCrop" in a.get("class_path", "")]
        assert len(crop) == 0

    def test_convert_ultralytics_applies_tiling_via_transforms_updater(self) -> None:
        """Tiling hyper_parameters should flow through the shared TransformsUpdater for Ultralytics."""
        geti_cfg = _make_geti_config(
            model_manifest_id="object-detection-yolo26-n",
            hyper_parameters={
                "dataset_preparation": {
                    "augmentation": {
                        "tiling": {
                            "enable": True,
                            "enable_adaptive_tiling": True,
                            "tile_size": 512,
                            "tile_overlap": 0.3,
                        }
                    }
                }
            },
        )

        result = GetiConfigConverter.convert(geti_cfg)

        assert result["backend"] == "ultralytics"
        tc = result["data"]["tile_config"]
        assert tc["enable_tiler"] is True
        assert tc["enable_adaptive_tiling"] is True
        assert tc["tile_size"] == (512, 512)
        assert tc["overlap"] == 0.3

    def test_convert_applies_batch_size(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(hyper_parameters={"training": {"batch_size": 16}})

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["data"]["train_subset"]["batch_size"] == 16
        assert result["data"]["val_subset"]["batch_size"] == 16

    def test_convert_applies_max_epochs(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(hyper_parameters={"training": {"max_epochs": 50}})

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["max_epochs"] == 50

    def test_convert_applies_early_stopping_patience(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            hyper_parameters={"training": {"early_stopping": {"enable": True, "patience": 20}}}
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        idx = GetiConfigConverter.get_callback_idx(result["callbacks"], EARLY_STOPPING_CLASS_PATH)
        assert idx >= 0
        assert result["callbacks"][idx]["init_args"]["patience"] == 20

    def test_convert_removes_early_stopping_when_disabled(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            hyper_parameters={"training": {"early_stopping": {"enable": False, "patience": 10}}}
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        idx = GetiConfigConverter.get_callback_idx(result["callbacks"], EARLY_STOPPING_CLASS_PATH)
        assert idx == -1

    def test_convert_excludes_deim_framework_when_tiling_enabled(self) -> None:
        """DEIM is incompatible with tiling: enabling tiling must exclude the DEIM scheduler callback."""
        getitune_cfg = _make_getitune_config(
            callbacks=[
                copy.deepcopy(EARLY_STOPPING_CALLBACK),
                copy.deepcopy(CHECKPOINT_CALLBACK),
                copy.deepcopy(AUG_SCHEDULER_CALLBACK),
            ]
        )
        geti_cfg = _make_geti_config(
            model_manifest_id="object-detection-dfine-m",
            hyper_parameters={
                "dataset_preparation": {
                    "augmentation": {
                        "deim_framework": True,
                        "tiling": {
                            "enable": True,
                            "enable_adaptive_tiling": True,
                            "tile_size": 512,
                            "tile_overlap": 0.3,
                        },
                    }
                }
            },
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        # Tiling is enabled ...
        assert result["data"]["tile_config"]["enable_tiler"] is True
        # ... and the DEIM augmentation scheduler callback has been excluded.
        assert GetiConfigConverter.get_callback_idx(result["callbacks"], AUG_SCHEDULER_CLASS_PATH) == -1

    def test_convert_keeps_deim_framework_when_tiling_disabled(self) -> None:
        """When tiling is disabled and DEIM is enabled, the DEIM scheduler callback is retained."""
        getitune_cfg = _make_getitune_config(
            callbacks=[
                copy.deepcopy(EARLY_STOPPING_CALLBACK),
                copy.deepcopy(CHECKPOINT_CALLBACK),
                copy.deepcopy(AUG_SCHEDULER_CALLBACK),
            ]
        )
        geti_cfg = _make_geti_config(
            model_manifest_id="object-detection-dfine-m",
            hyper_parameters={
                "dataset_preparation": {
                    "augmentation": {
                        "deim_framework": True,
                        "tiling": {"enable": False},
                    }
                }
            },
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["data"]["tile_config"]["enable_tiler"] is False
        assert GetiConfigConverter.get_callback_idx(result["callbacks"], AUG_SCHEDULER_CLASS_PATH) >= 0

    def test_convert_applies_input_size(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(hyper_parameters={"training": {"input_size_height": 640, "input_size_width": 640}})

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["data"]["input_size"] == (640, 640)

    def test_convert_applies_weight_decay(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(hyper_parameters={"training": {"weight_decay": 0.001}})

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["model"]["init_args"]["optimizer"]["init_args"]["weight_decay"] == 0.001

    def test_convert_applies_gradient_clip(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            hyper_parameters={"training": {"gradient_clip": {"enable": True, "max_grad_norm": 1.0}}}
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["engine"]["gradient_clip_val"] == 1.0

    def test_convert_disables_gradient_clip(self) -> None:
        getitune_cfg = _make_getitune_config()
        getitune_cfg["gradient_clip_val"] = 35.0  # pre-existing value
        geti_cfg = _make_geti_config(
            hyper_parameters={"training": {"gradient_clip": {"enable": False, "max_grad_norm": 35.0}}}
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["engine"]["gradient_clip_val"] is None

    def test_convert_applies_gradient_accumulation(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            hyper_parameters={"training": {"gradient_accumulation": {"enable": True, "batches": 4}}}
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["engine"]["accumulate_grad_batches"] == 4

    def test_convert_disables_gradient_accumulation(self) -> None:
        getitune_cfg = _make_getitune_config()
        getitune_cfg["engine"]["accumulate_grad_batches"] = 4
        geti_cfg = _make_geti_config(
            hyper_parameters={"training": {"gradient_accumulation": {"enable": False, "batches": 4}}}
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["engine"]["accumulate_grad_batches"] == 1

    def test_convert_applies_scheduler_reduce_lr_params(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            hyper_parameters={
                "training": {
                    "scheduler": {
                        "type": "reduce_lr_on_plateau",
                        "factor": 0.5,
                        "patience": 3,
                        "warmup": {"enable": False},
                    }
                }
            }
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        scheduler = result["model"]["init_args"]["scheduler"]
        main_sched = scheduler["init_args"]["main_scheduler_callable"]
        assert "ReduceLROnPlateau" in main_sched["class_path"]
        assert main_sched["init_args"]["factor"] == 0.5
        assert main_sched["init_args"]["patience"] == 3

    def test_convert_enables_warmup(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            hyper_parameters={
                "training": {
                    "scheduler": {
                        "type": "reduce_lr_on_plateau",
                        "warmup": {"enable": True, "epochs": 5},
                    }
                }
            }
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        scheduler = result["model"]["init_args"]["scheduler"]
        assert scheduler["init_args"]["num_warmup_steps"] == 5
        assert scheduler["init_args"]["warmup_interval"] == "epoch"

    def test_convert_disables_warmup(self) -> None:
        getitune_cfg = _make_getitune_config()
        getitune_cfg["model"]["init_args"]["scheduler"]["init_args"]["num_warmup_steps"] = 10
        geti_cfg = _make_geti_config(
            hyper_parameters={
                "training": {
                    "scheduler": {
                        "warmup": {"enable": False},
                    }
                }
            }
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        scheduler = result["model"]["init_args"]["scheduler"]
        assert scheduler["init_args"]["num_warmup_steps"] == 0

    def test_convert_classification_routes_sub_task_type(self) -> None:
        """Verify that classification models use the sub_task_type for recipe path."""
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-efficientnet-b0",
            sub_task_type="MULTI_CLASS_CLS",
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            GetiConfigConverter.convert(geti_cfg)

            # Verify AutoConfigurator was called with a path containing multi_class_cls
            call_args = MockAutoConfigurator.call_args
            model_path = call_args.kwargs.get("model") or call_args.args[0] if call_args.args else None
            if model_path is None:
                model_path = call_args[1].get("model")
            assert "multi_class_cls" in str(model_path)

    def test_convert_applies_tiling(self) -> None:
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            hyper_parameters={
                "dataset_preparation": {
                    "augmentation": {
                        "tiling": {
                            "enable": True,
                            "enable_adaptive_tiling": True,
                            "tile_size": 512,
                            "tile_overlap": 0.3,
                        }
                    }
                }
            }
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        assert result["data"]["tile_config"]["enable_tiler"] is True
        assert result["data"]["tile_config"]["enable_adaptive_tiling"] is True
        assert result["data"]["tile_config"]["tile_size"] == (512, 512)
        assert result["data"]["tile_config"]["overlap"] == 0.3


class TestHyperparametersUpdater:
    """Direct tests for HyperparametersUpdater methods."""

    def test_update_weight_decay(self) -> None:
        config = _make_getitune_config()
        HyperparametersUpdater._update_weight_decay(0.005, config)
        assert config["model"]["init_args"]["optimizer"]["init_args"]["weight_decay"] == 0.005

    def test_update_weight_decay_none(self) -> None:
        config = _make_getitune_config()
        original = config["model"]["init_args"]["optimizer"]["init_args"]["weight_decay"]
        HyperparametersUpdater._update_weight_decay(None, config)
        assert config["model"]["init_args"]["optimizer"]["init_args"]["weight_decay"] == original

    def test_update_gradient_clip_enable(self) -> None:
        config = _make_getitune_config()
        HyperparametersUpdater._update_gradient_clip({"enable": True, "max_grad_norm": 5.0}, config)
        assert config["engine"]["gradient_clip_val"] == 5.0

    def test_update_gradient_clip_disable(self) -> None:
        config = _make_getitune_config()
        config["gradient_clip_val"] = 35.0
        HyperparametersUpdater._update_gradient_clip({"enable": False}, config)
        assert config["engine"]["gradient_clip_val"] is None

    def test_update_gradient_accumulation_enable(self) -> None:
        config = _make_getitune_config()
        HyperparametersUpdater._update_gradient_accumulation({"enable": True, "batches": 8}, config)
        assert config["engine"]["accumulate_grad_batches"] == 8

    def test_update_gradient_accumulation_disable(self) -> None:
        config = _make_getitune_config()
        config["engine"]["accumulate_grad_batches"] = 4
        HyperparametersUpdater._update_gradient_accumulation({"enable": False}, config)
        assert config["engine"]["accumulate_grad_batches"] == 1

    def test_update_gradient_accumulation_single_batch_noop(self) -> None:
        """When enable=True but batches=1, no key should be set."""
        config = _make_getitune_config()
        HyperparametersUpdater._update_gradient_accumulation({"enable": True, "batches": 1}, config)
        assert "accumulate_grad_batches" not in config.get("engine", {})

    def test_update_scheduler_factor_patience(self) -> None:
        config = _make_getitune_config()
        HyperparametersUpdater._update_scheduler(
            {"type": "reduce_lr_on_plateau", "factor": 0.2, "patience": 7, "warmup": {"enable": False}},
            config,
        )
        main = config["model"]["init_args"]["scheduler"]["init_args"]["main_scheduler_callable"]
        assert main["init_args"]["factor"] == 0.2
        assert main["init_args"]["patience"] == 7

    def test_update_scheduler_warmup_enable(self) -> None:
        config = _make_getitune_config()
        HyperparametersUpdater._update_scheduler(
            {"warmup": {"enable": True, "epochs": 3}},
            config,
        )
        sched = config["model"]["init_args"]["scheduler"]["init_args"]
        assert sched["num_warmup_steps"] == 3
        assert sched["warmup_interval"] == "epoch"

    def test_update_scheduler_warmup_disable(self) -> None:
        config = _make_getitune_config()
        config["model"]["init_args"]["scheduler"]["init_args"]["num_warmup_steps"] = 10
        HyperparametersUpdater._update_scheduler(
            {"warmup": {"enable": False}},
            config,
        )
        sched = config["model"]["init_args"]["scheduler"]["init_args"]
        assert sched["num_warmup_steps"] == 0


class TestTransformsUpdater:
    """Tests for _TransformsUpdater."""

    def test_update_adds_new_augmentation(self) -> None:
        config = _make_getitune_config()
        TransformsUpdater.update(
            {"random_vertical_flip": {"enable": True, "probability": 0.3}},
            config,
        )
        gpu_augs = config["data"]["train_subset"]["augmentations_gpu"]
        vflip = [a for a in gpu_augs if "VerticalFlip" in a["class_path"]]
        assert len(vflip) == 1
        assert vflip[0]["init_args"]["p"] == 0.3

    def test_update_removes_augmentation(self) -> None:
        config = _make_getitune_config()
        TransformsUpdater.update(
            {"random_horizontal_flip": {"enable": False, "probability": 0.5}},
            config,
        )
        gpu_augs = config["data"]["train_subset"]["augmentations_gpu"]
        hflip = [a for a in gpu_augs if "HorizontalFlip" in a["class_path"]]
        assert len(hflip) == 0

    def test_update_modifies_existing_augmentation(self) -> None:
        config = _make_getitune_config()
        TransformsUpdater.update(
            {"random_horizontal_flip": {"enable": True, "probability": 0.9}},
            config,
        )
        gpu_augs = config["data"]["train_subset"]["augmentations_gpu"]
        hflip = [a for a in gpu_augs if "HorizontalFlip" in a["class_path"]]
        assert len(hflip) == 1
        assert hflip[0]["init_args"]["p"] == 0.9

    def test_update_disable_random_resize_crop_replaces_with_resize(self) -> None:
        """Disabling random_resize_crop should replace it with a plain Resize."""
        config = _make_getitune_config()
        # Add a RandomResizedCrop first
        config["data"]["train_subset"]["augmentations_cpu"].insert(
            0,
            {
                "class_path": "torchvision.transforms.v2.RandomResizedCrop",
                "init_args": {"size": "$(input_size)"},
            },
        )
        TransformsUpdater.update(
            {"random_resize_crop": {"enable": False}},
            config,
        )
        cpu_augs = config["data"]["train_subset"]["augmentations_cpu"]
        # The first item should now be a plain Resize
        assert cpu_augs[0]["class_path"] == "getitune.data.augmentation.transforms.Resize"

    def test_tiling_update(self) -> None:
        config = _make_getitune_config()
        TransformsUpdater.update_tiling(
            {
                "enable": True,
                "enable_adaptive_tiling": True,
                "tile_size": 256,
                "tile_overlap": 0.5,
            },
            config,
        )
        tc = config["data"]["tile_config"]
        assert tc["enable_tiler"] is True
        assert tc["enable_adaptive_tiling"] is True
        assert tc["tile_size"] == (256, 256)
        assert tc["overlap"] == 0.5

    def test_tiling_update_disabled(self) -> None:
        config = _make_getitune_config()
        TransformsUpdater.update_tiling(
            {
                "enable": False,
                "enable_adaptive_tiling": False,
                "tile_size": 256,
                "tile_overlap": 0.5,
            },
            config,
        )
        tc = config["data"]["tile_config"]
        assert tc["enable_tiler"] is False

    def test_mosaic_skipped_when_tiling_enabled(self) -> None:
        """Mosaic must never be added to a tiling pipeline (it is incompatible)."""
        config = _make_getitune_config()
        config["data"]["tile_config"]["enable_tiler"] = True
        TransformsUpdater.update(
            {"mosaic": {"enable": True, "probability": 1.0}},
            config,
        )
        cpu_augs = config["data"]["train_subset"]["augmentations_cpu"]
        assert not any("CachedMosaic" in a["class_path"] for a in cpu_augs)

    def test_mixup_skipped_when_tiling_enabled(self) -> None:
        """MixUp must never be added to a tiling pipeline (it is incompatible)."""
        config = _make_getitune_config()
        config["data"]["tile_config"]["enable_tiler"] = True
        TransformsUpdater.update(
            {"mixup": {"enable": True, "probability": 0.5, "alpha": 1.5}},
            config,
        )
        cpu_augs = config["data"]["train_subset"]["augmentations_cpu"]
        assert not any("CachedMixUp" in a["class_path"] for a in cpu_augs)

    def test_mosaic_added_when_tiling_disabled(self) -> None:
        """Sanity check: mosaic is still added when tiling is off."""
        config = _make_getitune_config()
        TransformsUpdater.update(
            {"mosaic": {"enable": True, "probability": 1.0}},
            config,
        )
        cpu_augs = config["data"]["train_subset"]["augmentations_cpu"]
        assert any("CachedMosaic" in a["class_path"] for a in cpu_augs)

    def test_mixup_added_when_tiling_disabled(self) -> None:
        """Sanity check: mixup is still added when tiling is off."""
        config = _make_getitune_config()
        TransformsUpdater.update(
            {"mixup": {"enable": True, "probability": 0.5, "alpha": 1.5}},
            config,
        )
        cpu_augs = config["data"]["train_subset"]["augmentations_cpu"]
        assert any("CachedMixUp" in a["class_path"] for a in cpu_augs)

    def test_gaussian_noise_sigma_renamed_to_std(self) -> None:
        config = _make_getitune_config()
        TransformsUpdater.update(
            {
                "gaussian_noise": {
                    "enable": True,
                    "mean": 0.0,
                    "sigma": 0.1,
                    "probability": 0.5,
                }
            },
            config,
        )
        gpu_augs = config["data"]["train_subset"]["augmentations_gpu"]
        noise = [a for a in gpu_augs if "GaussianNoise" in a["class_path"]]
        assert len(noise) == 1
        assert "std" in noise[0]["init_args"]
        assert noise[0]["init_args"]["std"] == 0.1

    def test_random_erasing_uses_mask_safe_class(self) -> None:
        """random_erasing must map to MaskSafeRandomErasing, not raw kornia class."""
        registry_entry = TransformsUpdater.AUGMENTATION_REGISTRY["random_erasing"]
        assert "getitune.data.augmentation.transforms.MaskSafeRandomErasing" in registry_entry["class_paths"]
        assert registry_entry["stage"] == "gpu"


class TestGetCallbackIdx:
    def test_found(self) -> None:
        callbacks = [
            {"class_path": "a.B"},
            {"class_path": EARLY_STOPPING_CLASS_PATH},
        ]
        assert GetiConfigConverter.get_callback_idx(callbacks, EARLY_STOPPING_CLASS_PATH) == 1

    def test_not_found(self) -> None:
        callbacks = [{"class_path": "a.B"}]
        assert GetiConfigConverter.get_callback_idx(callbacks, "missing.Class") == -1


def _make_deim_getitune_config() -> dict:
    """A getitune config that includes the DEIM AugmentationSchedulerCallback."""
    config = _make_getitune_config()
    config["callbacks"].append({"class_path": AUG_SCHEDULER_CLASS_PATH, "init_args": {}})
    return config


class TestUpdateParamsDeimTiling:
    """Tests for the DEIM framework / tiling interaction in _update_params.

    Tiling is incompatible with the DEIM adaptive augmentation scheduling
    framework (mosaic/mixup based). When tiling is enabled the converter must:
      * force DEIM off (remove the AugmentationSchedulerCallback),
      * apply the tiling-compatible user augmentations,
      * never add tiling-incompatible augmentations (mosaic/mixup).
    """

    def _has_scheduler(self, config: dict) -> bool:
        return GetiConfigConverter.get_callback_idx(config["callbacks"], AUG_SCHEDULER_CLASS_PATH) > -1

    def test_deim_enabled_no_tiling_keeps_scheduler_and_ignores_augs(self) -> None:
        config = _make_deim_getitune_config()
        param_dict = {
            "dataset_preparation": {
                "augmentation": {
                    "deim_framework": True,
                    "tiling": {"enable": False, "enable_adaptive_tiling": False, "tile_size": 256, "tile_overlap": 0.5},
                    "random_vertical_flip": {"enable": True, "probability": 0.3},
                }
            }
        }
        GetiConfigConverter._update_params(config, param_dict)

        # Scheduler kept (DEIM owns the pipeline) and user aug override ignored.
        assert self._has_scheduler(config)
        gpu_augs = config["data"]["train_subset"]["augmentations_gpu"]
        assert not any("VerticalFlip" in a["class_path"] for a in gpu_augs)

    def test_deim_enabled_with_tiling_disables_scheduler(self) -> None:
        config = _make_deim_getitune_config()
        param_dict = {
            "dataset_preparation": {
                "augmentation": {
                    "deim_framework": True,
                    "tiling": {"enable": True, "enable_adaptive_tiling": True, "tile_size": 256, "tile_overlap": 0.5},
                    "mosaic": {"enable": True, "probability": 1.0},
                    "random_vertical_flip": {"enable": True, "probability": 0.3},
                }
            }
        }
        GetiConfigConverter._update_params(config, param_dict)

        # Tiling forces DEIM off -> scheduler removed.
        assert not self._has_scheduler(config)
        # Tiling enabled in the data config.
        assert config["data"]["tile_config"]["enable_tiler"] is True
        # Mosaic (incompatible) must not be added.
        cpu_augs = config["data"]["train_subset"]["augmentations_cpu"]
        assert not any("CachedMosaic" in a["class_path"] for a in cpu_augs)
        # Compatible augmentations are applied.
        gpu_augs = config["data"]["train_subset"]["augmentations_gpu"]
        assert any("VerticalFlip" in a["class_path"] for a in gpu_augs)

    def test_deim_disabled_with_tiling_disables_scheduler_and_skips_mosaic(self) -> None:
        config = _make_deim_getitune_config()
        param_dict = {
            "dataset_preparation": {
                "augmentation": {
                    "deim_framework": False,
                    "tiling": {"enable": True, "enable_adaptive_tiling": True, "tile_size": 256, "tile_overlap": 0.5},
                    "mixup": {"enable": True, "probability": 0.5, "alpha": 1.5},
                }
            }
        }
        GetiConfigConverter._update_params(config, param_dict)

        assert not self._has_scheduler(config)
        assert config["data"]["tile_config"]["enable_tiler"] is True
        cpu_augs = config["data"]["train_subset"]["augmentations_cpu"]
        assert not any("CachedMixUp" in a["class_path"] for a in cpu_augs)


class TestFullConfigRoundTrip:
    """Test with a config dict resembling a real Geti model_dump output."""

    def test_detection_full_config(self) -> None:
        """Simulate a full detection training configuration from Geti."""
        getitune_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            model_manifest_id="object-detection-atss-mobilenet-v2",
            hyper_parameters={
                "dataset_preparation": {
                    "augmentation": {
                        "random_horizontal_flip": {"enable": True, "probability": 0.7},
                        "random_vertical_flip": {"enable": False, "probability": 0.5},
                        "random_affine": {
                            "enable": True,
                            "max_rotate_degree": 15.0,
                            "max_translate_ratio": 0.2,
                            "scaling_ratio_range": [0.5, 1.5],
                            "max_shear_degree": 5.0,
                            "probability": 0.5,
                        },
                    }
                },
                "training": {
                    "learning_rate": 0.002,
                    "batch_size": 16,
                    "max_epochs": 100,
                    "weight_decay": 0.0005,
                    "early_stopping": {"enable": True, "patience": 15},
                    "input_size_height": 640,
                    "input_size_width": 640,
                    "scheduler": {
                        "type": "reduce_lr_on_plateau",
                        "factor": 0.5,
                        "patience": 5,
                        "warmup": {"enable": True, "epochs": 3},
                    },
                    "gradient_clip": {"enable": True, "max_grad_norm": 10.0},
                    "gradient_accumulation": {"enable": True, "batches": 2},
                },
            },
            task_level_parameters={
                "dataset_preparation": {
                    "intensity_mapping": {
                        "mode": "window",
                        "window_center": 2048.0,
                        "window_width": 4096.0,
                    }
                }
            },
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        # Verify all hyperparameters were applied
        assert result["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.002
        assert result["model"]["init_args"]["optimizer"]["init_args"]["weight_decay"] == 0.0005
        assert result["data"]["train_subset"]["batch_size"] == 16
        assert result["data"]["val_subset"]["batch_size"] == 16
        assert result["max_epochs"] == 100
        assert result["data"]["input_size"] == (640, 640)
        assert result["engine"]["gradient_clip_val"] == 10.0
        assert result["engine"]["accumulate_grad_batches"] == 2

        # Scheduler
        sched = result["model"]["init_args"]["scheduler"]["init_args"]
        assert sched["num_warmup_steps"] == 3
        assert sched["warmup_interval"] == "epoch"
        main = sched["main_scheduler_callable"]
        assert main["init_args"]["factor"] == 0.5
        assert main["init_args"]["patience"] == 5

        # Early stopping
        es_idx = GetiConfigConverter.get_callback_idx(result["callbacks"], EARLY_STOPPING_CLASS_PATH)
        assert result["callbacks"][es_idx]["init_args"]["patience"] == 15

        # Augmentations
        gpu_augs = result["data"]["train_subset"]["augmentations_gpu"]
        hflip = [a for a in gpu_augs if "HorizontalFlip" in a["class_path"]]
        assert hflip[0]["init_args"]["p"] == 0.7

        vflip = [a for a in gpu_augs if "VerticalFlip" in a["class_path"]]
        assert len(vflip) == 0  # disabled

        affine = [a for a in gpu_augs if "RandomAffine" in a["class_path"]]
        assert len(affine) == 1
        assert affine[0]["init_args"]["degrees"] == 15.0

        # Intensity mapping
        for subset in ("train_subset", "val_subset", "test_subset"):
            intensity = result["data"][subset]["intensity"]
            assert intensity["mode"] == "window"
            assert intensity["window_center"] == 2048.0
            assert intensity["window_width"] == 4096.0


class TestIntensityMappingUpdate:
    """Tests for intensity mapping parameter handling in GetiConfigConverter."""

    def test_convert_applies_intensity_mapping_scale_to_unit(self) -> None:
        """scale_to_unit mode should set max_value on all subsets."""
        otx_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            task_level_parameters={
                "dataset_preparation": {"intensity_mapping": {"mode": "scale_to_unit", "max_intensity_value": 65535.0}}
            }
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = otx_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        for subset in ("train_subset", "val_subset", "test_subset"):
            intensity = result["data"][subset]["intensity"]
            assert intensity["mode"] == "scale_to_unit"
            assert intensity["max_value"] == 65535.0

    def test_convert_applies_intensity_mapping_window(self) -> None:
        """window mode should set window_center and window_width."""
        otx_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            task_level_parameters={
                "dataset_preparation": {
                    "intensity_mapping": {"mode": "window", "window_center": 500.0, "window_width": 1000.0}
                }
            }
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = otx_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        intensity = result["data"]["train_subset"]["intensity"]
        assert intensity["mode"] == "window"
        assert intensity["window_center"] == 500.0
        assert intensity["window_width"] == 1000.0
        assert "max_value" not in intensity

    def test_convert_applies_intensity_mapping_range_scale(self) -> None:
        """range_scale mode should set scale_factor, min_value, and max_value."""
        otx_cfg = _make_getitune_config()
        geti_cfg = _make_geti_config(
            task_level_parameters={
                "dataset_preparation": {
                    "intensity_mapping": {
                        "mode": "range_scale",
                        "scale_factor": 0.4,
                        "clip_min_value": 10.0,
                        "clip_max_value": 300.0,
                    }
                }
            }
        )

        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = otx_cfg
            result = GetiConfigConverter.convert(geti_cfg)

        intensity = result["data"]["train_subset"]["intensity"]
        assert intensity["mode"] == "range_scale"
        assert intensity["scale_factor"] == 0.4
        assert intensity["min_value"] == 10.0
        assert intensity["max_value"] == 300.0


class TestTimmRecipeConversion:
    """Tests for GetiConfigConverter's timm backbone handling in the Lightning path."""

    def _convert(self, geti_cfg: dict, getitune_cfg: dict) -> dict:
        with patch("getitune.tools.auto_configurator.AutoConfigurator") as MockAutoConfigurator:
            MockAutoConfigurator.return_value.config = getitune_cfg
            return GetiConfigConverter.convert(geti_cfg)

    def test_injects_model_name_from_catalog(self) -> None:
        """The model_name DYNAMIC value that only GetiConfigConverter (not HyperparametersUpdater)
        is responsible for resolving."""
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-timm-resnet18.a1_in1k",
            sub_task_type="MULTI_CLASS_CLS",
            hyper_parameters={"training": {"learning_rate": 0.02, "weight_decay": 0.0002}},
        )
        result = self._convert(geti_cfg, _make_timm_getitune_config())

        assert result["model"]["init_args"]["model_name"] == "resnet18.a1_in1k"

    def test_timm_optimizer_init_args(self) -> None:
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-timm-resnet18.a1_in1k",
            sub_task_type="MULTI_CLASS_CLS",
            hyper_parameters={"training": {"learning_rate": 0.03, "weight_decay": 0.0002}},
        )

        result = self._convert(geti_cfg, _make_timm_getitune_config())

        optimizer = result["model"]["init_args"]["optimizer"]
        assert optimizer["class_path"].endswith("TimmOptimizer")
        assert optimizer["init_args"]["lr"] == 0.03
        assert optimizer["init_args"]["weight_decay"] == 0.0002

    def test_raises_when_dynamic_placeholder_unresolved(self) -> None:
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-timm-resnet18.a1_in1k",
            sub_task_type="MULTI_CLASS_CLS",
            hyper_parameters={},  # no learning_rate/weight_decay supplied
        )

        with pytest.raises(ValueError, match="DYNAMIC"):
            self._convert(geti_cfg, _make_timm_getitune_config())

    def test_syncs_data_input_size_and_normalization_from_preprocessing(self) -> None:
        """input_size and Normalize mean/std in data.*_subset must match the backbone's
        own pretrained_cfg, not the generic recipe's ImageNet defaults."""
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-timm-resnet18.a1_in1k",
            sub_task_type="MULTI_CLASS_CLS",
            hyper_parameters={"training": {"learning_rate": 0.02, "weight_decay": 0.0002}},
        )
        fake_preprocessing = {
            "input_size": (299, 299),
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
        }

        with patch(
            "app.execution.common.geti_config_converter.TimmManifestProvider.get_preprocessing",
            return_value=fake_preprocessing,
        ):
            result = self._convert(geti_cfg, _make_timm_getitune_config())

        assert result["data"]["input_size"] == [299, 299]
        for subset_key in ("train_subset", "val_subset", "test_subset"):
            normalize = next(
                aug for aug in result["data"][subset_key]["augmentations_gpu"] if "Normalize" in aug["class_path"]
            )
            assert normalize["init_args"]["mean"] == [0.5, 0.5, 0.5]
            assert normalize["init_args"]["std"] == [0.5, 0.5, 0.5]

    def test_user_input_size_override_wins_over_timm_default(self) -> None:
        """Explicit hyperparameter overrides (applied after backbone injection) must
        still take precedence over the backbone's native input_size."""
        geti_cfg = _make_geti_config(
            model_manifest_id="image-classification-timm-resnet18.a1_in1k",
            sub_task_type="MULTI_CLASS_CLS",
            hyper_parameters={
                "training": {
                    "learning_rate": 0.02,
                    "weight_decay": 0.0002,
                    "input_size_height": 384,
                    "input_size_width": 384,
                }
            },
        )

        result = self._convert(geti_cfg, _make_timm_getitune_config())

        assert result["data"]["input_size"] == (384, 384)
