# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Converter for v1 config."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar
from warnings import warn

import yaml
from loguru import logger

from app.execution.common.recipe_resolver import RecipeResolver
from app.supported_models.timm import TimmManifestProvider, id_to_model_name


class TransformsUpdater:
    """Handles augmentation updates for the new CPU/GPU augmentation pipeline.

    Maps Geti augmentation names to getitune/kornia/torchvision class paths and the
    pipeline stage (cpu or gpu). Parameters come directly from the Geti model template;
    only a few Geti param names need renaming to match kornia's API.

    Example Geti model template augmentation section::

        random_affine:
            enable: false
            max_rotate_degree: 10.0
            max_translate_ratio: 0.1
            scaling_ratio_range: [0.5, 1.5]
            max_shear_degree: 2.0
        color_jitter:
            enable: false
            brightness: [0.875, 1.125]
            probability: 0.5
    """

    # Geti name -> (class_path, stage)
    # class_paths is a list to match multiple possible implementations in configs
    # (e.g. kornia GPU variant and torchvision CPU variant).
    #
    # For augmentations with both kornia and torchvision variants the first
    # entry is the kornia (GPU) default; the second is the torchvision (CPU)
    # fallback used by Ultralytics recipes which run all augmentations on CPU.
    #
    # ``param_rename`` maps *class_path* -> {manifest_name: class_arg_name}
    # for class paths that need parameter renaming beyond the global
    # ``PARAM_RENAME`` table.
    AUGMENTATION_REGISTRY: ClassVar[dict[str, dict]] = {
        "random_resize_crop": {
            "class_paths": [
                "torchvision.transforms.v2.RandomResizedCrop",
            ],
            "stage": "cpu",
        },
        "random_affine": {
            "class_paths": ["kornia.augmentation.RandomAffine"],
            "stage": "gpu",
        },
        "random_horizontal_flip": {
            "class_paths": [
                "kornia.augmentation.RandomHorizontalFlip",
                "torchvision.transforms.v2.RandomHorizontalFlip",
            ],
            "stage": "gpu",
        },
        "random_vertical_flip": {
            "class_paths": [
                "kornia.augmentation.RandomVerticalFlip",
                "torchvision.transforms.v2.RandomVerticalFlip",
            ],
            "stage": "gpu",
        },
        "gaussian_blur": {
            "class_paths": ["kornia.augmentation.RandomGaussianBlur"],
            "stage": "gpu",
        },
        "gaussian_noise": {
            "class_paths": ["kornia.augmentation.RandomGaussianNoise"],
            "stage": "gpu",
            "param_rename": {
                "kornia.augmentation.RandomGaussianNoise": {"sigma": "std"},
            },
        },
        "color_jitter": {
            "class_paths": [
                "kornia.augmentation.ColorJiggle",
                "torchvision.transforms.v2.ColorJitter",
            ],
            "stage": "gpu",
            "drop_params": {
                "torchvision.transforms.v2.ColorJitter": {"p"},
            },
        },
        "iou_random_crop": {
            "class_paths": ["getitune.data.augmentation.transforms.RandomIoUCrop"],
            "stage": "cpu",
        },
        "random_zoom_out": {
            "class_paths": ["torchvision.transforms.v2.RandomZoomOut"],
            "stage": "cpu",
        },
        "mixup": {
            "class_paths": ["getitune.data.augmentation.transforms.CachedMixUp"],
            "stage": "cpu",
        },
        "mosaic": {
            "class_paths": ["getitune.data.augmentation.transforms.CachedMosaic"],
            "stage": "cpu",
        },
        "random_erasing": {
            "class_paths": [
                "getitune.data.augmentation.transforms.MaskSafeRandomErasing",
                "kornia.augmentation.RandomErasing",
                "torchvision.transforms.v2.RandomErasing",
            ],
            "stage": "gpu",
        },
        "random_grayscale": {
            "class_paths": [
                "kornia.augmentation.RandomGrayscale",
                "torchvision.transforms.v2.RandomGrayscale",
            ],
            "stage": "gpu",
        },
        "random_sharpness": {
            "class_paths": [
                "kornia.augmentation.RandomSharpness",
                "torchvision.transforms.v2.RandomAdjustSharpness",
            ],
            "stage": "gpu",
            "param_rename": {
                "torchvision.transforms.v2.RandomAdjustSharpness": {"sharpness": "sharpness_factor"},
            },
        },
    }

    # Geti param name -> kornia/torchvision param name
    PARAM_RENAME: ClassVar[dict[str, str]] = {
        "probability": "p",
        "max_rotate_degree": "degrees",
        "scaling_ratio_range": "scale",
        "crop_ratio_range": "scale",
        "aspect_ratio_range": "ratio",
        "max_translate_ratio": "translate",
        "max_shear_degree": "shear",
    }

    # Augmentations that cannot be combined with the tiling pipeline.  Mosaic and
    # MixUp stitch/blend multiple full images together, which is fundamentally
    # incompatible with splitting an image into tiles, so they must never be added
    # to a tiling recipe.
    TILING_INCOMPATIBLE_AUGMENTATIONS: ClassVar[set[str]] = {"mosaic", "mixup"}

    @classmethod
    def update(cls, augmentation_params: dict, config: dict) -> None:  # noqa: C901, PLR0912
        """Update augmentations in the config based on Geti model template.

        For each augmentation in augmentation_params:
        - If enable=True and aug exists in config -> update its parameters
        - If enable=True and aug does NOT exist -> add it with template params
        - If enable=False and aug exists -> remove it
        - If enable=False and aug does NOT exist -> no-op

        Special case: disabling random_resize_crop replaces it with plain Resize.

        Backend routing:
        - Ultralytics recipes run all augmentations on CPU.  When a new
          augmentation is added the torchvision / getitune CPU variant is
          preferred and placed in ``augmentations_cpu``.
        - Lightning recipes use the registry default (typically kornia on GPU).

        Args:
            augmentation_params: Dict mapping Geti aug names to their parameter dicts.
            config: The full getitune config dictionary.
        """
        if not augmentation_params:
            return

        tiling = config["data"].get("tile_config", {}).get("enable_tiler", False)
        train_subset = config["data"]["train_subset"]
        is_ultralytics = config.get("backend") == "ultralytics"

        for aug_name, aug_value in augmentation_params.items():
            if tiling and aug_name in cls.TILING_INCOMPATIBLE_AUGMENTATIONS and aug_value.get("enable", True):
                # The user explicitly enabled an augmentation that cannot coexist with tiling. Warn so it is clear
                # we are overriding their request.
                logger.warning(
                    "Augmentation '{}' is incompatible with the Tiling pipeline and will be overridden (disabled)",
                    aug_name,
                )
                continue
            if aug_name not in cls.AUGMENTATION_REGISTRY:
                if tiling:
                    logger.info("Augmentation '{}' is not applicable in Tiling pipeline", aug_name)
                    continue
                msg = f"Unknown augmentation: '{aug_name}'. Available: {list(cls.AUGMENTATION_REGISTRY.keys())}"
                raise ValueError(msg)

            registry_entry = cls.AUGMENTATION_REGISTRY[aug_name]
            params = dict(aug_value)
            enable = params.pop("enable", True)

            aug_list, existing_idx = cls._locate_augmentation(
                train_subset, registry_entry["class_paths"], registry_entry["stage"]
            )

            if enable:
                if existing_idx is not None:
                    # Update existing augmentation parameters
                    aug_config = aug_list[existing_idx]
                    class_path = aug_config.get("class_path", "")
                    per_aug_rename = cls._get_param_rename(registry_entry, class_path)
                    init_args = cls._remap_params(params, per_aug_rename, aug_name=aug_name)
                    cls._drop_unsupported_params(init_args, registry_entry, class_path)
                    aug_config.setdefault("init_args", {}).update(init_args)
                    aug_config.pop("enable", None)
                else:
                    # Add new augmentation
                    class_path, target_stage = cls._choose_variant(registry_entry, is_ultralytics)
                    per_aug_rename = cls._get_param_rename(registry_entry, class_path)
                    init_args = cls._remap_params(params, per_aug_rename, aug_name=aug_name)
                    cls._drop_unsupported_params(init_args, registry_entry, class_path)

                    target_list = train_subset.setdefault(f"augmentations_{target_stage}", [])
                    new_aug: dict[str, Any] = {"class_path": class_path}
                    if init_args:
                        new_aug["init_args"] = init_args
                    target_list.insert(cls._get_insert_position(target_list, target_stage), new_aug)
            elif existing_idx is not None:
                if aug_name == "random_resize_crop":
                    aug_list[existing_idx] = {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {"size": "$(input_size)"},
                    }
                elif aug_name == "mosaic":
                    # CachedMosaic includes a built-in letterbox resize; replace
                    # with a plain Resize so the pipeline still produces
                    # correctly sized images when mosaic is disabled.
                    aug_list[existing_idx] = {
                        "class_path": "getitune.data.augmentation.transforms.Resize",
                        "init_args": {
                            "size": "$(input_size)",
                            "keep_aspect_ratio": True,
                            "center_padding": True,
                        },
                    }
                else:
                    aug_list.pop(existing_idx)

    @classmethod
    def _locate_augmentation(
        cls,
        train_subset: dict,
        class_paths: list[str],
        primary_stage: str,
    ) -> tuple[list[dict], int] | tuple[list[dict], None]:
        """Find an augmentation in either stage list, preferring the primary stage.

        Searches the primary stage list first (e.g. ``augmentations_gpu`` for
        kornia-default augmentations), then falls back to the alternate stage
        (e.g. ``augmentations_cpu`` where Ultralytics recipes place torchvision
        variants of the same augmentation).

        Returns:
            Tuple of ``(aug_list, index)`` if found.
            If not found, returns ``(primary_list, None)`` so callers always
            have a valid list reference for insertions.
        """
        alt_stage = "cpu" if primary_stage == "gpu" else "gpu"
        for stage in (primary_stage, alt_stage):
            key = f"augmentations_{stage}"
            if key in train_subset:
                idx = cls._find_augmentation(train_subset[key], class_paths)
                if idx is not None:
                    return train_subset[key], idx
        # Ensure primary list exists for potential insertion
        primary_list = train_subset.setdefault(f"augmentations_{primary_stage}", [])
        return primary_list, None

    @classmethod
    def _choose_variant(cls, registry_entry: dict, is_ultralytics: bool) -> tuple[str, str]:
        """Choose class_path and target stage for a new augmentation.

        For Ultralytics backends, prefers torchvision CPU variants over
        getitune custom variants (which may include kornia GPU subclasses).
        For other backends, uses the registry default (typically kornia GPU).

        Returns:
            Tuple of ``(class_path, stage)``.
        """
        if is_ultralytics:
            for cp in registry_entry["class_paths"]:
                if cp.startswith("torchvision."):
                    return cp, "cpu"
            for cp in registry_entry["class_paths"]:
                if cp.startswith("getitune."):
                    return cp, "cpu"
        return registry_entry["class_paths"][0], registry_entry["stage"]

    @classmethod
    def _get_param_rename(cls, registry_entry: dict, class_path: str) -> dict[str, str] | None:
        """Get per-variant parameter rename map for the chosen class path.

        The ``param_rename`` field maps *specific class paths* to their rename
        dictionaries.  Only class paths that need renaming beyond the global
        ``PARAM_RENAME`` table are listed.
        """
        param_rename = registry_entry.get("param_rename")
        if not param_rename:
            return None
        return param_rename.get(class_path)

    @classmethod
    def _drop_unsupported_params(cls, init_args: dict, registry_entry: dict, class_path: str) -> None:
        """Remove parameters that the target class does not accept.

        Some torchvision transforms lack parameters that their kornia
        counterparts support (e.g. ``ColorJitter`` has no ``p``).  The
        ``drop_params`` registry field lists parameter names to strip
        per class path.
        """
        drop_params = registry_entry.get("drop_params")
        if not drop_params:
            return
        to_drop = drop_params.get(class_path)
        if not to_drop:
            return
        for key in to_drop:
            init_args.pop(key, None)

    @classmethod
    def _remap_params(
        cls,
        params: dict,
        per_aug_rename: dict[str, str] | None = None,
        *,
        aug_name: str = "",
    ) -> dict:
        """Rename Geti parameter names to kornia/torchvision names and adjust values.

        1. Rename keys via PARAM_RENAME (probability->p, max_translate_ratio->translate, etc.)
           plus any per-augmentation overrides supplied via per_aug_rename.
        2. Adjust values where kornia expects a different format than a single scalar.
           These adjustments are scoped to the augmentation that needs them so that
           parameters with the same name on other augmentations (e.g. ``translate``
           on CachedMosaic) are not accidentally transformed.

        Args:
            params: Raw Geti parameter dict for the augmentation.
            per_aug_rename: Optional extra rename map applied after PARAM_RENAME.  Use this
                when a kornia class uses a different argument name than the global default
                (e.g. RandomGaussianNoise uses ``std`` while the manifest stores ``sigma``).
            aug_name: Geti augmentation name (used to scope value adjustments).
        """
        # Step 1: rename keys (global renames + per-augmentation overrides)
        rename_map = {**cls.PARAM_RENAME, **(per_aug_rename or {})}
        init_args: dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            init_args[rename_map.get(key, key)] = value

        # Step 2: per-augmentation value adjustments
        if aug_name == "random_affine":
            if "translate" in init_args and not isinstance(init_args["translate"], list):
                v = init_args["translate"]
                init_args["translate"] = [v, v]
            if "shear" in init_args and not isinstance(init_args["shear"], list):
                v = init_args["shear"]
                init_args["shear"] = [-v, v]
        if aug_name == "gaussian_blur" and "kernel_size" in init_args and isinstance(init_args["kernel_size"], int):
            v = init_args["kernel_size"]
            init_args["kernel_size"] = [v, v]

        return init_args

    @staticmethod
    def _find_augmentation(aug_list: list[dict], class_paths: list[str]) -> int | None:
        """Find the index of an augmentation in the list by its class path."""
        for idx, aug_config in enumerate(aug_list):
            if aug_config.get("class_path") in class_paths:
                return idx
        return None

    @staticmethod
    def _get_insert_position(aug_list: list[dict], stage: str) -> int:
        """Determine where to insert a new augmentation.

        GPU: insert before Normalize (should always be last).
        CPU: insert before Resize or at the end.
        """
        if stage == "gpu":
            for idx, aug in enumerate(aug_list):
                if "Normalize" in aug.get("class_path", ""):
                    return idx
        elif stage == "cpu":
            for idx, aug in enumerate(aug_list):
                class_path = aug.get("class_path", "")
                if "Resize" in class_path and "RandomResizedCrop" not in class_path:
                    return idx
        return len(aug_list)

    @staticmethod
    def update_tiling(tiling_dict: dict | None, config: dict) -> None:
        """Update tiling parameters in the config.

        Args:
            tiling_dict: Dict with keys: enable, enable_adaptive_tiling, tile_size, tile_overlap.
            config: The full getitune config dictionary.
        """
        if tiling_dict is None:
            logger.info("Tiling parameters are not provided, skipping update.")
            return

        config["data"]["tile_config"]["enable_tiler"] = tiling_dict["enable"]
        if tiling_dict["enable"]:
            config["data"]["tile_config"]["enable_adaptive_tiling"] = tiling_dict["enable_adaptive_tiling"]
            config["data"]["tile_config"]["tile_size"] = (
                tiling_dict["tile_size"],
                tiling_dict["tile_size"],
            )
            config["data"]["tile_config"]["overlap"] = tiling_dict["tile_overlap"]


class HyperparametersUpdater:
    """Handles training hyperparameter updates (learning rate, batch size, etc.)."""

    @staticmethod
    def update(hyperparameters: dict, config: dict) -> None:  # noqa: C901
        """Update hyperparameters in the config.

        Supported keys:
        - learning_rate: float
        - batch_size: int
        - max_epochs: int (alias for num_iters)
        - early_stopping: dict with keys {enable, patience}
        - input_size: tuple (height, width)

        Args:
            hyperparameters: Dict of hyperparameter updates.
            config: The full getitune config dictionary.
        """
        for key, value in hyperparameters.items():
            if key == "learning_rate":
                HyperparametersUpdater._update_learning_rate(value, config)
            elif key == "batch_size":
                HyperparametersUpdater._update_batch_size(value, config)
            elif key == "max_epochs":
                HyperparametersUpdater._update_max_epochs(value, config)
            elif key == "early_stopping":
                HyperparametersUpdater._update_early_stopping(value, config)
            elif key == "input_size":
                HyperparametersUpdater._update_input_size(value, config)
            elif key == "weight_decay":
                HyperparametersUpdater._update_weight_decay(value, config)
            elif key == "scheduler":
                HyperparametersUpdater._update_scheduler(value, config)
            elif key == "gradient_accumulation":
                HyperparametersUpdater._update_gradient_accumulation(value, config)
            elif key == "gradient_clip":
                HyperparametersUpdater._update_gradient_clip(value, config)
            else:
                logger.warning("Unknown hyperparameter '%s' - skipping update", key)

    @staticmethod
    def _update_learning_rate(param_value: float | None, config: dict) -> None:
        """Update learning rate in the optimizer config."""
        if param_value is None:
            logger.info("Learning rate is not provided, skipping update.")
            return
        optimizer = config["model"]["init_args"]["optimizer"]
        if isinstance(optimizer, dict) and "init_args" in optimizer:
            optimizer["init_args"]["lr"] = param_value
        else:
            warn("Warning: learning_rate is not updated", stacklevel=1)

    @staticmethod
    def _update_batch_size(param_value: int | None, config: dict) -> None:
        """Update batch size for train and val subsets."""
        if param_value is None:
            logger.info("Batch size is not provided, skipping update.")
            return
        config["data"]["train_subset"]["batch_size"] = param_value
        config["data"]["val_subset"]["batch_size"] = param_value

    @staticmethod
    def _update_max_epochs(param_value: int | None, config: dict) -> None:
        """Update max_epochs in the config."""
        if param_value is None:
            logger.info("Max epochs is not provided, skipping update.")
            return
        config["max_epochs"] = param_value

    @staticmethod
    def _update_early_stopping(early_stopping_cfg: dict | None, config: dict) -> None:
        """Update early stopping parameters in the config."""
        if early_stopping_cfg is None:
            logger.info("Early stopping parameters are not provided, skipping update.")
            return

        enable = early_stopping_cfg["enable"]
        patience = early_stopping_cfg.get("patience")

        idx = GetiConfigConverter.get_callback_idx(
            config["callbacks"],
            "getitune.backend.lightning.callbacks.adaptive_early_stopping.EarlyStoppingWithWarmup",
        )
        if not enable and idx > -1:
            config["callbacks"].pop(idx)
            return

        if patience is not None:
            config["callbacks"][idx]["init_args"]["patience"] = patience

    @staticmethod
    def update_tiling(tiling_dict: dict | None, config: dict) -> None:
        """Update tiling parameters in the config."""
        if tiling_dict is None:
            logger.info("Tiling parameters are not provided, skipping update.")
            return

        config["data"]["tile_config"]["enable_tiler"] = tiling_dict["enable"]
        if tiling_dict["enable"]:
            config["data"]["tile_config"]["enable_adaptive_tiling"] = tiling_dict["enable_adaptive_tiling"]
            config["data"]["tile_config"]["tile_size"] = (tiling_dict["tile_size"], tiling_dict["tile_size"])
            config["data"]["tile_config"]["overlap"] = tiling_dict["tile_overlap"]

    @staticmethod
    def _update_input_size(size_value: tuple[int, int] | None, config: dict) -> None:
        """Update input size in the config.

        Args:
            size_value: Tuple of (height, width) or None.
            config: The full getitune config dictionary.
        """
        if size_value is None or any(v is None for v in size_value):
            logger.info("Input size is not provided, skipping update.")
            return
        config["data"]["input_size"] = size_value

    @staticmethod
    def _update_weight_decay(param_value: float | None, config: dict) -> None:
        """Update weight_decay in the optimizer config."""
        if param_value is None:
            logger.info("Weight decay is not provided, skipping update.")
            return
        optimizer = config["model"]["init_args"]["optimizer"]
        if isinstance(optimizer, dict) and "init_args" in optimizer:
            optimizer["init_args"]["weight_decay"] = param_value
        else:
            warn("Warning: weight_decay is not updated", stacklevel=1)

    @staticmethod
    def _update_scheduler(scheduler_cfg: dict | None, config: dict) -> None:
        """Update scheduler parameters in the config."""
        if scheduler_cfg is None:
            logger.info("Scheduler parameters are not provided, skipping update.")
            return

        scheduler = config["model"]["init_args"].get("scheduler")
        if not isinstance(scheduler, dict) or "init_args" not in scheduler:
            warn("Warning: scheduler config not found in recipe, skipping update.", stacklevel=1)
            return

        scheduler_init_args = scheduler["init_args"]

        # Update warmup
        warmup_cfg = scheduler_cfg.get("warmup")
        if warmup_cfg is not None:
            enable_warmup = warmup_cfg.get("enable", False)
            warmup_epochs = warmup_cfg.get("epochs", 0)
            if enable_warmup and warmup_epochs > 0:
                scheduler_init_args["num_warmup_steps"] = warmup_epochs
                scheduler_init_args["warmup_interval"] = "epoch"
            else:
                scheduler_init_args["num_warmup_steps"] = 0

        # Update the main scheduler parameters
        main_scheduler = scheduler_init_args.get("main_scheduler_callable")
        if not isinstance(main_scheduler, dict) or "init_args" not in main_scheduler:
            return

        scheduler_cfg.get("type")
        main_init_args = main_scheduler["init_args"]

        # Update type-specific params regardless of whether type was changed
        factor = scheduler_cfg.get("factor")
        patience = scheduler_cfg.get("patience")
        scheduler_cfg.get("min_lr")

        if "ReduceLROnPlateau" in main_scheduler.get("class_path", ""):
            if factor is not None:
                main_init_args["factor"] = factor
            if patience is not None:
                main_init_args["patience"] = patience

    @staticmethod
    def _update_gradient_clip(gradient_clip_cfg: dict | None, config: dict) -> None:
        """Update gradient clipping in the config.

        The value is stored at ``config["gradient_clip_val"]`` and passed to ``pl.Trainer``.
        """
        if gradient_clip_cfg is None:
            logger.info("Gradient clip parameters are not provided, skipping update.")
            return

        enable = gradient_clip_cfg.get("enable", False)
        if enable:
            max_grad_norm = gradient_clip_cfg.get("max_grad_norm")
            if max_grad_norm is not None:
                config["engine"]["gradient_clip_val"] = max_grad_norm
        else:
            # Explicitly disable gradient clipping
            config["engine"]["gradient_clip_val"] = None

    @staticmethod
    def _update_gradient_accumulation(gradient_accum_cfg: dict | None, config: dict) -> None:
        """Update gradient accumulation in the config.

        The value is stored at ``config["engine"]["accumulate_grad_batches"]``.
        """
        if gradient_accum_cfg is None:
            logger.info("Gradient accumulation parameters are not provided, skipping update.")
            return

        enable = gradient_accum_cfg.get("enable", False)
        if enable:
            batches = gradient_accum_cfg.get("batches", 1)
            if batches > 1:
                config.setdefault("engine", {})
                config["engine"]["accumulate_grad_batches"] = batches
        # Disable accumulation (set to 1 or remove)
        elif "engine" in config and "accumulate_grad_batches" in config.get("engine", {}):
            config["engine"]["accumulate_grad_batches"] = 1


class GetiConfigConverter:
    """Convert Geti model manifest to getitune recipe dictionary.

    Example:
        The following examples show how to use the Converter class.
        We expect a config file with ModelTemplate information in json form.

        Convert template.json to dictionary::

            converter = GetiConfigConverter()
            config = converter.convert("train_config.yaml")

        Instantiate an object from the configuration dictionary::

            engine, train_kwargs = converter.instantiate(
                config=config,
                work_dir="getitune-workspace",
                data_root="tests/assets/detection_coco",
            )

        Train the model::

            engine.train(**train_kwargs)
    """

    @staticmethod
    def convert(config: dict) -> dict:
        """Convert a geti configuration file to a default configuration dictionary.

        Args:
            config (dict): The path to the Geti yaml configuration file.

        Returns:
            dict: The default configuration dictionary.

        """
        from getitune.utils import get_getitune_root_path

        hyper_parameters = config["hyper_parameters"]

        resolver = RecipeResolver(get_getitune_root_path() / "recipe")
        recipe_path = resolver.resolve(
            config["model_manifest_id"],
            config["sub_task_type"],
        )

        if GetiConfigConverter._is_ultralytics_recipe(recipe_path):
            return GetiConfigConverter._convert_ultralytics(recipe_path, config, hyper_parameters)

        return GetiConfigConverter._convert_lightning(recipe_path, config, hyper_parameters)

    @staticmethod
    def _is_ultralytics_recipe(recipe_path: Path) -> bool:
        """Return whether a recipe declares the Ultralytics backend."""
        with recipe_path.open() as f:
            recipe = yaml.safe_load(f)
        return isinstance(recipe, dict) and recipe.get("backend") == "ultralytics"

    @staticmethod
    def _convert_ultralytics(recipe_path: Path, config: dict, hyper_parameters: dict) -> dict:
        from getitune.backend.ultralytics.tools.configurator import Configurator as UltralyticsConfigurator

        config_dict = UltralyticsConfigurator.convert(recipe_path, hyper_parameters)
        # Apply the standard augmentation / tiling updates to the data section.
        # This is the same TransformsUpdater path that Lightning recipes use,
        # so all augmentations are supported identically across backends.
        if hyper_parameters:
            GetiConfigConverter._update_data_transforms(config_dict, hyper_parameters)
        GetiConfigConverter._apply_intensity_mapping_from_task_params(config_dict, config)
        return config_dict

    @staticmethod
    def _convert_lightning(recipe_path: Path, config: dict, hyper_parameters: dict) -> dict:
        from getitune.tools.auto_configurator import AutoConfigurator

        augmentation_cfg = hyper_parameters.get("dataset_preparation", {}).get("augmentation", {})
        tiling_enabled = augmentation_cfg.get("tiling", {}).get("enable", False)
        if tiling_enabled and "_tile" not in recipe_path.stem:
            recipe_path = recipe_path.with_name(f"{recipe_path.stem}_tile.yaml")

        default_config = AutoConfigurator(model=recipe_path).config

        manifest_id = config["model_manifest_id"]
        is_timm = TimmManifestProvider.is_timm_id(manifest_id)
        if is_timm:
            GetiConfigConverter._inject_timm_backbone(default_config, id_to_model_name(manifest_id))

        if hyper_parameters:
            GetiConfigConverter._update_params(default_config, hyper_parameters)
        GetiConfigConverter._apply_intensity_mapping_from_task_params(default_config, config)
        GetiConfigConverter._remove_unused_key(default_config)

        if is_timm:
            GetiConfigConverter._assert_no_dynamic_placeholders(default_config)

        return default_config

    @staticmethod
    def _apply_intensity_mapping_from_task_params(target_config: dict, geti_config: dict) -> None:
        task_level_params = geti_config.get("task_level_parameters", {})
        intensity_mapping = task_level_params.get("dataset_preparation", {}).get("intensity_mapping")
        if intensity_mapping:
            GetiConfigConverter._update_intensity_mapping(target_config, intensity_mapping)

    @staticmethod
    def _inject_timm_backbone(config: dict, model_name: str) -> None:
        """Inject the selected timm backbone name and its native preprocessing.

        timm_generic.yaml is architecture-agnostic; the concrete backbone and its
        pretrained_cfg-derived params (input size, mean, std) are only known once
        the user has selected a specific backbone (encoded in model_manifest_id).
        """
        from app.supported_models.timm import TimmManifestProvider

        preprocessing = TimmManifestProvider.get_preprocessing(model_name)
        init_args = config["model"]["init_args"]
        init_args["model_name"] = model_name
        # Sync the data pipeline's resize target and normalization with the backbone.
        config["data"]["input_size"] = list(preprocessing["input_size"])
        mean, std = list(preprocessing["mean"]), list(preprocessing["std"])
        for subset_key in ("train_subset", "val_subset", "test_subset"):
            subset = config["data"].get(subset_key)
            if subset is None:
                raise ValueError(f"Missing '{subset_key}' in recipe config")
            for stage in ("augmentations_gpu", "augmentations_cpu"):
                for aug in subset.get(stage, []):
                    if "Normalize" in aug.get("class_path", ""):
                        aug.setdefault("init_args", {}).update(mean=mean, std=std)
                        break

    @staticmethod
    def _assert_no_dynamic_placeholders(config: dict) -> None:
        """Fail fast if any `DYNAMIC` placeholder survived hyperparameter resolution.

        Indicates the training configuration was not fully seeded from the
        selected model manifest's defaults before conversion.
        """
        optimizer_args = config["model"]["init_args"].get("optimizer", {}).get("init_args", {})
        unresolved = [k for k, v in optimizer_args.items() if isinstance(v, float) and math.isnan(v)]
        model_args = config["model"]["init_args"]
        if model_args.get("model_name") == "DYNAMIC":
            unresolved.append("model_name")
        if unresolved:
            msg = f"Unresolved DYNAMIC placeholder(s) in timm recipe: {unresolved}"
            raise ValueError(msg)

    @staticmethod
    def _get_params(hyperparameters: dict) -> dict:
        """Get configurable parameters from ModelTemplate config hyperparameters field."""
        param_dict = {}
        for param_name, param_info in hyperparameters.items():
            if isinstance(param_info, dict):
                if "value" in param_info:
                    param_dict[param_name] = param_info["value"]
                else:
                    param_dict = param_dict | GetiConfigConverter._get_params(param_info)

        return param_dict

    @staticmethod
    def _update_params(config: dict, param_dict: dict) -> None:
        """Update params of getitune recipe from Geti configurable params.

        Uses TransformsUpdater and HyperparametersUpdater classes to apply updates
        from the Geti model template to the getitune recipe config.
        """
        augmentation_params = param_dict.get("dataset_preparation", {}).get("augmentation", {})
        tiling = augmentation_params.pop("tiling", None)
        deim_framework = augmentation_params.pop("deim_framework", None)
        training_parameters = param_dict.get("training", {})

        # Update tiling first so downstream augmentation handling can react to it.
        TransformsUpdater.update_tiling(tiling, config)
        tile_enabled = bool(config["data"].get("tile_config", {}).get("enable_tiler", False))

        # The DEIM adaptive augmentation scheduling framework (mosaic/mixup based)
        # is incompatible with tiling. The tile recipe intentionally omits the
        # AugmentationSchedulerCallback and ships a static tiling-friendly pipeline,
        # so DEIM must be treated as disabled whenever tiling is enabled, regardless
        # of the requested value. Otherwise the converter would silently ignore the
        # user's augmentations (deim=True) or add incompatible ones for a tiling run.
        if not deim_framework or tile_enabled:
            # DEIM disabled by the user, or forced off because tiling is on ->
            # remove the scheduler callback and fall back to the static pipeline.
            TransformsUpdater.update(augmentation_params, config)
            GetiConfigConverter._disable_deim_framework(config)

        # Update training hyperparameters
        hyperparams: dict[str, Any] = {
            "learning_rate": training_parameters.get("learning_rate"),
            "batch_size": training_parameters.get("batch_size"),
            "max_epochs": training_parameters.get("max_epochs"),
            "early_stopping": training_parameters.get("early_stopping"),
            "input_size": (
                training_parameters.get("input_size_height"),
                training_parameters.get("input_size_width"),
            ),
            "weight_decay": training_parameters.get("weight_decay"),
            "scheduler": training_parameters.get("scheduler"),
            "gradient_clip": training_parameters.get("gradient_clip"),
            "gradient_accumulation": training_parameters.get("gradient_accumulation"),
        }
        HyperparametersUpdater.update(hyperparams, config)

    @staticmethod
    def get_callback_idx(callbacks: list, name: str) -> int:
        """Return required callbacks index from callback list."""
        for idx, callback in enumerate(callbacks):
            if callback["class_path"] == name:
                return idx
        return -1

    @staticmethod
    def _disable_deim_framework(config: dict) -> None:
        """Disable the DEIM adaptive augmentation scheduling framework.

        Removes the AugmentationSchedulerCallback from the callbacks list,
        falling back to the static augmentation pipeline defined in
        ``data.train_subset``.
        """
        callbacks = config.get("callbacks", [])
        idx = GetiConfigConverter.get_callback_idx(
            callbacks,
            "getitune.backend.lightning.callbacks.aug_scheduler.AugmentationSchedulerCallback",
        )
        if idx > -1:
            callbacks.pop(idx)
            logger.info("DEIM framework disabled: removed AugmentationSchedulerCallback")

    @staticmethod
    def _update_intensity_mapping(config: dict, intensity_mapping: dict) -> None:
        """Apply intensity mapping parameters to the data subset configs.

        Maps the Geti application intensity_mapping parameters to the library's
        IntensityConfig format and sets them on train/val/test subsets.

        Args:
            config: The full getitune config dictionary.
            intensity_mapping: Dict with keys: mode, max_intensity_value, clip_min_value,
                clip_max_value, window_center, window_width, scale_factor.
        """
        mode = intensity_mapping.get("mode", "scale_to_unit")
        intensity_config: dict[str, Any] = {"mode": mode}

        if mode == "scale_to_unit":
            intensity_config["max_value"] = intensity_mapping.get("max_intensity_value", 255.0)
        elif mode == "window":
            intensity_config["window_center"] = intensity_mapping.get("window_center", 127.5)
            intensity_config["window_width"] = intensity_mapping.get("window_width", 255.0)
        elif mode == "range_scale":
            intensity_config["scale_factor"] = intensity_mapping.get("scale_factor", 1.0)
            intensity_config["min_value"] = intensity_mapping.get("clip_min_value", 0.0)
            intensity_config["max_value"] = intensity_mapping.get("clip_max_value", 255.0)
        else:
            raise ValueError(f"Unsupported intensity mapping mode: {mode}")

        # Apply to all subsets
        for subset_key in ("train_subset", "val_subset", "test_subset"):
            if subset_key in config.get("data", {}):
                config["data"][subset_key]["intensity"] = intensity_config

    @staticmethod
    def _remove_unused_key(config: dict) -> None:
        """Remove unused keys from the config dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        config.pop("config")  # Remove config key that for CLI
        config["data"].pop("__path__", None)  # Remove __path__ key that for CLI overriding

    @staticmethod
    def _update_data_transforms(config: dict, param_dict: dict) -> None:
        """Apply augmentation and tiling updates to the ``data`` section of *config*.

        This is the shared path used by **all** backends — the same
        ``TransformsUpdater`` logic that Lightning uses also runs for
        Ultralytics configs, so every augmentation in the getitune pipeline
        is supported identically.
        """
        augmentation_params = param_dict.get("dataset_preparation", {}).get("augmentation", {})
        if not augmentation_params:
            return

        # Work on a copy so we don't mutate the caller's dict.
        augmentation_params = dict(augmentation_params)
        tiling = augmentation_params.pop("tiling", None)
        augmentation_params.pop("deim_framework", None)  # Not applicable for Ultralytics

        TransformsUpdater.update_tiling(tiling, config)
        TransformsUpdater.update(augmentation_params, config)
