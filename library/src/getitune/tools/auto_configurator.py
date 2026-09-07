# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Auto-Configurator class & util functions for getitune Auto-Configuration."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from jsonargparse import ArgumentParser, Namespace

from getitune.backend.lightning.models.base import DataInputParams, LightningModel
from getitune.config.data import SamplerConfig, SubsetConfig, TileConfig
from getitune.data.factory import TransformLibFactory
from getitune.data.module import DataModule
from getitune.types import PathLike
from getitune.types.label import LabelInfoTypes
from getitune.types.task import TaskType
from getitune.utils import get_getitune_root_path, list_models
from getitune.utils.utils import can_pass_tile_config, get_model_cls_from_config, should_pass_label_info

if TYPE_CHECKING:
    from getitune.backend.openvino.models.base import OVModel


logger = logging.getLogger()
RECIPE_PATH = get_getitune_root_path() / "recipe"

# Memory budget (in MiB) for the decoded image tensors of a *single* OpenVINO
# evaluation batch.  The ``openvino_model.yaml`` recipes hard-code a batch size
# (e.g. 64 for instance segmentation) that was tuned for low-resolution models.
# That same batch size becomes a memory bomb for high-resolution architectures
# such as MaskRCNN SwinT (1344x1344): 64 * 3 * 1344 * 1344 * 4 B ~= 1.4 GiB of
# float32 pixels *per batch*, multiplied again by the dataloader prefetch queue
# (num_workers * prefetch_factor) and once more by the per-instance full-image
# masks that ModelAPI allocates during post-processing.  The resulting RSS spike
# gets the worker process killed by the OS OOM killer, which surfaces as a
# training job that dies during evaluation without any Python traceback.
#
# Capping the batch size by a memory budget keeps evaluation results identical
# (batching only affects throughput, not metrics) while making the peak memory
# roughly independent of the model input resolution.
OV_EVAL_BATCH_BUDGET_MB = int(os.environ.get("GETITUNE_OV_EVAL_BATCH_BUDGET_MB", "256"))

DEFAULT_CONFIG_PER_TASK = {
    TaskType.MULTI_CLASS_CLS: RECIPE_PATH / "classification" / "multi_class_cls" / "mobilenet_v3_large.yaml",
    TaskType.MULTI_LABEL_CLS: RECIPE_PATH / "classification" / "multi_label_cls" / "mobilenet_v3_large.yaml",
    TaskType.DETECTION: RECIPE_PATH / "detection" / "yolox_s.yaml",
    TaskType.SEMANTIC_SEGMENTATION: RECIPE_PATH / "semantic_segmentation" / "litehrnet_18.yaml",
    TaskType.INSTANCE_SEGMENTATION: RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_small.yaml",
    TaskType.KEYPOINT_DETECTION: RECIPE_PATH / "keypoint_detection" / "rtmpose_tiny.yaml",
}


OVMODEL_PER_TASK = {
    TaskType.MULTI_CLASS_CLS: "getitune.backend.openvino.models.OVMulticlassClassificationModel",
    TaskType.MULTI_LABEL_CLS: "getitune.backend.openvino.models.OVMultilabelClassificationModel",
    TaskType.DETECTION: "getitune.backend.openvino.models.OVDetectionModel",
    TaskType.INSTANCE_SEGMENTATION: "getitune.backend.openvino.models.OVInstanceSegmentationModel",
    TaskType.SEMANTIC_SEGMENTATION: "getitune.backend.openvino.models.OVSegmentationModel",
    TaskType.KEYPOINT_DETECTION: "getitune.backend.openvino.models.OVKeypointDetectionModel",
}


class AutoConfigurator:
    """This Class is used to configure the DataModule, LightningModel, Optimizer, and Scheduler with getitune Default.

    Args:
        data_root (PathLike | None, optional): The root directory for data storage. Defaults to None.
        task (TaskType | None, optional): The task type. If None, the task will be configured based on the model.
            Defaults to None.
        model (PathLike | str | None, optional): Path to the model config file or name of the model to use.
            If None, the task should be provided and the default model for the task will be used.
            Defaults to None.

    Example:
        The following examples show how to use the AutoConfigurator class.

        >>> auto_configurator = AutoConfigurator(
        ...     data_root=<dataset/path>,
        ...     task=<TaskType>,
        ... )

        # If task is None, the task will be configured based on the data root.
        >>> auto_configurator = AutoConfigurator(
        ...     data_root=<dataset/path>,
        ... )
    """

    def __init__(
        self,
        data_root: PathLike | None = None,
        task: TaskType | None = None,
        model: PathLike | str | None = None,
    ) -> None:
        self.data_root = data_root
        self._task = task
        model_config_path: PathLike | None = None
        if model is not None:
            if not str(model).endswith(".yaml"):
                if task is None:
                    msg = "If model is provided as a name, task must be provided to find the model."
                    raise ValueError(msg)
                recipe_list = list_models(task=task, pattern=str(model), return_recipes=True)
                if len(recipe_list) > 1:
                    msg = (
                        "There is more than 1 model match the given name."
                        "It may happen with overlap of the tasks. Using the first one."
                        "To use the specific model, provide model config instead."
                    )
                    logger.warning(msg)
                elif len(recipe_list) == 0:
                    msg = f"Model {model} does not exist."
                    raise FileNotFoundError(msg)
                model_config_path = recipe_list[0]
            else:
                model_config_path = model
            if not Path(model_config_path).exists():
                msg = f"Model config path {model} does not exist."
                raise FileNotFoundError(msg)
        if model_config_path:
            self._config: dict = self._load_default_config(config_path=model_config_path)
            self._task = TaskType(self._config.get("task", task))
        elif task:
            self._config = self._load_default_config(task=task)
        else:
            msg = "Either task or model must be provided."
            raise ValueError(msg)

    @property
    def task(self) -> TaskType:
        """Returns the current task.

        Raises:
            RuntimeError: If there are no ready tasks.

        Returns:
            TaskType | str: The current task.
        """
        if self._task is not None:
            return self._task
        if self._config is not None and "task" in self._config:
            return TaskType(self._config["task"])
        msg = "There are no ready task"
        raise RuntimeError(msg)

    @property
    def config(self) -> dict:
        """Retrieves the configuration for the auto configurator.

        Returns:
            dict: The configuration as a dict object.
        """
        return self._config

    def _load_default_config(self, config_path: PathLike | None = None, task: TaskType | None = None) -> dict:
        """Load the default configuration for the specified model.

        Args:
            model_name (str | None): The name of the model. If provided, the configuration
                file name will be modified to use the specified model.

        Returns:
            dict: The loaded configuration.

        Raises:
            ValueError: If the task doesn't supported for auto-configuration.
        """
        from getitune.cli.utils.jsonargparse import get_configuration

        task = task if task is not None else self._task
        if config_path is None:
            if task is None:
                msg = "Either config_path or task must be provided."
                raise ValueError(msg)
            config_path = DEFAULT_CONFIG_PER_TASK[task]

        return get_configuration(config_path)

    def get_datamodule(self, data_root: PathLike | None = None) -> DataModule:
        """Returns an instance of DataModule with the configured data root.

        Returns:
            DataModule | None: An instance of DataModule.
        """
        if data_root is None and self.data_root is None:
            msg = "No data root provided."
            raise ValueError(msg)
        if data_root is not None and not isinstance(data_root, (str, os.PathLike)):
            msg = f"data_root should be of type PathLike, but got {type(data_root)}"
            raise TypeError(msg)

        data_root = data_root if data_root is not None else self.data_root
        self.config["data"]["data_root"] = data_root
        data_config: dict = deepcopy(self.config["data"])
        train_config = data_config.pop("train_subset")
        val_config = data_config.pop("val_subset")
        test_config = data_config.pop("test_subset")
        tile_config = data_config.pop("tile_config", {})

        _ = data_config.pop("__path__", {})  # Remove __path__ key that for CLI
        _ = data_config.pop("config", {})  # Remove config key that for CLI

        return DataModule(
            train_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
            val_subset=SubsetConfig(sampler=SamplerConfig(**val_config.pop("sampler", {})), **val_config),
            test_subset=SubsetConfig(sampler=SamplerConfig(**test_config.pop("sampler", {})), **test_config),
            tile_config=TileConfig(**tile_config),
            **data_config,
        )

    def get_model(
        self,
        model_name: str | None = None,
        label_info: LabelInfoTypes | None = None,
        data_input_params: DataInputParams | dict | None = None,
    ) -> LightningModel:
        """Retrieves the LightningModel instance based on the provided model name and meta information.

        Args:
            model_name (str | None): The name of the model to retrieve. If None, the default model will be used.
            label_info (LabelInfoTypes | None): The meta information about the labels.
                If provided, the number of classes will be updated in the model's configuration.
            data_input_params (DataInputParams | dict | None, optional): The data input parameters
                containing the input size, input mean and std.

        Returns:
            LightningModel: The instantiated LightningModel instance.

        Example:
            The following examples show how to get the LightningModel class.

            # If model_name is None, the default model will be used from task.
            >>> auto_configurator.get_model(
            ...     label_info=<LabelInfo>,
            ... )

            # If model_name is str, the default config file is changed.
            >>> auto_configurator.get_model(
            ...     model_name=<model_name, str>,
            ...     label_info=<LabelInfo>,
            ... )
        """
        # TODO(vinnamki): There are some overlaps with src/getitune/cli/cli.py::CLI::instantiate_model
        if model_name is not None:
            self._config = self._load_default_config(model_name)

        skip = set()

        model_config = deepcopy(self.config["model"])

        if data_input_params is not None:
            model_config["init_args"]["data_input_params"] = (
                data_input_params if isinstance(data_input_params, dict) else data_input_params.as_dict()
            )
        elif (datamodule := self.get_datamodule()) is not None:
            # get data_input_params info from datamodule
            if datamodule.input_size is None:
                msg = (
                    "Input size is not specified in the datamodule. Ensure that the datamodule has a valid input size."
                )
                raise ValueError(msg)
            # NOTE: pass mean/std through as-is. None when the CPU augmentation pipeline has no torchvision Normalize to
            # derive them from, such as when normalization lives in augmentations_gpu instead
            model_config["init_args"]["data_input_params"] = DataInputParams(
                input_size=datamodule.input_size,
                mean=datamodule.input_mean,
                std=datamodule.input_std,
            ).as_dict()

        model_cls = get_model_cls_from_config(Namespace(model_config))

        if should_pass_label_info(model_cls):
            if label_info is None:
                msg = f"Given model class {model_cls} requires a valid label_info to instantiate."
                raise ValueError(msg)

            model_config["init_args"]["label_info"] = label_info
            skip.add("label_info")

        if can_pass_tile_config(model_cls) and (datamodule := self.get_datamodule()) is not None:
            model_config["init_args"]["tile_config"] = datamodule.tile_config
            skip.add("tile_config")

        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(
            LightningModel,
            "model",
            skip=skip,
            required=False,
            fail_untyped=False,
        )
        return model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

    def get_ov_model(self, model_name: PathLike, task: TaskType | None = None) -> OVModel:
        """Retrieves the OVModel instance based on the given model name and label information.

        Args:
            model_name (str): The name of the model.
            label_info (LabelInfo): The label information.

        Returns:
            OVModel: The OVModel instance.

        Raises:
            NotImplementedError: If the OVModel for the given task is not supported.
        """
        task = task if task is not None else self.task
        class_path = OVMODEL_PER_TASK.get(task)
        if class_path is None:
            msg = f"{task} doesn't support OVModel."
            raise NotImplementedError(msg)
        class_module, class_name = class_path.rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        ov_model = getattr(module, class_name)
        return ov_model(
            model_path=model_name,
        )

    def update_ov_subset_pipeline(
        self,
        datamodule: DataModule,
        subset: str = "test",
        task: TaskType | None = None,
        input_size: tuple[int, int] | None = None,
        keep_aspect_ratio: bool = False,
        center_padding: bool = False,
        pad_value: int = 0,
    ) -> DataModule:
        """Returns an DataModule object with OpenVINO subset transforms applied.

        Args:
            datamodule (DataModule): The original DataModule object.
            subset (str, optional): The subset to update. Defaults to "test".
            input_size (tuple[int, int] | None, optional): Model input size (H, W)
                from the OV model metadata.  When provided this overrides
                ``datamodule.input_size`` so that ``$(input_size)`` placeholders
                in the OV recipe augmentations resolve to the correct value.
            keep_aspect_ratio (bool, optional): When ``True`` and the model uses
                letterbox preprocessing (fit_to_window_letterbox), every Resize
                transform in the OV pipeline is patched to preserve aspect ratio
                (letterbox-style) instead of stretching. This avoids padding/pad-value
                mismatches between the DataModule Resize and the model's native
                letterbox.  When ``False``, the OV recipe augmentations are
                applied as usual (simple stretch resize).
                Defaults to ``False``.
            center_padding (bool, optional): When ``True``, patches Resize
                transforms to use centered letterbox padding (equal on both
                sides).  This matches ``fit_to_window_letterbox`` preprocessing.
                Defaults to ``False``.
            pad_value (int, optional): Padding fill value for Resize when
                ``keep_aspect_ratio`` is ``True``.  YOLO models use ``114``
                (gray); default is ``0`` (black).

        Returns:
            DataModule: The modified DataModule object with OpenVINO subset transforms applied.
        """
        task = task if task is not None else self._task
        if task is None:
            msg = "Task must be provided to update OpenVINO subset pipeline."
            raise ValueError(msg)
        ov_config_path = DEFAULT_CONFIG_PER_TASK[task].parent / "openvino_model.yaml"
        ov_config = self._load_default_config(config_path=ov_config_path)["data"]
        subset_config = getattr(datamodule, f"{subset}_subset")
        ov_subset = ov_config.get(f"{subset}_subset", ov_config["test_subset"])

        # Capture the tiling intent before overwriting the subset augmentations below.
        tiling_enabled = datamodule.tile_config.enable_tiler

        if keep_aspect_ratio:
            subset_config.batch_size = ov_subset["batch_size"]
            subset_config.augmentations_cpu = ov_subset["augmentations_cpu"]
            subset_config.augmentations_gpu = ov_subset.get("augmentations_gpu", [])
            self._patch_resize_keep_aspect_ratio(
                subset_config.augmentations_cpu,
                center_padding=center_padding,
                pad_value=pad_value,
            )
            self._patch_resize_keep_aspect_ratio(
                subset_config.augmentations_gpu,
                center_padding=center_padding,
                pad_value=pad_value,
            )
        else:
            subset_config.batch_size = ov_subset["batch_size"]
            subset_config.augmentations_cpu = ov_subset["augmentations_cpu"]
            subset_config.augmentations_gpu = ov_subset.get("augmentations_gpu", [])

        if tiling_enabled:
            # ModelAPI's tiler performs tiling, resizing and coordinate/mask mapping
            # internally on native-resolution crops (see OVModel.forward_tiles ->
            # Tiler.predict_tiles). Keep DataModule tiling enabled so the loader emits
            # per-image tile batches, and strip the model-input Resize from the OV
            # pipeline so tiles reach ModelAPI at their native resolution — resizing
            # tiles here would both duplicate the resize and corrupt the tile-to-image
            # coordinate mapping. Intensity scaling is preserved (the CPU augmentation
            # pipeline prepends its intensity transform independently of Resize).
            self._strip_resize_transforms(subset_config.augmentations_cpu)

            # IMPORTANT: the OV recipe's batch_size (e.g. 64) is tuned for *non-tiled*
            # evaluation, where one dataset item == one model input. With tiling
            # enabled, a single original image expands into *every* grid tile inside
            # the dataset (see DataModule._eval_loader_kwargs), so a loader batch_size
            # of N images can balloon into N * num_tiles tiles collated into a single
            # in-memory batch. With tile_size=400/overlap=0.2 on large images this can
            # produce a batch so large that building/collating it appears to hang
            # (severe memory pressure / thrashing) rather than raising an error.
            # The model still groups tiles into TileConfig.tile_inference_batch_size
            # chunks for the actual forward passes, so the loader batch_size can (and
            # must) stay small here regardless of what the OV recipe specifies.
            if subset_config.batch_size != 1:
                logger.info(
                    "update_ov_subset_pipeline: tiling is enabled, overriding OV recipe "
                    "%s_subset.batch_size (%d) -> 1 to avoid collating %d images' worth of "
                    "tiles (tile_size=%s, overlap=%s) into a single dataloader batch, which "
                    "can hang/OOM. Tile-level batching for the model forward pass is still "
                    "controlled independently via tile_config.tile_inference_batch_size=%d.",
                    subset,
                    subset_config.batch_size,
                    subset_config.batch_size,
                    datamodule.tile_config.tile_size,
                    datamodule.tile_config.overlap,
                    datamodule.tile_config.tile_inference_batch_size,
                )
                subset_config.batch_size = 1
        else:
            datamodule.tile_config.enable_tiler = False

        # Resolve input size: prefer model IR metadata, fall back to
        # the training recipe's default, raise if neither is available.
        actual_input_size = input_size or datamodule.input_size
        subset_config.input_size = actual_input_size

        if actual_input_size is None:
            msg = (
                "Cannot determine input_size for the OpenVINO pipeline. "
                "The OV model has dynamic input shapes and the datamodule "
                "does not specify input_size. Please provide input_size "
                "explicitly when calling update_ov_subset_pipeline()."
            )
            raise ValueError(msg)

        if not tiling_enabled:
            # The recipe batch size is resolution-agnostic; clamp it so that a
            # high-resolution model does not blow up host memory (see
            # OV_EVAL_BATCH_BUDGET_MB).  Tiled pipelines already force batch_size=1.
            subset_config.batch_size = self._cap_eval_batch_size(
                batch_size=subset_config.batch_size,
                input_size=actual_input_size,
                subset=subset,
            )

        msg = (
            f"For OpenVINO IR models, Update the following {subset} \n"
            f"\t augmentations_cpu: {subset_config.augmentations_cpu} \n"
            f"\t batch_size: {subset_config.batch_size} \n"
            + (
                "And the tiler is kept enabled; ModelAPI performs tiled inference on native crops."
                if tiling_enabled
                else "And the tiler is disabled."
            )
        )
        warn(msg, stacklevel=1)

        # If the datamodule was created from pre-constructed datasets (no data_root),
        # rebuild using from_vision_datasets to avoid re-importing from disk.
        # This is useful for the quantization pipeline.
        if not datamodule.data_root and datamodule.subsets:
            datamodule.train_subset.input_size = actual_input_size

            if tiling_enabled:
                existing_dataset = datamodule.subsets[subset_config.subset_name]
                existing_dataset.transforms = TransformLibFactory.generate(subset_config)

            return DataModule.from_vision_datasets(
                train_dataset=datamodule.subsets[datamodule.train_subset.subset_name],
                val_dataset=datamodule.subsets[datamodule.val_subset.subset_name],
                test_dataset=datamodule.subsets.get(datamodule.test_subset.subset_name),
                train_subset=datamodule.train_subset,
                val_subset=datamodule.val_subset,
                test_subset=datamodule.test_subset,
                auto_num_workers=datamodule.auto_num_workers,
                device=datamodule.device,
            )

        return DataModule(
            task=datamodule.task,
            data_root=datamodule.data_root,
            train_subset=datamodule.train_subset,
            val_subset=datamodule.val_subset,
            test_subset=datamodule.test_subset,
            input_size=actual_input_size,
            tile_config=datamodule.tile_config,
            ignore_index=datamodule.ignore_index,
            unannotated_items_ratio=datamodule.unannotated_items_ratio,
            auto_num_workers=datamodule.auto_num_workers,
            device=datamodule.device,
        )

    @staticmethod
    def _cap_eval_batch_size(batch_size: int, input_size: tuple[int, int], subset: str = "test") -> int:
        """Clamp an OpenVINO evaluation batch size to a host-memory budget.

        The ``openvino_model.yaml`` recipes declare a single, resolution-agnostic
        batch size (64 for most tasks).  Combined with a high-resolution model such
        as MaskRCNN SwinT (1344x1344) this makes one dataloader batch hold ~1.4 GiB
        of float32 pixels, which is then multiplied by the dataloader prefetch queue
        and by the per-instance, full-image masks produced during post-processing.
        The resulting allocation spike is typically resolved by the OS OOM killer,
        i.e. the evaluation process dies by SIGKILL without a Python traceback.

        Batch size only affects evaluation throughput, never the computed metrics,
        so clamping it is always safe.

        Args:
            batch_size: Batch size requested by the OpenVINO recipe.
            input_size: ``(H, W)`` the images are resized to before inference.
            subset: Name of the subset being configured, used for logging only.

        Returns:
            int: ``batch_size``, or a smaller value that fits the memory budget
            (never less than 1).
        """
        height, width = int(input_size[0]), int(input_size[1])
        if height <= 0 or width <= 0:
            return batch_size

        # 3 channels, float32 (the OV CPU pipeline scales pixels to float).
        bytes_per_image = 3 * height * width * 4
        budget_bytes = max(OV_EVAL_BATCH_BUDGET_MB, 1) * 1024 * 1024
        max_batch_size = max(1, budget_bytes // bytes_per_image)

        if batch_size <= max_batch_size:
            return batch_size

        logger.warning(
            "update_ov_subset_pipeline: OpenVINO recipe %s_subset.batch_size=%d would allocate "
            "~%.1f MiB of float32 pixels per batch at input_size=%dx%d (plus dataloader prefetch "
            "and per-instance full-image masks), which risks the evaluation process being killed "
            "by the OS OOM killer. Reducing batch_size to %d to stay within the %d MiB budget "
            "(override via GETITUNE_OV_EVAL_BATCH_BUDGET_MB). Metrics are unaffected.",
            subset,
            batch_size,
            batch_size * bytes_per_image / 1024 / 1024,
            height,
            width,
            max_batch_size,
            OV_EVAL_BATCH_BUDGET_MB,
        )
        return int(max_batch_size)

    @staticmethod
    def _strip_resize_transforms(augmentations: list[dict]) -> None:
        """Remove every Resize step from an augmentation list, in place.

        Used for the OpenVINO tiling path: ModelAPI's tiler resizes each native
        crop to the model input internally, so a DataModule-side Resize would both
        duplicate the resize and corrupt the tile-to-image coordinate mapping.

        Args:
            augmentations: List of augmentation config dicts, each with a
                ``class_path`` key (and optionally ``init_args``).
        """
        augmentations[:] = [aug for aug in augmentations if "Resize" not in aug.get("class_path", "")]

    @staticmethod
    def _patch_resize_keep_aspect_ratio(
        augmentations: list[dict],
        *,
        center_padding: bool = False,
        pad_value: int = 0,
    ) -> None:
        """Set ``keep_aspect_ratio=True`` on every Resize step in an augmentation list.

        The OV recipe templates default to ``keep_aspect_ratio: false``.  When the
        exported model's ``resize_type`` metadata indicates aspect-ratio-preserving
        resize was used during training, this method patches the configs so that
        OV inference preprocessing matches training preprocessing exactly.

        When ``center_padding`` is True, also sets ``center_padding=True`` on
        Resize transforms to match ``fit_to_window_letterbox`` preprocessing.

        Args:
            augmentations: List of augmentation config dicts, each with
                ``class_path`` and optionally ``init_args`` keys.
            center_padding: Whether to also set ``center_padding=True``.
            pad_value: Padding fill value for the Resize transform.  YOLO
                models use ``114`` (gray); default is ``0`` (black).
        """
        for aug in augmentations:
            if "Resize" in aug.get("class_path", ""):
                init_args = aug.setdefault("init_args", {})
                init_args["keep_aspect_ratio"] = True
                if center_padding:
                    init_args["center_padding"] = True
                if pad_value != 0:
                    init_args["pad_value"] = pad_value
