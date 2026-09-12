# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LightningDataModule extension for getitune."""

from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING

from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler

from getitune.config.data import SubsetConfig, TileConfig
from getitune.data.augmentation import CPUAugmentationPipeline
from getitune.data.dataset.tile import TileDatasetFactory
from getitune.data.entity.utils import detect_storage_dtype
from getitune.data.factory import DatasetFactory
from getitune.data.utils import get_adaptive_num_workers, instantiate_sampler
from getitune.types.device import DeviceType
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset
    from lightning.pytorch.utilities.parsing import AttributeDict

    from getitune.data.dataset.base import VisionDataset

logger = logging.getLogger(__name__)


_MP_CONTEXT = multiprocessing.get_context("spawn")


# Mapping from getitune subset config names to Datumaro experimental Subset enums
_SUBSET_NAME_TO_ENUM: dict[str, Subset] = {
    "train": Subset.TRAINING,
    "val": Subset.VALIDATION,
    "test": Subset.TESTING,
    "training": Subset.TRAINING,
    "validation": Subset.VALIDATION,
    "testing": Subset.TESTING,
}


class DataModule(LightningDataModule):
    """LightningDataModule extension for getitune.

    Handles data loading, transformation, and preparation for getitune pipelines.

    Args:
        task (TaskType): Task type (e.g., classification, detection).
        data_root (str): Root directory of the dataset.
        train_subset (SubsetConfig, optional): Training subset configuration. Defaults to None.
        val_subset (SubsetConfig, optional): Validation subset configuration. Defaults to None.
        test_subset (SubsetConfig, optional): Test subset configuration. Defaults to None.
        tile_config (TileConfig, optional): Tiling configuration. Defaults to TileConfig(enable_tiler=False).
        ignore_index (int, optional): Ignore index for segmentation. Defaults to 255.
        unannotated_items_ratio (float, optional): Ratio of unannotated items. Defaults to 0.0.
        auto_num_workers (bool, optional): Automatically determine number of workers. Defaults to False.
        device (DeviceType, optional): Device type ('cpu', 'gpu', etc.). Defaults to DeviceType.auto.
        input_size (tuple[int, int] | None, optional): Final image/video shape after transformation. Defaults to None.

    Note:
        To create an DataModule from pre-constructed datasets, use the `from_vision_datasets` class method.
    """

    def __init__(
        self,
        task: TaskType,
        data_root: str,
        train_subset: SubsetConfig | None = None,
        val_subset: SubsetConfig | None = None,
        test_subset: SubsetConfig | None = None,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        ignore_index: int = 255,
        unannotated_items_ratio: float = 0.0,
        auto_num_workers: bool = False,
        device: DeviceType = DeviceType.auto,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Constructor."""
        super().__init__()

        self.task = task
        self.data_root = data_root

        if input_size is not None and not isinstance(input_size, (tuple, list)):
            msg = f"input_size should be a tuple or list of ints, but got {input_size!r}"
            raise ValueError(msg)

        subset_configs = self.get_default_subset_configs(input_size)
        self.train_subset = train_subset if train_subset is not None else subset_configs["train_subset"]
        self.val_subset = val_subset if val_subset is not None else subset_configs["val_subset"]
        self.test_subset = test_subset if test_subset is not None else subset_configs["test_subset"]
        self.tile_config = tile_config

        self.ignore_index = ignore_index
        self.unannotated_items_ratio = unannotated_items_ratio

        self.auto_num_workers = auto_num_workers
        self.device = device

        self.subsets: dict[str, VisionDataset] = {}
        self.save_hyperparameters(ignore=["input_size"])

        dataset = import_dataset(self.data_root)

        if input_size is not None:
            # override input_size to all subset configs when it is given
            for subset_cfg in [self.train_subset, self.val_subset, self.test_subset]:
                if subset_cfg.input_size is None:
                    subset_cfg.input_size = input_size  # type: ignore[assignment]

        # Derive mean/std from the CPU pipeline's Normalize transform.
        # If no Normalize is present (e.g. GPU-only normalization via Kornia),
        # leave as None so models fall back to their own defaults.
        # The GPUAugmentationCallback.setup() will later override the model's
        # mean/std with the GPU pipeline's values if applicable.
        if getattr(self.train_subset, "augmentations_cpu", None):
            cpu_pipeline = CPUAugmentationPipeline.from_config(self.train_subset)
            self.input_mean: tuple[float, float, float] | None = cpu_pipeline.mean
            self.input_std: tuple[float, float, float] | None = cpu_pipeline.std
        else:
            self.input_mean = None
            self.input_std = None
        self.input_size = input_size

        self._setup_dataset(dataset)
        # Propagate intensity config from train subset for use in export.
        self.input_intensity_config = getattr(self.train_subset, "intensity", None)

    def _setup_dataset(self, dataset: Dataset) -> None:
        """Setup VisionDataset instances from a Datumaro experimental Dataset.

        Args:
            dataset: A ``datumaro.experimental.Dataset`` loaded via ``import_dataset``.
        """
        storage_dtype: str | None = None
        num_channels: int | None = None
        subset_configs = [self.train_subset, self.val_subset, self.test_subset]
        subset_names = [cfg.subset_name for cfg in subset_configs]
        if len(set(subset_names)) != len(subset_names):
            msg = (
                f"Subset names must be unique, got {subset_names}. Ensure each SubsetConfig has a distinct subset_name."
            )
            raise ValueError(msg)
        config_mapping = {cfg.subset_name: cfg for cfg in subset_configs}

        if self.auto_num_workers:
            if self.device not in [DeviceType.gpu, DeviceType.auto]:
                logger.warning(
                    "Only GPU device type support auto_num_workers. "
                    f"Current deveice type is {self.device!s}. auto_num_workers is skipped.",
                )
            elif (num_workers := get_adaptive_num_workers()) is not None:
                for subset_name, subset_config in config_mapping.items():
                    logger.info(
                        f"num_workers of {subset_name} subset is changed : "
                        f"{subset_config.num_workers} -> {num_workers}",
                    )
                    subset_config.num_workers = num_workers

        label_infos: list[LabelInfo] = []
        for name, subset_cfg in config_mapping.items():
            subset_enum = _SUBSET_NAME_TO_ENUM.get(name.lower())
            if subset_enum is None:
                logger.warning(f"{name} has no Subset enum mapping. Skip it")
                continue

            dm_subset = dataset.filter_by_subset(subset_enum)
            if len(dm_subset) == 0:
                logger.warning(f"Subset '{name}' is empty in the dataset. Skip it")
                continue

            if storage_dtype is None:
                storage_dtype, num_channels = detect_storage_dtype(dm_subset)

            if subset_cfg.intensity.storage_dtype != storage_dtype:
                logger.warning(
                    f"Overriding intensity storage_dtype '{subset_cfg.intensity.storage_dtype}' "
                    f"with auto-detected '{storage_dtype}'",
                )
                subset_cfg.intensity.storage_dtype = storage_dtype

            # Auto-set repeat_channels for single-channel (grayscale) images.
            # The model backbone expects 3-channel input, so channel-repeat is
            # needed for both the training pipeline and the exported OV model
            # (embedded preprocessing metadata).
            if num_channels == 1 and subset_cfg.intensity.repeat_channels < 3:
                logger.info(
                    f"Single-channel image detected ({num_channels}ch). "
                    f"Auto-setting intensity.repeat_channels=3 for subset '{name}'.",
                )
                subset_cfg.intensity.repeat_channels = 3

            subset_dataset = DatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset,
                cfg_subset=subset_cfg,
                storage_dtype=storage_dtype,
                ignore_index=self.ignore_index,
            )

            if self.tile_config.enable_tiler:
                subset_dataset = TileDatasetFactory.create(
                    dataset=subset_dataset,
                    tile_config=self.tile_config,
                )
            self.subsets[subset_cfg.subset_name] = subset_dataset
            label_infos += [self.subsets[subset_cfg.subset_name].label_info]

        if self._is_meta_info_valid(label_infos) is False:
            msg = "All data meta infos of subsets should be the same."
            raise ValueError(msg)

        self.label_info = next(iter(label_infos))

    @classmethod
    def from_vision_datasets(
        cls,
        train_dataset: VisionDataset,
        val_dataset: VisionDataset,
        test_dataset: VisionDataset | None = None,
        train_subset: SubsetConfig | None = None,
        val_subset: SubsetConfig | None = None,
        test_subset: SubsetConfig | None = None,
        auto_num_workers: bool = False,
        device: DeviceType = DeviceType.auto,
    ) -> DataModule:
        """Create an DataModule from pre-constructed VisionDataset instances.

        This is a factory method that provides a clean way to create DataModule instances
        when you already have constructed datasets, without needing to provide data_root
        or other data loading parameters.

        Args:
            train_dataset (VisionDataset): Pre-constructed training dataset.
            val_dataset (VisionDataset): Pre-constructed validation dataset.
            test_dataset (VisionDataset | None, optional): Pre-constructed test dataset. Defaults to None.
            train_subset (SubsetConfig): Configuration for the training dataloader.
                Must have ``input_size`` set to the fixed model input size (H, W).
                The ``input_size`` value is used to resolve ``$(input_size)`` placeholders
                in the augmentation pipeline and is exposed as ``datamodule.input_size``.
            val_subset (SubsetConfig | None, optional): Configuration for the validation dataloader.
                Defaults to None.
            test_subset (SubsetConfig | None, optional): Configuration for the test dataloader.
                Defaults to None.
            auto_num_workers (bool, optional): Whether to automatically determine the number of workers.
                Defaults to False.
            device (DeviceType, optional): Device type to use (e.g., 'cpu', 'gpu').
                Defaults to DeviceType.auto.

        Returns:
            DataModule: Configured data module with the provided datasets.

        Raises:
            ValueError: If datasets have inconsistent label metadata.
            ValueError: If ``train_subset`` is None or ``train_subset.input_size`` is not set.

        Examples:
            >>> from getitune.config.data import SubsetConfig
            >>> from getitune.data.module import DataModule
            >>>
            >>> train_config = SubsetConfig(
            ...     batch_size=8,
            ...     input_size=(512, 512),
            ... )
            >>> datamodule = DataModule.from_vision_datasets(
            ...     train_dataset=my_train_dataset,
            ...     val_dataset=my_val_dataset,
            ...     test_dataset=my_test_dataset,
            ...     train_subset=train_config,
            ... )
        """
        # Validate label info consistency across datasets
        if test_dataset is None:
            test_dataset = val_dataset

        if not all(
            label_info == train_dataset.label_info for label_info in [val_dataset.label_info, test_dataset.label_info]
        ):
            msg = "All data meta infos of provided datasets should be the same."
            raise ValueError(msg)

        # Create instance
        instance = cls.__new__(cls)
        LightningDataModule.__init__(instance)
        # Set basic attributes
        instance.task = train_dataset.task_type  # type: ignore[assignment]
        instance.data_root = ""
        instance.tile_config = (
            train_dataset.tile_config if hasattr(train_dataset, "tile_config") else TileConfig(enable_tiler=False)
        )
        instance.ignore_index = 255
        instance.unannotated_items_ratio = 0.0
        instance.auto_num_workers = auto_num_workers
        instance.device = device

        # Store datasets and label info
        instance.label_info = train_dataset.label_info

        # input_size must come from the subset config (set by recipe / manifest).
        if train_subset is not None and getattr(train_subset, "input_size", None) is not None:
            instance.input_size = train_subset.input_size
        else:
            msg = (
                "input_size is not set on the train_subset config. "
                "When using from_vision_datasets, the caller must provide a SubsetConfig "
                "with an explicit input_size (from the recipe or application manifest)."
            )
            raise ValueError(msg)

        # merge default configs with provided subsets
        default_subset_configs: dict[str, SubsetConfig] = {}  # Initialize lazily if needed)
        for name, subset in zip(["train", "val", "test"], [train_subset, val_subset, test_subset]):
            if subset is not None:
                # Use provided subset config
                subset_to_assign = subset
                if getattr(subset, "augmentations_cpu", None):
                    logger.warning(
                        f"The provided {name} SubsetConfig contains augmentations_cpu which will be overridden "
                        "by the transforms of the provided VisionDataset. When building DataModule from "
                        "pre-constructed datasets, developers should set up the transforms when creating the datasets.",
                    )
            else:
                # Use default config but get transforms from the dataset
                if not default_subset_configs:
                    default_subset_configs = instance.get_default_subset_configs(instance.input_size)
                subset_to_assign = default_subset_configs[f"{name}_subset"]

            # The pre-constructed datasets already have their transforms configured.
            # No need to override - just set the subset config.

            # Set the 'train_subset', 'val_subset', 'test_subset' attributes
            setattr(instance, f"{name}_subset", subset_to_assign)

        instance.subsets = {}
        for name, ds in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
            cfg = getattr(instance, f"{name}_subset")
            instance.subsets[cfg.subset_name] = ds

        # Derive normalization params from the CPU pipeline's Normalize transform if available.
        if getattr(instance.train_subset, "augmentations_cpu", None):
            _cpu_pipeline = CPUAugmentationPipeline.from_config(instance.train_subset)
            instance.input_mean = _cpu_pipeline.mean
            instance.input_std = _cpu_pipeline.std
        else:
            instance.input_mean = None
            instance.input_std = None

        # Propagate intensity config from train subset.
        instance.input_intensity_config = getattr(instance.train_subset, "intensity", None)

        # Save hyperparameters
        instance.save_hyperparameters(
            ignore=["subsets", "input_size"],
            logger=False,
        )

        return instance

    def get_default_subset_configs(self, input_size: tuple[int, int] | None = None) -> dict[str, SubsetConfig]:
        """Create a default SubsetConfig for a given subset when not provided.

        This method loads the configuration from the base YAML files in
        getitune/recipe/_base_/data based on the task type.

        Args:
            input_size: input size of the image to set in subset configs.

        Returns:
            dict[str, SubsetConfig]: Default configuration for the subsets loaded from base config.
        """
        # Map task type to config file name
        task_to_data_config_file = {
            TaskType.MULTI_CLASS_CLS: "classification.yaml",
            TaskType.MULTI_LABEL_CLS: "classification.yaml",
            TaskType.DETECTION: "detection.yaml",
            TaskType.INSTANCE_SEGMENTATION: "instance_segmentation.yaml",
            TaskType.SEMANTIC_SEGMENTATION: "semantic_segmentation.yaml",
            TaskType.KEYPOINT_DETECTION: "keypoint_detection.yaml",
        }

        config_file = task_to_data_config_file.get(self.task)
        if config_file is None:
            msg = f"No base config file found for task type: {self.task}"
            raise ValueError(msg)

        # Load the YAML configuration
        base_path = Path(__file__).parent.parent / "recipe" / "_base_" / "data" / config_file
        if not base_path.exists():
            msg = f"Base config file not found: {base_path}"
            raise FileNotFoundError(msg)

        # Load and parse the YAML
        config_dict = OmegaConf.load(base_path)

        # Get the subset configuration
        subset_dicts = {}
        for subset_key in ["train_subset", "val_subset", "test_subset"]:
            if subset_key not in config_dict:
                msg = f"Subset '{subset_key}' not found in config file {config_file}"
                raise ValueError(msg)

            # Extract subset config and convert to container (dict)
            subset_config_dict = OmegaConf.to_container(config_dict[subset_key], resolve=True)  # type: ignore[index]
            subset_input_size = subset_config_dict.get("input_size")
            if subset_input_size is None and input_size is not None:
                subset_config_dict["input_size"] = input_size
            elif subset_input_size is None and input_size is None:
                subset_config_dict["input_size"] = config_dict.get("input_size")

            if subset_config_dict["input_size"] is None:
                msg = "input size is not specified in both the config file and the DataModule constructor."
                raise ValueError(msg)
            subset_config_dict = SubsetConfig(**subset_config_dict)
            subset_dicts[subset_key] = subset_config_dict
        return subset_dicts

    def _is_meta_info_valid(self, label_infos: list[LabelInfo]) -> bool:
        """Check whether there are mismatches in the metainfo for the all subsets."""
        return bool(all(label_info == label_infos[0] for label_info in label_infos))

    def _get_dataset(self, subset: str) -> VisionDataset:
        if (dataset := self.subsets.get(subset)) is None:
            msg = f"Dataset has no '{subset}'. Available subsets = {list(self.subsets.keys())}"
            raise KeyError(msg)
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        config = self.train_subset
        dataset = self._get_dataset(config.subset_name)
        sampler = instantiate_sampler(config.sampler, dataset=dataset, batch_size=config.batch_size)

        num_workers = config.num_workers

        # Pre-populate cached augmentation caches (CachedMosaic, CachedMixUp)
        # before workers start, so all workers inherit full, diverse caches.
        if num_workers > 0 and hasattr(dataset.transforms, "prepare"):
            dataset.transforms.prepare(dataset)

        common_args = {
            "dataset": dataset,
            "batch_size": config.batch_size,
            "num_workers": num_workers,
            "pin_memory": self._pin_memory,
            "collate_fn": dataset.collate_fn,
            "persistent_workers": num_workers > 0,
            "sampler": sampler,
            "shuffle": sampler is None,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "multiprocessing_context": _MP_CONTEXT if num_workers > 0 else None,
        }

        tile_config = self.tile_config
        if tile_config.enable_tiler and tile_config.sampling_ratio < 1:
            num_samples = max(1, int(len(dataset) * tile_config.sampling_ratio))
            logger.info(f"Using tiled sampling with {num_samples} samples")
            common_args.update(
                {
                    "shuffle": False,
                    "sampler": RandomSampler(dataset, num_samples=num_samples),
                },
            )
        return DataLoader(**common_args)

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader."""
        config = self.val_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(**self._eval_loader_kwargs(dataset, config))

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        config = self.test_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(**self._eval_loader_kwargs(dataset, config))

    def predict_dataloader(self) -> DataLoader:
        """Get predict dataloader."""
        config = self.test_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(**self._eval_loader_kwargs(dataset, config))

    def _eval_loader_kwargs(self, dataset: VisionDataset, config: SubsetConfig) -> dict:
        """Build ``DataLoader`` keyword arguments for a val/test/predict dataset.

        The batch size, number of workers and pinned-memory behaviour are taken
        directly from the subset config (i.e. the recipe). This keeps evaluation
        loading configurable per recipe rather than hardcoded here.

        For tiled evaluation a single original image expands into *every* grid tile
        (tiles are not filtered by annotation at inference time), so one dataset item
        can become a large nested tile batch. Recipes therefore keep the tiled
        val/test ``batch_size`` small (typically 1) so the collated tile batch stays
        bounded; the model still forwards tiles in groups of
        ``TileConfig.tile_inference_batch_size`` for GPU efficiency.
        """
        num_workers = config.num_workers
        return {
            "dataset": dataset,
            "batch_size": config.batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": self._pin_memory,
            "collate_fn": dataset.collate_fn,
            "persistent_workers": num_workers > 0,
            "multiprocessing_context": _MP_CONTEXT if num_workers > 0 else None,
        }

    @property
    def _pin_memory(self) -> bool:
        """Whether batches should be staged in page-locked ("pinned") host memory.

        For CPU-only consumers it never is (eg OV evaluation), the copy is redundant and the
        locked pages just reduce the memory available to the rest of the process.
        """
        return self.device != DeviceType.cpu

    def setup(self, stage: str) -> None:
        """Setup for each stage."""

    def teardown(self, stage: str) -> None:
        """Teardown for each stage."""
        # clean up after fit or test
        # called on every process in DDP

    @property
    def hparams_initial(self) -> AttributeDict:
        """The collection of hyperparameters saved with `save_hyperparameters()`. It is read-only.

        The reason why we override is that we have some custom resolvers for `DictConfig`.
        Some resolved Python objects has not a primitive type, so that is not loggable without errors.
        Therefore, we need to unresolve it this time.
        """
        hp = super().hparams_initial
        for key, value in hp.items():
            if isinstance(value, DictConfig):
                # It should be unresolved to make it loggable
                hp[key] = OmegaConf.to_container(value, resolve=False)

        return hp

    def __reduce__(self):
        """Re-initialize object when unpickled."""
        return (
            self.__class__,
            (
                self.task,
                self.data_root,
                self.train_subset,
                self.val_subset,
                self.test_subset,
                self.tile_config,
                self.ignore_index,
                self.unannotated_items_ratio,
                self.auto_num_workers,
                self.device,
                self.input_size,
            ),
        )
