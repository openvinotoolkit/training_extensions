"""OTX Anomaly Datamodules."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from otx.core.config.data import DataModuleConfig, SubsetConfig, TilerConfig
from otx.core.data import OTXDataModule
from otx.core.types.task import OTXTaskType
from otx.core.types.transformer_libs import TransformLibType


class AnomalyDataModule(OTXDataModule):
    """Anomaly DataModule."""

    def __init__(
        self,
        task_type: OTXTaskType,
        data_dir: str,
        data_format: str = "mvtec",
        # Train args.
        train_batch_size: int = 32,
        train_num_workers: int = 8,
        train_transforms: list[dict[str, Any]] | None = None,
        train_transform_lib_type: TransformLibType = TransformLibType.TORCHVISION,
        # Validation args.
        val_batch_size: int = 32,
        val_num_workers: int = 8,
        val_transforms: list[dict[str, Any]] | None = None,
        val_transform_lib_type: TransformLibType = TransformLibType.TORCHVISION,
        # Test args.
        test_batch_size: int = 32,
        test_num_workers: int = 8,
        test_transforms: list[dict[str, Any]] | None = None,
        test_transform_lib_type: TransformLibType = TransformLibType.TORCHVISION,
        # Tiler args.
        enable_tiler: bool = False,
        grid_size: tuple[int, int] = (2, 2),
        overlap: float = 0.0,
    ) -> None:
        # Create the train subset.
        train_subset_config = SubsetConfig(
            batch_size=train_batch_size,
            subset_name="train",
            transforms=train_transforms,  # type: ignore[arg-type]
            transform_lib_type=train_transform_lib_type,
            num_workers=train_num_workers,
        )
        # Create the validation subset.
        val_subset_config = SubsetConfig(
            batch_size=val_batch_size,
            subset_name="test",  # use test as validation
            transforms=val_transforms,  # type: ignore[arg-type]
            transform_lib_type=val_transform_lib_type,
            num_workers=val_num_workers,
        )

        # Create the test subset.
        test_subset_config = SubsetConfig(
            batch_size=test_batch_size,
            subset_name="test",
            transforms=test_transforms,  # type: ignore[arg-type]
            transform_lib_type=test_transform_lib_type,
            num_workers=test_num_workers,
        )

        # Create the tiler config.
        tiler_config = TilerConfig(
            enable_tiler=enable_tiler,
            grid_size=grid_size,
            overlap=overlap,
        )

        # Create the datamodule config.
        datamodule_config = DataModuleConfig(
            data_format=data_format,
            data_root=data_dir,
            train_subset=train_subset_config,
            val_subset=val_subset_config,
            test_subset=test_subset_config,
            tile_config=tiler_config,
        )
        super().__init__(task=task_type, config=datamodule_config)
