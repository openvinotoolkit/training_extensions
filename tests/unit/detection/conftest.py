from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omegaconf import DictConfig
from otx.core.config.data import (
    DataModuleConfig,
    SubsetConfig,
)
from otx.core.data.module import OTXDataModule
from otx.core.types.task import OTXTaskType
from otx.core.utils.config import mmconfig_dict_to_dict

if TYPE_CHECKING:
    from mmengine.config import Config as MMConfig


@pytest.fixture()
def fxt_mmcv_det_transform_config(fxt_rtmdet_tiny_config: MMConfig) -> list[DictConfig]:
    return [
        DictConfig(cfg)
        for cfg in mmconfig_dict_to_dict(fxt_rtmdet_tiny_config.train_pipeline)
    ]


@pytest.fixture()
def fxt_datamodule(fxt_asset_dir, fxt_mmcv_det_transform_config) -> OTXDataModule:
    data_root = fxt_asset_dir / "car_tree_bug"

    batch_size = 8
    num_workers = 0
    config = DataModuleConfig(
        format="coco_instances",
        data_root=data_root,
        subsets={
            "train": SubsetConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                transform_lib_type="MMDET",
                transforms=fxt_mmcv_det_transform_config,
            ),
            "val": SubsetConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                transform_lib_type="MMDET",
                transforms=fxt_mmcv_det_transform_config,
            ),
        },
    )
    datamodule = OTXDataModule(
        task=OTXTaskType.DETECTION,
        config=config,
    )
    datamodule.prepare_data()
    return datamodule
