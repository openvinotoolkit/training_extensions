"""MMEngine general build functions."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Callable

from mmengine.config import Config
from torch.utils.data import DataLoader, Dataset

from ._builder_build_data_parallel import build_data_parallel

__all__ = ["build_data_parallel"]


def build_dataset(
    config: Config,
    subset: str,
    dataset_builder: Callable,
    *,
    consume: bool = False,
) -> Dataset:
    """Build dataset."""

    default_args = {"test_mode": True} if subset in ["test", "val"] else {"test_mode": False}

    dataset_cfg = config.data.pop(subset) if consume else config.data.get(subset)
    dataset = dataset_builder(dataset_cfg, default_args)
    return dataset


def build_dataloader(
    dataset: Dataset,
    config: Config,
    subset: str,
    dataloader_builder: Callable,
    *,
    distributed: bool = False,
    consume: bool = False,
    **kwargs,
) -> DataLoader:
    """Build dataloader."""

    loader_cfg = {
        "samples_per_gpu": config.data.get("samples_per_gpu", 1),
        "workers_per_gpu": config.data.get("workers_per_gpu", 0),
        "num_gpus": len(config.gpu_ids),
        "dist": distributed,
        "seed": config.get("seed", None),
        "shuffle": not subset in ["test", "val"],
    }

    # The overall dataloader settings
    loader_cfg.update(
        {
            k: v
            for k, v in config.data.items()
            if k
            not in [
                "train",
                "val",
                "test",
                "unlabeled",
                "train_dataloader",
                "val_dataloader",
                "test_dataloader",
                "unlabeled_dataloader",
            ]
        },
    )

    specific_loader_cfg = (
        config.data.pop(f"{subset}_dataloader", {}) if consume else config.data.get(f"{subset}_dataloader", {})
    )
    loader_cfg = Config(cfg_dict={**loader_cfg, **specific_loader_cfg, **kwargs})

    dataloader = dataloader_builder(
        dataset,
        **loader_cfg,
    )
    return dataloader
