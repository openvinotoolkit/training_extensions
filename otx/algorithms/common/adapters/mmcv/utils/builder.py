"""MMCV general build functions."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import TYPE_CHECKING, Callable, Literal, Union, overload

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from otx.api.utils.argument_checks import check_input_parameters_type


@check_input_parameters_type()
def build_dataset(
    config: Config,
    subset: str,
    dataset_builder: Callable,
    *,
    consume: bool = False,
) -> Dataset:
    """Build dataset."""

    if subset in ["test", "val"]:
        default_args = dict(test_mode=True)
    else:
        default_args = dict(test_mode=False)

    dataset_cfg = config.data.pop(subset) if consume else config.data.get(subset)
    dataset = dataset_builder(dataset_cfg, default_args)
    return dataset


@check_input_parameters_type()
def build_dataloader(
    dataset,
    config: Config,
    subset: str,
    dataloader_builder: Callable,
    *,
    distributed: bool = False,
    consume: bool = False,
    **kwargs,
) -> DataLoader:
    """Build dataloader."""

    #  samples_per_gpu = config.data.get("samples_per_gpu", 1)
    #  if subset in ["test", "val"]:
    #      samples_per_gpu = 1

    loader_cfg = dict(
        samples_per_gpu=config.data.get("samples_per_gpu", 1),
        workers_per_gpu=config.data.get("workers_per_gpu", 0),
        num_gpus=len(config.gpu_ids),
        dist=distributed,
        seed=config.get("seed", None),
        shuffle=False if subset in ["test", "val"] else True,  # pylint: disable=simplifiable-if-expression
    )

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
        }
    )

    specific_loader_cfg = (
        config.data.pop(f"{subset}_dataloader", {}) if consume else config.data.get(f"{subset}_dataloader", {})
    )
    loader_cfg = {**loader_cfg, **specific_loader_cfg, **kwargs}

    dataloader = dataloader_builder(
        dataset,
        **loader_cfg,
    )
    return dataloader


if TYPE_CHECKING:

    @overload
    def build_data_parallel(
        model: torch.nn.Module,
        config: Config,
        *,
        distributed: Literal[True],
    ) -> MMDistributedDataParallel:
        ...

    @overload
    def build_data_parallel(
        model: torch.nn.Module,
        config: Config,
        *,
        distributed: Literal[False] = False,
    ) -> MMDataParallel:
        ...

    @overload
    def build_data_parallel(
        model: torch.nn.Module,
        config: Config,
        *,
        distributed: bool,
    ) -> Union[MMDataParallel, MMDistributedDataParallel]:
        ...


@check_input_parameters_type()
def build_data_parallel(
    model: torch.nn.Module,
    config: Config,
    *,
    distributed: bool = False,
) -> Union[MMDataParallel, MMDistributedDataParallel]:
    """Prepare model for execution.

    Return model import ast, MMDataParallel or MMDataCPU.

    :param model: Model.
    :param config: config.
    :param distributed: Enable distributed training mode.
    :return:
    """
    if torch.cuda.is_available():
        if distributed:
            model = model.cuda()
            # put model on gpus
            find_unused_parameters = config.get("find_unused_parameters", False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model = model.cuda(config.gpu_ids[0])
            model = MMDataParallel(model, device_ids=config.gpu_ids)
    else:
        model = MMDataParallel(model, device_ids=[-1])
    return model
