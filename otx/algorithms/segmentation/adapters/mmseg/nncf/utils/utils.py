# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmseg.datasets import build_dataloader as mmseg_build_dataloader, build_dataset


def build_dataloader(config, subset, distributed):
    loader_cfg = dict(
        samples_per_gpu=config.data.get("samples_per_gpu", 1),
        workers_per_gpu=config.data.get("workers_per_gpu", 0),
        num_gpus=len(config.gpu_ids),
        dist=distributed,
        seed=config.seed
    )
    if subset == "train":
        default_args = dict(test_mode=False)
    else:
        default_args = dict(test_mode=True)
        loader_cfg["shuffle"] = False
        loader_cfg["samples_per_gpu"] = 1

    dataset = build_dataset(config.data.get(subset), default_args)

    loader_cfg = {**loader_cfg, **config.data.get(f'{subset}_dataloader', {})}
    dataloader = mmseg_build_dataloader(
        dataset,
        **loader_cfg,
    )
    return dataloader
