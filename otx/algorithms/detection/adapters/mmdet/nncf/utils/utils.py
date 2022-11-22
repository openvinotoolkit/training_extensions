# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from mmcv.parallel import MMDataParallel
from mmcv.parallel import MMDistributedDataParallel
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

from otx.algorithms.common.adapters.mmcv.data_cpu import MMDataCPU


def prepare_mmdet_model_for_execution(model, cfg, distributed=False):
    """
    Prepare model for execution.
    Return model MMDistributedDataParallel, MMDataParallel or MMDataCPU.

    :param model: Model.
    :param cfg: training mmdet config.
    :param distributed: Enable distributed training mode.
    :return:
    """
    if torch.cuda.is_available():
        if distributed:
            # put model on gpus
            find_unused_parameters = cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDataCPU(model)
    return model


def build_val_dataloader(cfg, distributed):
    # Support batch_size > 1 in validation
    val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    if val_samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_ImageToTensor(
            cfg.data.val.pipeline)
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    return val_dataloader
