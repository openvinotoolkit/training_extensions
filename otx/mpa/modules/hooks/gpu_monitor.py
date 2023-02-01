# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.runner import HOOKS, Hook

from otx.mpa.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class GPUMonitorHook(Hook):
    def __init__(self, **kwargs):
        super(GPUMonitorHook, self).__init__()
        self.available = True
        if torch.cuda.is_available():
            self.gpu_ids = kwargs.get("gpu_ids", [0])
            for id in self.gpu_ids:
                total_memory_gb = torch.cuda.get_device_properties(id).total_memory / 2**30
                logger.info(f"\t== total memory for GPU {id}: {total_memory_gb} GB")
        else:
            logger.warning("No available CUDA device")
            self.available = False

    def after_epoch(self, runner):
        if self.available:
            for id in self.gpu_ids:
                logger.info(f"\t== GPU {(id)} memory states ==")
                logger.info(
                    f"\t* reserved : {torch.cuda.memory_reserved(id)/2**20:.2f}/\
                    {torch.cuda.max_memory_reserved(id)/2**20:.2f}"
                )
                logger.info(
                    f"\t* allocated: {torch.cuda.memory_allocated(id)/2**20:.2f}/\
                    {torch.cuda.max_memory_allocated(id)/2**20:.2f}"
                )
