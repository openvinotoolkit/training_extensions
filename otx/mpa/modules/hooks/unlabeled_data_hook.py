# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

from mmcv.runner import HOOKS, Hook, get_dist_info

from otx.mpa.modules.datasets.composed_dataloader import ComposedDL
from otx.mpa.utils.logger import get_logger

logger = get_logger()
TASK_LIB_NAME = {"classification": "mmcls", "detection": "mmdet", "segmentation": "mmseg"}


@HOOKS.register_module()
class UnlabeledDataHook(Hook):
    def __init__(self, unlabeled_data_cfg, samples_per_gpu, workers_per_gpu, model_task, seed=None, **kwargs):
        super().__init__()

        # Build unlabeled dataset & loader
        task_lib_module = importlib.import_module(f"{TASK_LIB_NAME[model_task]}.datasets")
        build_dataset = getattr(task_lib_module, "build_dataset")
        build_dataloader = getattr(task_lib_module, "build_dataloader")

        self.unlabeled_dataset = build_dataset(unlabeled_data_cfg)

        _, world_size = get_dist_info()

        logger.info("In UnlabeledDataHook, creating unlabeled data_loader...")
        self.unlabeled_loader = build_dataloader(
            self.unlabeled_dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=world_size,
            dist=(world_size > 1),
            seed=seed,
            **kwargs,
        )
        self.composed_loader = None

    def before_epoch(self, runner):
        if self.composed_loader is None:
            logger.info(
                "In UnlabeledDataHook.before_epoch, creating ComposedDL"
                f"([labeled({len(runner.data_loader.dataset)}, unlabeled({len(self.unlabeled_loader.dataset)})])"
            )
            self.composed_loader = ComposedDL([runner.data_loader, self.unlabeled_loader])
        runner.data_loader = self.composed_loader
