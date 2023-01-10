# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import ConfigDict

from otx.mpa.seg.stage import SegStage
from otx.mpa.utils.config_utils import remove_custom_hook, update_or_add_custom_hook
from otx.mpa.utils.logger import get_logger

logger = get_logger()


class SemiSLSegStage(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation"""
        super().configure_task(cfg, training, **kwargs)

        # Set unlabeled data hook
        if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
            update_or_add_custom_hook(
                cfg,
                ConfigDict(
                    type="UnlabeledDataHook",
                    unlabeled_data_cfg=cfg.data.unlabeled,
                    samples_per_gpu=cfg.data.unlabeled.pop("samples_per_gpu", cfg.data.samples_per_gpu),
                    workers_per_gpu=cfg.data.unlabeled.pop("workers_per_gpu", cfg.data.workers_per_gpu),
                    model_task=cfg.model_task,
                    seed=cfg.seed,
                ),
            )

        # Don't pass task_adapt arg to semi-segmentor
        if cfg.model.type != "ClassIncrSegmentor" and cfg.model.get("task_adapt", False):
            cfg.model.pop("task_adapt")

        # Remove task adapt hook (set default torch random sampler)
        remove_custom_hook(cfg, "TaskAdaptHook")
