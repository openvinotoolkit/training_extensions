# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.mmcv.utils.config_utils import remove_custom_hook
from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.seg.stage import SegStage

logger = get_logger()


class SemiSLSegStage(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_data(self, cfg, training, data_cfg, **kwargs):
        """Patch cfg.data."""
        super().configure_data(cfg, training, data_cfg, **kwargs)
        # Set unlabeled data hook
        if training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                self.configure_unlabeled_dataloader(cfg, self.distributed)

    def configure_task(self, cfg, training, **kwargs):
        """Adjust settings for task adaptation"""
        super().configure_task(cfg, training, **kwargs)

        # Don't pass task_adapt arg to semi-segmentor
        if cfg.model.type != "ClassIncrEncoderDecoder" and cfg.model.get("task_adapt", False):
            cfg.model.pop("task_adapt")

        # Remove task adapt hook (set default torch random sampler)
        remove_custom_hook(cfg, "TaskAdaptHook")
