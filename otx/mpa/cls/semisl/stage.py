# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv import ConfigDict

from otx.mpa.cls.stage import ClsStage
from otx.mpa.utils.config_utils import update_or_add_custom_hook
from otx.mpa.utils.logger import get_logger

logger = get_logger()


class SemiSLClsStage(ClsStage):
    """Patch config to support semi supervised learning for object Cls"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_data(self, cfg, data_cfg, training, **kwargs):
        """Patch cfg.data."""
        super().configure_data(cfg, data_cfg, training, **kwargs)
        if training:
            if "unlabeled" in cfg.data and cfg.train_type == "SEMISUPERVISED":
                samples_per_gpu = cfg.data.unlabeled.pop("samples_per_gpu", cfg.data.samples_per_gpu)
                workers_per_gpu = cfg.data.unlabeled.pop("workers_per_gpu", cfg.data.workers_per_gpu)
                update_or_add_custom_hook(
                    cfg,
                    ConfigDict(
                        type="UnlabeledDataHook",
                        unlabeled_data_cfg=cfg.data.unlabeled,
                        samples_per_gpu=samples_per_gpu,
                        workers_per_gpu=workers_per_gpu,
                        model_task=cfg.model_task,
                        seed=cfg.seed,
                        persistent_workers=True if workers_per_gpu > 0 else False,
                    ),
                )
