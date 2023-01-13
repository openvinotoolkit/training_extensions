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
                update_or_add_custom_hook(
                    cfg,
                    ConfigDict(
                        type="UnlabeledDataHook",
                        unlabeled_data_cfg=cfg.data.unlabeled,
                        samples_per_gpu=cfg.data.unlabeled.pop("samples_per_gpu", cfg.data.samples_per_gpu),
                        workers_per_gpu=cfg.data.unlabeled.pop("workers_per_gpu", cfg.data.workers_per_gpu),
                        model_task=cfg.model_task,
                        seed=cfg.seed,
                        persistent_workers=False,
                    ),
                )

    # def configure_task(self, cfg, training, **kwargs):
    #     """Patch config to support training algorithm."""
    #     logger.info(f"Semi-SL task config!!!!: training={training}")
    #     if "task_adapt" in cfg:
    #         self.task_adapt_type = cfg["task_adapt"].get("type", None)
    #         self.task_adapt_op = cfg["task_adapt"].get("op", "REPLACE")
    #         self.configure_classes(cfg)

    #         if self.data_classes != self.model_classes:
    #             self.configure_task_data_pipeline(cfg)
    #         # TODO[JAEGUK]: configure_anchor is not working
    #         if cfg["task_adapt"].get("use_mpa_anchor", False):
    #             self.configure_anchor(cfg)
    #         if self.task_adapt_type == "mpa":
    #             self.configure_bbox_head(cfg)
    #             self.configure_val_interval(cfg)
    #         else:
    #             src_data_cfg = self.get_data_cfg(cfg, "train")
    #             src_data_cfg.pop("old_new_indices", None)
