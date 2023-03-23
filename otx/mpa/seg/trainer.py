# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os
import os.path as osp
import time

from mmcv import get_git_hash
from mmseg import __version__
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.utils import collect_env
from torch import nn

from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.registry import STAGES

from .stage import SegStage

logger = get_logger()


@STAGES.register_module()
class SegTrainer(SegStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage for segmentation

        - Configuration
        - Environment setup
        - Run training via MMSegmentation -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"Supported modes are {self.mode} but '{mode}' is given.")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, **kwargs)
        logger.info("train!")

        # FIXME: what is this? Why do we need?
        if cfg.runner.type == "IterBasedRunner":
            cfg.runner = dict(type=cfg.runner.type, max_iters=cfg.runner.max_iters)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Environment
        logger.info(f"cfg.gpu_ids = {cfg.gpu_ids}, distributed = {self.distributed}")
        env_info_dict = collect_env()
        env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]

        # FIXME: Currently segmentor does not support multi batch evaluation.
        # For the Self-SL case, there is no val data. So, need to check the
        if "val" in cfg.data and "val_dataloader" in cfg.data:
            cfg.data.val_dataloader.samples_per_gpu = 1

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
        else:
            target_classes = datasets[0].CLASSES

        # Metadata
        meta = dict()
        meta["env_info"] = env_info
        meta["seed"] = cfg.seed
        meta["exp_name"] = cfg.work_dir
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(
                mmseg_version=__version__ + get_git_hash()[:7],
                CLASSES=target_classes,
            )

        self.configure_samples_per_gpu(cfg, "train", self.distributed)
        self.configure_fp16_optimizer(cfg, self.distributed)

        # Model
        model_builder = kwargs.get("model_builder", None)
        model = self.build_model(cfg, model_builder)
        model.train()
        model.CLASSES = target_classes

        if self.distributed:
            self._modify_cfg_for_distributed(model, cfg)

        self.configure_compat_cfg(cfg)

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.py'))
        # logger.info(f'Config:\n{cfg.pretty_text}')

        validate = True if cfg.data.get("val", None) else False
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=self.distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta,
        )

        # Save outputs
        output_ckpt_path = os.path.join(cfg.work_dir, "latest.pth")
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, "best_mDice_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        best_ckpt_path = glob.glob(os.path.join(cfg.work_dir, "best_mIoU_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return dict(
            final_ckpt=output_ckpt_path,
        )

    def _modify_cfg_for_distributed(self, model, cfg):
        nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.dist_params.get("linear_scale_lr", False):
            new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
            logger.info(
                f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}"
            )
            cfg.optimizer.lr = new_lr
