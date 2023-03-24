# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import os.path as osp
import time

from mmcls import __version__
from mmcls.apis import train_model
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import collect_env
from torch import nn

from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.registry import STAGES

from .stage import ClsStage

logger = get_logger()


@STAGES.register_module()
class ClsTrainer(ClsStage):
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):  # noqa: C901
        """Run training stage for classification

        - Configuration
        - Environment setup
        - Run training via MMClassification -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"Supported modes are {self.mode} but '{mode}' is given.")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=True, **kwargs)
        logger.info("train!")

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Environment
        logger.info(f"cfg.gpu_ids = {cfg.gpu_ids}, distributed = {self.distributed}")
        env_info_dict = collect_env()
        env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]

        # Metadata
        meta = dict()
        meta["env_info"] = env_info
        # meta['config'] = cfg.pretty_text
        meta["seed"] = cfg.seed

        repr_ds = datasets[0]

        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(mmcls_version=__version__)
            if hasattr(repr_ds, "tasks"):
                cfg.checkpoint_config.meta["tasks"] = repr_ds.tasks
            if hasattr(repr_ds, "CLASSES"):
                cfg.checkpoint_config.meta["CLASSES"] = repr_ds.CLASSES

        self.configure_samples_per_gpu(cfg, "train", self.distributed)
        self.configure_fp16_optimizer(cfg, self.distributed)

        # Model
        model_builder = kwargs.get("model_builder", None)
        model = self.build_model(cfg, model_builder)
        model.train()

        if self.distributed:
            self._modify_cfg_for_distributed(model, cfg)

        self.configure_compat_cfg(cfg)

        # register custom eval hooks
        validate = True if cfg.data.get("val", None) else False
        if validate:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_loader_cfg = {
                # cfg.gpus will be ignored if distributed
                "num_gpus": len(cfg.gpu_ids),
                "dist": self.distributed,
                "round_up": True,
                "seed": cfg.seed,
                "shuffle": False,  # Not shuffle by default
                "sampler_cfg": None,  # Not use sampler by default
                **cfg.data.get("val_dataloader", {}),
            }
            val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
            eval_cfg = cfg.get("evaluation", {})
            eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
            cfg.custom_hooks.append(
                dict(
                    type="DistCustomEvalHook" if self.distributed else "CustomEvalHook",
                    dataloader=val_dataloader,
                    priority="ABOVE_NORMAL",
                    **eval_cfg,
                )
            )

        train_model(
            model,
            datasets,
            cfg,
            distributed=self.distributed,
            validate=False,
            timestamp=timestamp,
            meta=meta,
        )

        # Save outputs
        output_ckpt_path = osp.join(cfg.work_dir, "latest.pth")
        best_ckpt_path = glob.glob(osp.join(cfg.work_dir, "best_*.pth"))
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
