# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import os.path as osp
import time
import warnings

import mmcv
import torch.distributed as dist
from mmcls import __version__
from mmcls.core import DistOptimizerHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    Fp16OptimizerHook,
    build_optimizer,
    build_runner,
)
from torch import nn

from otx.mpa.cls.stage import ClsStage
from otx.mpa.modules.datasets.composed_dataloader import ComposedDL
from otx.mpa.modules.hooks.eval_hook import CustomEvalHook, DistCustomEvalHook
from otx.mpa.modules.hooks.fp16_sam_optimizer_hook import Fp16SAMOptimizerHook
from otx.mpa.registry import STAGES
from otx.mpa.stage import Stage
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class ClsTrainer(ClsStage):
    # noqa: C901
    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run training stage"""
        self._init_logger()
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=True, **kwargs)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Environment
        env_info_dict = collect_env()
        env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        dash_line = "-" * 60 + "\n"
        logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

        # Data
        datasets = [build_dataset(cfg.data.train)]

        # Dataset for HPO
        hp_config = kwargs.get("hp_config", None)
        if hp_config is not None:
            import hpopt

            if isinstance(datasets[0], list):
                for idx, ds in enumerate(datasets[0]):
                    datasets[0][idx] = hpopt.createHpoDataset(ds, hp_config)
            else:
                datasets[0] = hpopt.createHpoDataset(datasets[0], hp_config)

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
            else:
                cfg.checkpoint_config.meta["CLASSES"] = repr_ds.CLASSES
            if "task_adapt" in cfg:
                if hasattr(self, "model_tasks"):  # for incremnetal learning
                    cfg.checkpoint_config.meta.update({"tasks": self.model_tasks})
                    # instead of update(self.old_tasks), update using "self.model_tasks"
                else:
                    cfg.checkpoint_config.meta.update({"CLASSES": self.model_classes})

        # Save config
        # cfg.dump(osp.join(cfg.work_dir, 'config.yaml')) # FIXME bug to save
        # logger.info(f'Config:\n{cfg.pretty_text}')

        # model
        model = build_classifier(cfg.model)

        if self.distributed:
            self._modify_cfg_for_distributed(model, cfg)

        # prepare data loaders
        datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
        train_data_cfg = Stage.get_data_cfg(cfg, "train")
        otx_dataset = train_data_cfg.get("otx_dataset", None)
        drop_last = False
        dataset_len = len(otx_dataset) if otx_dataset else 0
        # if task == h-label & dataset size is bigger than batch size
        num_gpus = dist.get_world_size() if self.distributed else 1
        if (
            train_data_cfg.get("hierarchical_info", None)
            and dataset_len > cfg.data.get("samples_per_gpu", 2) * num_gpus
        ):
            drop_last = True
        # updated to adapt list of dataset for the 'train'
        data_loaders = [
            build_dataloader(
                datasets[0],
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=self.distributed,
                round_up=True,
                seed=cfg.seed,
                drop_last=drop_last,
                persistent_workers=False,
            )
        ]

        # put model on gpus
        model = self._put_model_on_gpu(model, cfg)

        # build runner
        optimizer = build_optimizer(model, cfg.optimizer)

        if cfg.get("runner") is None:
            cfg.runner = {"type": "EpochBasedRunner", "max_epochs": cfg.total_epochs}
            warnings.warn(
                "config is now expected to have a `runner` section, " "please set `runner` in your config.", UserWarning
            )

        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model, batch_processor=None, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
            ),
        )

        # an ugly walkaround to make the .log and .log.json filenames the same
        runner.timestamp = f"{timestamp}"

        # fp16 setting
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            if cfg.optimizer_config.get("type", False) == "SAMOptimizerHook":
                opt_hook = Fp16SAMOptimizerHook
            else:
                opt_hook = Fp16OptimizerHook
            cfg.optimizer_config.pop("type")
            optimizer_config = opt_hook(**cfg.optimizer_config, **fp16_cfg, distributed=self.distributed)
        elif self.distributed and "type" not in cfg.optimizer_config:
            optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

        # register hooks
        runner.register_training_hooks(
            cfg.lr_config, optimizer_config, None, cfg.log_config, cfg.get("momentum_config", None)
        )
        if cfg.get("checkpoint_config", False):
            runner.register_hook(ClsTrainer.register_checkpoint_hook(cfg.checkpoint_config))

        if self.distributed:
            runner.register_hook(DistSamplerSeedHook())

        for hook in cfg.get("custom_hooks", ()):
            runner.register_hook_from_cfg(hook)

        validate = True if cfg.data.get("val", None) else False
        # register eval hooks
        if validate:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=self.distributed,
                shuffle=False,
                round_up=True,
                persistent_workers=False,
            )
            eval_cfg = cfg.get("evaluation", {})
            eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
            eval_hook = DistCustomEvalHook if self.distributed else CustomEvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="ABOVE_NORMAL")

        if cfg.get("resume_from", False):
            runner.resume(cfg.resume_from)
        elif cfg.get("load_from", False):
            if self.distributed:
                runner.load_checkpoint(cfg.load_from, map_location=f"cuda:{cfg.gpu_ids[0]}")
            else:
                runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow)

        logger.info(f"called train_worker() distributed={self.distributed}, validate=True")

        # Save outputs
        output_ckpt_path = osp.join(
            cfg.work_dir, "best_model.pth" if osp.exists(osp.join(cfg.work_dir, "best_model.pth")) else "latest.pth"
        )
        return dict(final_ckpt=output_ckpt_path)

    def _modify_cfg_for_distributed(self, model, cfg):
        nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.dist_params.get("linear_scale_lr", False):
            new_lr = len(cfg.gpu_ids) * cfg.optimizer.lr
            logger.info(
                f"enabled linear scaling rule to the learning rate. \
                changed LR from {cfg.optimizer.lr} to {new_lr}"
            )
            cfg.optimizer.lr = new_lr

    @staticmethod
    def register_checkpoint_hook(checkpoint_config):
        if checkpoint_config.get("type", False):
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            checkpoint_config.setdefault("type", "CheckpointHook")
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        return hook
