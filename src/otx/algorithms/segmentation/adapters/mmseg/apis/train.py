"""Train function for segmentation task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import mmcv
import torch
from mmcv.runner import HOOKS, DistSamplerSeedHook, EpochBasedRunner, build_runner
from mmcv.utils import build_from_cfg
from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook, build_optimizer
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import build_ddp, find_latest_checkpoint, get_root_logger
from mmseg.utils.util_distribution import build_dp, dp_factory

from otx.algorithms.common.adapters.mmcv.utils import HPUDataParallel, XPUDataParallel
from otx.algorithms.common.adapters.mmcv.utils.hpu_optimizers import HABANA_OPTIMIZERS

dp_factory["xpu"] = XPUDataParallel
dp_factory["hpu"] = HPUDataParallel


def train_segmentor(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True,
    )
    # The overall dataloader settings
    loader_cfg.update(
        {
            k: v
            for k, v in cfg.data.items()
            if k not in ["train", "val", "test", "train_dataloader", "val_dataloader", "test_dataloader"]
        }
    )

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get("train_dataloader", {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    if cfg.device == "xpu":
        model.to(f"xpu:{cfg.gpu_ids[0]}")

    # put model on devices
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # DDP wrapper
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        if not torch.cuda.is_available():  # noqa
            assert digit_version(mmcv.__version__) >= digit_version(
                "1.4.4"
            ), "Please use MMCV >= 1.4.4 for CPU training!"

        if cfg.device == "hpu":
            use_autocast = bool(cfg.get("fp16_", False))
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids, enable_autocast=use_autocast)
            model.to(model.src_device_obj)
        else:
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    if cfg.device == "hpu":
        optim_type = cfg.optimizer.get("type", "SGD")
        if optim_type == "Adam":  # to avoid segmentation fault
            optim_type = "AdamW"
            cfg.optimizer.type = optim_type
        if (new_type := "Fused" + optim_type) in HABANA_OPTIMIZERS:
            cfg.optimizer["type"] = new_type

    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.device == "xpu":
        if cfg.optimizer_config.get("bf16_training", False):
            logger.warning("XPU supports fp32 training only currently.")
        model.train()

    if "bf16_training" in cfg.optimizer_config:
        # Remove unused parameters in runner
        cfg.optimizer_config.pop("bf16_training")

    if cfg.get("runner") is None:
        cfg.runner = {"type": "IterBasedRunner", "max_iters": cfg.total_iters}
        warnings.warn(
            "config is now expected to have a `runner` section, " "please set `runner` in your config.", UserWarning
        )

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model, batch_processor=None, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
        ),
    )

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config, cfg.optimizer_config, cfg.checkpoint_config, cfg.log_config, cfg.get("momentum_config", None)
    )
    if distributed:
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            "samples_per_gpu": 1,
            "shuffle": False,  # Not shuffle by default
            **cfg.data.get("val_dataloader", {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    # user-defined hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got " f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
