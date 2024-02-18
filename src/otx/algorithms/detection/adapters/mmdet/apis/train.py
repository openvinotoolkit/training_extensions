"""Train function for detection task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    OptimizerHook,
    build_runner,
    get_dist_info,
)
from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import build_ddp, compat_cfg, find_latest_checkpoint, get_root_logger
from mmdet.utils.util_distribution import build_dp, dp_factory

from otx.algorithms.common.adapters.mmcv.utils import HPUDataParallel, XPUDataParallel
from otx.algorithms.common.adapters.mmcv.utils.hpu_optimizers import HABANA_OPTIMIZERS

dp_factory["xpu"] = XPUDataParallel
dp_factory["hpu"] = HPUDataParallel


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ("auto_scale_lr" not in cfg) or (not cfg.auto_scale_lr.get("enable", False)):
        logger.info("Automatic scaling of learning rate (LR)" " has been disabled.")
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get("base_batch_size", None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(
        f"Training with {num_gpus} GPU(s) with {samples_per_gpu} "
        f"samples per GPU. The total batch size is {batch_size}."
    )

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info("LR has been automatically scaled " f"from {cfg.optimizer.lr} to {scaled_lr}")
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info(
            "The batch size match the "
            f"base batch size: {base_batch_size}, "
            f"will not scaling the LR ({cfg.optimizer.lr})."
        )


def train_detector(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Trains a detector via mmdet."""

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = "EpochBasedRunner" if "runner" not in cfg else cfg.runner["type"]

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False,
    )

    train_loader_cfg = {**train_dataloader_default_args, **cfg.data.get("train_dataloader", {})}

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    fp16_cfg = cfg.get("fp16_", None)
    # put model on gpus
    if cfg.device == "xpu":
        model.to(f"xpu:{cfg.gpu_ids[0]}")

    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    elif cfg.device == "hpu":
        model = build_dp(
            model, cfg.device, device_ids=cfg.gpu_ids, dim=0, enable_autocast=bool(fp16_cfg), put_gt_on_device=False
        )
        # patch optimizer
        if (new_type := "Fused" + cfg.optimizer.get("type", "SGD")) in HABANA_OPTIMIZERS:
            cfg.optimizer["type"] = new_type
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)

    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.device == "xpu":
        if cfg.optimizer_config.get("bf16_training", False):
            logger.warning("XPU supports fp32 training only currently.")
        dtype = torch.float32
        model.train()
        model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=dtype)

    if "bf16_training" in cfg.optimizer_config:
        # Remove unused parameters in runner
        cfg.optimizer_config.pop("bf16_training")

    runner = build_runner(
        cfg.runner, default_args=dict(model=model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta)
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if fp16_cfg is None and distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get("custom_hooks", None),
    )

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False, persistent_workers=False
        )

        val_dataloader_args = {**val_dataloader_default_args, **cfg.data.get("val_dataloader", {})}
        # Support batch_size > 1 in validation

        if val_dataloader_args["samples_per_gpu"] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    resume_from = None
    if cfg.resume_from is None and cfg.get("auto_resume"):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
