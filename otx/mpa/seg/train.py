# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import Fp16OptimizerHook, build_optimizer, build_runner
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader

# from mmdet.core import DistEvalHook, EvalHook
# from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_params_manager
from mmseg.utils import get_root_logger

from otx.mpa.seg.builder import build_dataset
from otx.mpa.utils.data_cpu import MMDataCPU


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=False,
        )
        for ds in dataset
    ]

    if torch.cuda.is_available():
        if distributed:
            find_unused_parameters = cfg.get("find_unused_parameters", False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        model = MMDataCPU(model)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

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

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        grad_clip = cfg.optimizer_config.get("grad_clip", None)
        cfg.optimizer_config = Fp16OptimizerHook(**fp16_cfg, grad_clip=grad_clip, distributed=distributed)

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config, cfg.optimizer_config, cfg.checkpoint_config, cfg.log_config, cfg.get("momentum_config", None)
    )

    # register parameters manager hook
    params_manager_cfg = cfg.get("params_config", None)
    if params_manager_cfg is not None:
        runner.register_hook(build_params_manager(params_manager_cfg))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    for hook in cfg.get("custom_hooks", ()):
        runner.register_hook_from_cfg(hook)

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
