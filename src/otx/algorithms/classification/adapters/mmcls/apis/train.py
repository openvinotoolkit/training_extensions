"""Train function for classification task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from mmcls.core import DistEvalHook, DistOptimizerHook, EvalHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger, wrap_distributed_model, wrap_non_distributed_model
from mmcv.runner import DistSamplerSeedHook, build_optimizer, build_runner

from otx.algorithms.common.adapters.mmcv.utils import HPUDataParallel, XPUDataParallel
from otx.algorithms.common.adapters.mmcv.utils.hpu_optimizers import HABANA_OPTIMIZERS


def train_model(model, dataset, cfg, distributed=False, validate=False, timestamp=None, device=None, meta=None):
    """Train a model.

    This method will build dataloaders, wrap the model and build a runner
    according to the provided config.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        dataset (:obj:`mmcls.datasets.BaseDataset` | List[BaseDataset]):
            The dataset used to train the model. It can be a single dataset,
            or a list of dataset with the same length as workflow.
        cfg (:obj:`mmcv.utils.Config`): The configs of the experiment.
        distributed (bool): Whether to train the model in a distributed
            environment. Defaults to False.
        validate (bool): Whether to do validation with
            :obj:`mmcv.runner.EvalHook`. Defaults to False.
        timestamp (str, optional): The timestamp string to auto generate the
            name of log files. Defaults to None.
        device (str, optional): TODO
        meta (dict, optional): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
    """
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=cfg.ipu_replicas if device == "ipu" else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get("seed"),
        sampler_cfg=cfg.get("sampler", None),
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

    fp16_cfg = cfg.get("fp16_", None)
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = wrap_distributed_model(
            model, cfg.device, broadcast_buffers=False, find_unused_parameters=find_unused_parameters
        )
    elif cfg.device == "xpu":
        assert len(cfg.gpu_ids) == 1
        model.to(f"xpu:{cfg.gpu_ids[0]}")
        model = XPUDataParallel(model, dim=0, device_ids=cfg.gpu_ids)
    elif cfg.device == "hpu":
        assert len(cfg.gpu_ids) == 1
        model = HPUDataParallel(model.cuda(), dim=0, device_ids=cfg.gpu_ids, enable_autocast=bool(fp16_cfg))
    else:
        model = wrap_non_distributed_model(model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    if cfg.device == "hpu":
        if (new_type := "Fused" + cfg.optimizer.get("type", "SGD")) in HABANA_OPTIMIZERS:
            cfg.optimizer["type"] = new_type

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
    runner.timestamp = timestamp

    if fp16_cfg is None and distributed and "type" not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
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
    if distributed and cfg.runner["type"] == "EpochBasedRunner":
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            "shuffle": False,  # Not shuffle by default
            "sampler_cfg": None,  # Not use sampler by default
            "drop_last": False,  # Not drop last by default
            **cfg.data.get("val_dataloader", {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook if distributed else EvalHook
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority="LOW")

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
