import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.decorators import set_default_argument
from otx.v2.api.utils.logger import get_logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils.dl_utils import collect_env

logger = get_logger()
MMENGINE_DTYPE = ("float16", "bfloat16", "float32", "float64")


class MMXEngine(Engine):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.module_registry = MMEngineRegistry()

    @set_default_argument(Runner.__init__)
    def train(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        work_dir: str,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Union[dict, Optimizer]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = 20,
        distributed: bool = False,
        seed: Optional[int] = None,
        deterministric: bool = False,
        precision: str = "float32",
        # Validation (Dependent on val_dataloader)
        eval_interval: int = 1,
        eval_metric: Optional[Union[str, List[str]]] = ["accuracy", "class_accuracy"],
        # Hooks for mmX
        custom_hooks: Optional[Union[List, Dict, Hook]] = None,
        **kwargs,
    ):
        # Set Logger
        env_info_dict = collect_env()
        "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])

        # Set Runners
        if "train_cfg" not in kwargs or kwargs["train_cfg"] is None:
            if max_iters is not None and max_epochs is not None:
                raise ValueError("Only one of `max_epochs` or `max_iters` can be set.")

            if max_epochs:
                kwargs["train_cfg"] = dict(by_epoch=True, max_epochs=max_epochs, val_interval=eval_interval)
                self._by_epoch = True
            else:
                kwargs["train_cfg"] = dict(by_epoch=False, max_iters=max_iters, val_interval=eval_interval)
                self._by_epoch = False
        if seed is not None:
            kwargs["randomness"] = dict(seed=seed)

        # Model Wrapping
        if precision.upper() in ["FP16", "16"]:
            # TODO: TrainLoop & ValLoop
            pass

        if "optim_wrapper" not in kwargs or kwargs["optim_wrapper"] is None:
            if optimizer is None:
                optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
            kwargs["optim_wrapper"] = dict(type="AmpOptimWrapper", dtype=precision, optimizer=optimizer)

        base_runner = self.module_registry.get("Runner")
        runner = base_runner(
            model=model, work_dir=work_dir, train_dataloader=train_dataloader, val_dataloader=val_dataloader, **kwargs
        )

        # Setting Hooks
        # TODO: default hook -> arguments
        default_hooks = dict(
            # record the time of every iterations.
            timer=dict(type="IterTimerHook"),
            # print log every 100 iterations.
            logger=dict(type="LoggerHook", interval=100),
            # enable the parameter scheduler.
            # TODO: lr_config -> param_scheduler
            param_scheduler=dict(type="ParamSchedulerHook"),
            # save checkpoint per epoch, and automatically save the best checkpoint.
            checkpoint=dict(
                # type='CheckpointHookWithValResults',
                type="CheckpointHook",
                interval=1,
                # out_dir=,
                max_keep_ckpts=1,
                save_best="auto",
            ),
            # set sampler seed in distributed evrionment.
            sampler_seed=dict(type="DistSamplerSeedHook") if distributed else None,
        )
        runner.register_hooks(default_hooks=default_hooks, custom_hooks=custom_hooks)

        output_model = runner.train()
        return output_model

    def infer(self, model: torch.nn.Module, img):
        raise NotImplementedError()

    def evaluate(self, **params):
        raise NotImplementedError()
