import copy
from typing import Dict, List, Optional, Union

import torch
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig as Config
from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.importing import get_non_default_args
from otx.v2.api.utils.logger import get_logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mmengine.hooks import Hook
from mmengine.runner import Runner

logger = get_logger()
MMENGINE_DTYPE = ("float16", "bfloat16", "float32", "float64")


class MMXEngine(Engine):
    def __init__(
        self,
        config: Optional[Union[Dict, Config, str]] = None,
    ) -> None:
        super().__init__()
        self.runner = None
        self.module_registry = MMEngineRegistry()
        self.base_runner = self.module_registry.get("Runner")
        self.initial_config(config)

    def initial_config(self, config: Optional[Union[Dict, Config, str]]):
        if config is not None:
            if isinstance(config, str):
                self.config = Config.fromfile(config)
            elif isinstance(config, Config):
                self.config = copy.deepcopy(config)
            elif isinstance(config, dict):
                self.config = Config(config)
        else:
            self.config = Config(dict())

    def update_config(
        self,
        func_args: Dict,
        **kwargs,
    ):
        update_check = not all(value is None for value in func_args.values()) or not all(
            value is None for value in kwargs.values()
        )

        # Update Model & Dataloaders & Custom hooks
        model = func_args.get("model", None)
        train_dataloader = func_args.get("train_dataloader", None)
        val_dataloader = func_args.get("val_dataloader", None)
        test_dataloader = func_args.get("test_dataloader", None)
        custom_hooks = func_args.get("custom_hooks", None)
        if model is not None:
            kwargs["model"] = model
        if train_dataloader is not None:
            kwargs["train_dataloader"] = train_dataloader
        if val_dataloader is not None:
            kwargs["val_dataloader"] = val_dataloader
        if test_dataloader is not None:
            kwargs["test_dataloader"] = test_dataloader
        if custom_hooks is not None:
            kwargs["custom_hooks"] = custom_hooks

        # Update train_cfg
        max_iters = func_args.get("max_iters", None)
        max_epochs = func_args.get("max_epochs", None)
        precision = func_args.get("precision", None)
        eval_interval = func_args.get("eval_interval", 1)
        if max_iters is not None and max_epochs is not None:
            raise ValueError("Only one of `max_epochs` or `max_iters` can be set.")
        if "train_cfg" not in kwargs or kwargs["train_cfg"] is None:
            eval_interval = eval_interval if eval_interval is not None else 1
            kwargs["train_cfg"] = dict(val_interval=eval_interval)
        if max_epochs is not None:
            kwargs["train_cfg"]["by_epoch"] = True
            kwargs["train_cfg"]["max_epochs"] = max_epochs
        elif max_iters is not None:
            kwargs["train_cfg"]["by_epoch"] = False
            kwargs["train_cfg"]["max_iters"] = max_iters

        # Update Optimizer
        if "optim_wrapper" not in kwargs or kwargs["optim_wrapper"] is None:
            optimizer = func_args.get("optimizer", None)
            if optimizer is None:
                # FIXME: Remove default setting here
                optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
            kwargs["optim_wrapper"] = dict(type="AmpOptimWrapper", dtype=precision, optimizer=optimizer)

        # Update val_cfg (ValLoop)
        if val_dataloader is not None:
            if "val_cfg" not in kwargs or kwargs["val_cfg"] is None:
                kwargs["val_cfg"] = dict()
            if precision == "float16":
                kwargs["val_cfg"]["fp16"] = True

            # Update val_evaluator
            if "val_evaluator" not in kwargs or kwargs["val_evaluator"] is None:
                # TODO: Need to set val_evaluator as task-agnostic way
                kwargs["val_evaluator"] = dict(type="mmpretrain.Accuracy")

        # Update test_cfg (TestLoop)
        if test_dataloader is not None:
            if "test_cfg" not in kwargs or kwargs["test_cfg"] is None:
                kwargs["test_cfg"] = dict()
            if precision == "float16":
                kwargs["test_cfg"]["fp16"] = True

            # Update test_evaluator
            if "test_evaluator" not in kwargs or kwargs["test_evaluator"] is None:
                # TODO: Need to set test_evaluator as task-agnostic way
                kwargs["test_evaluator"] = self.config.get("val_evaluator", dict(type="mmpretrain.Accuracy"))

        # Update randomness
        seed = func_args.get("seed", None)
        deterministic = func_args.get("deterministic", False)
        if func_args.get("seed", None) is not None:
            kwargs["randomness"] = dict(seed=seed, deterministic=deterministic)

        distributed = func_args.get("distributed", False)
        if "default_hooks" not in kwargs or kwargs["default_hooks"] is None:
            # FIXME: Default hooks need to align
            kwargs["default_hooks"] = dict(
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

        # kwargs -> Update config
        for kwarg_key, kwarg_value in kwargs.items():
            if kwarg_value is None:
                continue
            self.config[kwarg_key] = kwarg_value

        # Check Config Default is not None
        runner_default_args = get_non_default_args(Runner.__init__)
        for not_none_arg, default_value in runner_default_args:
            if self.config.get(not_none_arg) is None:
                self.config[not_none_arg] = default_value

        return update_check

    def train(
        self,
        model: Optional[Union[torch.nn.Module, Dict]] = None,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        optimizer: Optional[Union[dict, Optimizer]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        precision: Optional[str] = None,
        eval_interval: Optional[int] = None,
        custom_hooks: Optional[Union[List, Dict, Hook]] = None,
        **kwargs,
    ):
        train_args = {
            "model": model,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "optimizer": optimizer,
            "max_iters": max_iters,
            "max_epochs": max_epochs,
            "distributed": distributed,
            "seed": seed,
            "deterministic": deterministic,
            "precision": precision,
            "eval_interval": eval_interval,
            "custom_hooks": custom_hooks,
        }

        update_check = self.update_config(func_args=train_args, **kwargs)
        if self.runner is None or update_check:
            self.runner = self.base_runner.from_cfg(self.config)
            self.config = self.runner.cfg

        output_model = self.runner.train()
        return output_model

    def val(
        self,
        model: Optional[Union[torch.nn.Module, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):  # Metric (data_class or dict)
        val_args = {
            "model": model,
            "val_dataloader": val_dataloader,
            "precision": precision,
        }
        update_check = self.update_config(func_args=val_args, **kwargs)
        if self.runner is None:
            self.runner = self.base_runner.from_cfg(self.config)
            self.config = self.runner.cfg
        elif update_check:
            self.runner._val_dataloader = self.config["val_dataloader"]
            self.runner._val_loop = self.config["val_cfg"]
            self.runner._val_evaluator = self.config["val_evaluator"]

        return self.runner.val()

    def test(
        self,
        model: Optional[Union[torch.nn.Module, Dict]] = None,
        test_dataloader: Optional[DataLoader] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):  # Metric (data_class or dict)
        test_args = {
            "model": model,
            "test_dataloader": test_dataloader,
            "precision": precision,
        }
        update_check = self.update_config(func_args=test_args, **kwargs)
        if self.runner is None:
            self.runner = self.base_runner.from_cfg(self.config)
            self.config = self.runner.cfg
        elif update_check:
            self.runner._test_dataloader = self.config["test_dataloader"]
            self.runner._test_loop = self.config["test_cfg"]
            self.runner._test_evaluator = self.config["test_evaluator"]

        return self.runner.test()

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, Dict]] = None,
        dataloader: Optional[Union[DataLoader, Dict]] = None,
        checkpoint: Optional[str] = None,
    ):  # TorchInferencer -> Tensor
        raise NotImplementedError()

    def export(self):  # IR Model (file: xml, bin) return file_path
        raise NotImplementedError()


### DONE ### CLI -> Geti~
"""

"""
