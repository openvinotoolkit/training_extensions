"""OTX adapters.torch.mmengine functions related to Runner config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

from mmengine.device import get_device

from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config


def get_value_from_config(
    arg_key: str,
    positional_args: dict,
    config: Config,
    default: Optional[Union[int, str]] = None,
) -> Optional[Union[dict, list]]:
    arg_config = positional_args.get(arg_key, None)
    return config.get(arg_key, default) if arg_config is None else arg_config


def configure_evaluator(
    evaluator: Union[list, dict],
    num_classes: int,
    scope: Optional[str] = None,
) -> Union[list, dict]:
    if isinstance(evaluator, list):
        for metric in evaluator:
            if isinstance(metric, dict):
                if scope is not None:
                    metric["_scope_"] = scope
                if "topk" in metric:
                    metric["topk"] = [1] if num_classes < 5 else [1, 5]
    elif isinstance(evaluator, dict):
        if scope is not None:
            evaluator["_scope_"] = scope
        if "topk" in evaluator:
            evaluator["topk"] = [1] if num_classes < 5 else [1, 5]
    return evaluator


def update_train_config(func_args: dict, config: Config, precision: Optional[str], **kwargs) -> tuple:
    if kwargs.get("train_dataloader", None) is not None:
        max_iters = get_value_from_config("max_iters", func_args, config=config)
        max_epochs = get_value_from_config("max_epochs", func_args, config=config)
        val_interval = get_value_from_config("val_interval", func_args, config=config, default=1)
        if max_iters is not None and max_epochs is not None:
            msg = "Only one of `max_epochs` or `max_iters` can be set."
            raise ValueError(msg)
        if "train_cfg" not in kwargs or kwargs["train_cfg"] is None:
            kwargs["train_cfg"] = {"val_interval": val_interval, "by_epoch": True}
        if max_epochs is not None:
            kwargs["train_cfg"]["by_epoch"] = True
            kwargs["train_cfg"]["max_epochs"] = max_epochs
        elif max_iters is not None:
            kwargs["train_cfg"]["by_epoch"] = False
            kwargs["train_cfg"]["max_iters"] = max_iters
        # Update Optimizer
        if "optim_wrapper" not in kwargs or kwargs["optim_wrapper"] is None:
            optimizer = get_value_from_config("optimizer", func_args, config=config)
            if optimizer is None:
                # FIXME: Remove default setting here
                optimizer = {"type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}
            if get_device() not in ("cuda", "gpu", "npu", "mlu"):
                kwargs["optim_wrapper"] = {"type": "OptimWrapper", "optimizer": optimizer}
            else:
                kwargs["optim_wrapper"] = {"type": "AmpOptimWrapper", "dtype": precision, "optimizer": optimizer}
    elif isinstance(config.get("train_dataloader", None), dict):
        # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
        config["train_dataloader"] = None
        config["train_cfg"] = None
        config["optim_wrapper"] = None
    return config, kwargs


def update_val_test_config(
    func_args: dict,
    config: Config,
    precision: Optional[str],
    num_classes: int,
    scope: Optional[str],
    **kwargs,
) -> tuple:
    if kwargs.get("val_dataloader", None) is not None:
        if "val_cfg" not in kwargs or kwargs["val_cfg"] is None:
            kwargs["val_cfg"] = {}
        if precision in ["float16", "fp16"]:
            kwargs["val_cfg"]["fp16"] = True
        # Update val_evaluator
        val_evaluator = get_value_from_config("val_evaluator", func_args, config=config)
        if val_evaluator is None:
            # FIXME: Need to set val_evaluator as task-agnostic way
            val_evaluator = [{"type": "Accuracy"}]
        kwargs["val_evaluator"] = configure_evaluator(val_evaluator, num_classes=num_classes, scope=scope)
    elif isinstance(config.get("val_dataloader", None), dict):
        # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
        config["val_dataloader"] = None
        config["val_cfg"] = None
        config["val_evaluator"] = None

    if kwargs.get("test_dataloader", None) is not None:
        if "test_cfg" not in kwargs or kwargs["test_cfg"] is None:
            kwargs["test_cfg"] = {}
        if precision in ["float16", "fp16"]:
            kwargs["test_cfg"]["fp16"] = True
        # Update test_evaluator
        test_evaluator = get_value_from_config("test_evaluator", func_args, config=config)
        if test_evaluator is None:
            # FIXME: Need to set test_evaluator as task-agnostic way
            test_evaluator = config.get("val_evaluator", [{"type": "Accuracy"}])
        kwargs["test_evaluator"] = configure_evaluator(test_evaluator, num_classes=num_classes, scope=scope)
    elif isinstance(config.get("test_dataloader", None), dict):
        # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
        config["test_dataloader"] = None
        config["test_cfg"] = None
        config["test_evaluator"] = None
    return config, kwargs


def update_runner_config(
    func_args: dict,
    config: Config,
    precision: Optional[str],
    num_classes: int,
    scope: Optional[str] = None,
    **kwargs,
) -> tuple:
    config, kwargs = update_train_config(func_args=func_args, config=config, precision=precision, **kwargs)
    return update_val_test_config(
        func_args=func_args,
        config=config,
        precision=precision,
        num_classes=num_classes,
        scope=scope,
        **kwargs,
    )
