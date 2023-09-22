"""OTX adapters.torch.mmengine.Engine module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import copy
import glob
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from mmengine.device import get_device
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook
from mmengine.optim import _ParamScheduler
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from otx.v2.adapters.torch.mmengine.mmdeploy import AVAILABLE
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import dump_lazy_config
from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.importing import get_all_args, get_non_default_args
from otx.v2.api.utils.logger import get_logger

logger = get_logger()
MMENGINE_DTYPE = ("float16", "bfloat16", "float32", "float64")


class MMXEngine(Engine):
    def __init__(
        self,
        work_dir: Optional[Union[str, Path]] = None,
        config: Optional[Union[dict, Config, str]] = None,
    ) -> None:
        super().__init__(work_dir=work_dir)
        self.runner: Runner
        self.latest_model = {"model": None, "checkpoint": None}
        self.registry = MMEngineRegistry()
        self._initial_config(config)
        self.dumped_config = Config({})

    def _initial_config(self, config: Optional[Union[dict, Config, str]]) -> None:
        if config is not None:
            if isinstance(config, str):
                self.config = Config.fromfile(config)
            elif isinstance(config, Config):
                self.config = copy.deepcopy(config)
            elif isinstance(config, dict):
                self.config = Config(config)
        else:
            self.config = Config({})

    def _update_config(
        self,
        func_args: dict,
        **kwargs,
    ) -> bool:
        # TODO: Need to clean up.
        update_check = not all(value is None for value in func_args.values()) or not all(
            value is None for value in kwargs.values()
        )

        def _get_config_value(
            arg_key: str,
            positional_args: dict,
            default: Optional[Union[int, str]] = None,
        ) -> Optional[Union[dict, list]]:
            arg_config = positional_args.get(arg_key, None)
            arg_config = self.config.get(arg_key, default) if arg_config is None else arg_config
            return arg_config

        def _set_evaluator(evaluator: Union[list, dict], num_classes: int) -> Union[list, dict]:
            if isinstance(evaluator, list):
                for metric in evaluator:
                    if isinstance(metric, dict):
                        metric["_scope_"] = self.registry.name
                        if "topk" in metric:
                            metric["topk"] = [1] if num_classes < 5 else [1, 5]
            elif isinstance(evaluator, dict):
                evaluator["_scope_"] = self.registry.name
                if "topk" in evaluator:
                    evaluator["topk"] = [1] if num_classes < 5 else [1, 5]
            return evaluator

        # Update Model & Dataloaders & Custom hooks
        model = func_args.get("model", None)
        num_classes = -1
        if model is not None:
            kwargs["model"] = model
            if isinstance(model, torch.nn.Module):
                head = model.head if hasattr(model, "head") else None
                num_classes = head.num_classes if hasattr(head, "num_classes") else -1
            else:
                head = model.get("head", {})
                num_classes = head.get("num_classes", -1)
        for subset_dataloader in ("train_dataloader", "val_dataloader", "test_dataloader"):
            data_loader = func_args.get(subset_dataloader, None)
            if data_loader is not None:
                kwargs[subset_dataloader] = data_loader
        # Sub Arguments
        arg_keys = ["param_scheduler", "custom_hooks"]
        for arg_key in arg_keys:
            arg_config = _get_config_value(arg_key, func_args)
            if arg_config is not None:
                kwargs[arg_key] = arg_config

        # Update train_cfg
        precision = _get_config_value("precision", func_args)
        if kwargs.get("train_dataloader", None) is not None:
            max_iters = _get_config_value("max_iters", func_args)
            max_epochs = _get_config_value("max_epochs", func_args)
            val_interval = _get_config_value("val_interval", func_args, default=1)
            if max_iters is not None and max_epochs is not None:
                raise ValueError("Only one of `max_epochs` or `max_iters` can be set.")
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
                optimizer = _get_config_value("optimizer", func_args)
                if optimizer is None:
                    # FIXME: Remove default setting here
                    optimizer = {"type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}
                if get_device() not in ("cuda", "gpu", "npu", "mlu"):
                    logger.warning(f"{get_device()} device do not support mixed precision.")
                    kwargs["optim_wrapper"] = {"type": "OptimWrapper", "optimizer": optimizer}
                else:
                    kwargs["optim_wrapper"] = {"type": "AmpOptimWrapper", "dtype": precision, "optimizer": optimizer}
        elif isinstance(self.config.get("train_dataloader", None), dict):
            # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
            self.config["train_dataloader"] = None
            self.config["train_cfg"] = None
            self.config["optim_wrapper"] = None

        # Update val_cfg (ValLoop)
        if kwargs.get("val_dataloader", None) is not None:
            if "val_cfg" not in kwargs or kwargs["val_cfg"] is None:
                kwargs["val_cfg"] = {}
            if precision in ["float16", "fp16"]:
                kwargs["val_cfg"]["fp16"] = True
            # Update val_evaluator
            val_evaluator = _get_config_value("val_evaluator", func_args)
            if val_evaluator is None:
                # FIXME: Need to set val_evaluator as task-agnostic way
                val_evaluator = [{"type": "Accuracy"}]
            kwargs["val_evaluator"] = _set_evaluator(val_evaluator, num_classes=num_classes)
        elif isinstance(self.config.get("val_dataloader", None), dict):
            # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
            logger.warning("Currently, OTX does not accept val_dataloader as a dict configuration.")
            self.config["val_dataloader"] = None
            self.config["val_cfg"] = None
            self.config["val_evaluator"] = None

        # Update test_cfg (TestLoop)
        if kwargs.get("test_dataloader", None) is not None:
            if "test_cfg" not in kwargs or kwargs["test_cfg"] is None:
                kwargs["test_cfg"] = {}
            if precision in ["float16", "fp16"]:
                kwargs["test_cfg"]["fp16"] = True
            # Update test_evaluator
            test_evaluator = _get_config_value("test_evaluator", func_args)
            if test_evaluator is None:
                # FIXME: Need to set test_evaluator as task-agnostic way
                test_evaluator = self.config.get("val_evaluator", [{"type": "Accuracy"}])
            kwargs["test_evaluator"] = _set_evaluator(test_evaluator, num_classes=num_classes)
        elif isinstance(self.config.get("test_dataloader", None), dict):
            # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
            logger.warning("Currently, OTX does not accept test_dataloader as a dict configuration.")
            self.config["test_dataloader"] = None
            self.config["test_cfg"] = None
            self.config["test_evaluator"] = None

        # Update randomness
        seed = func_args.get("seed", self.config.pop("seed", None))
        deterministic = func_args.get("deterministic", self.config.pop("deterministic", None))
        if seed is not None:
            kwargs["randomness"] = {"seed": seed, "deterministic": deterministic}

        distributed = _get_config_value("distributed", func_args, default=False)
        default_hooks = _get_config_value("default_hooks", func_args)
        if default_hooks is None:
            # FIXME: Default hooks need to align
            default_hooks = {
                # record the time of every iterations.
                "timer": {"type": "IterTimerHook"},
                # print log every 100 iterations.
                "logger": {"type": "LoggerHook", "interval": 100},
                # enable the parameter scheduler.
                # TODO: lr_config -> param_scheduler
                "param_scheduler": {"type": "ParamSchedulerHook"},
                # save checkpoint per epoch, and automatically save the best checkpoint.
                "checkpoint": {
                    "type": "CheckpointHook",
                    "interval": 1,
                    "max_keep_ckpts": 1,
                    "save_best": "auto",
                },
                # set sampler seed in distributed evrionment.
                "sampler_seed": {"type": "DistSamplerSeedHook"} if distributed else None,
            }
        kwargs["default_hooks"] = default_hooks
        visualizer = _get_config_value("visualizer", func_args)
        if visualizer is not None:
            self.config["visualizer"] = visualizer
            if isinstance(visualizer, dict):
                self.config["visualizer"]["_scope_"] = self.registry.name

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
        # Last Check for Runner.__init__
        runner_arg_list = get_all_args(Runner.__init__)
        removed_key = []
        for config_key in self.config:
            if config_key not in runner_arg_list:
                removed_key.append(config_key)
        if removed_key:
            logger.warning(f"In Engine.config, remove {removed_key} " "that are unavailable to the Runner.")
            for config_key in removed_key:
                self.config.pop(config_key)
        return update_check

    def train(
        self,
        model: Optional[Union[torch.nn.Module, dict]] = None,
        train_dataloader: Optional[Union[DataLoader, dict]] = None,
        val_dataloader: Optional[Union[DataLoader, dict]] = None,
        optimizer: Optional[Union[dict, Optimizer]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        precision: Optional[str] = None,
        val_interval: Optional[int] = None,
        val_evaluator: Optional[Union[Evaluator, dict, list]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, dict, list]] = None,
        default_hooks: Optional[dict] = None,
        custom_hooks: Optional[Union[list, dict, Hook]] = None,
        visualizer: Optional[Union[Visualizer, dict]] = None,
        **kwargs,
    ) -> dict:
        r"""Training Functions with the MMEngine Framework.

        Args:
            model (Optional[Union[torch.nn.Module, Dict]], optional): The models available in Engine. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            train_dataloader (Optional[Union[DataLoader, Dict]], optional): Training Dataset's pipeline. Defaults to None.
            val_dataloader (Optional[Union[DataLoader, Dict]], optional): Validation Dataset's pipeline. Defaults to None.
            optimizer (Optional[Union[dict, Optimizer]], optional): _description_. Defaults to None.
            max_iters (Optional[int], optional): Specifies the maximum iters of training. Defaults to None.
            max_epochs (Optional[int], optional): Specifies the maximum epoch of training. Defaults to None.
            distributed (Optional[bool], optional): Whether to use the distributed setting. Defaults to None.
            seed (Optional[int], optional): The seed to use for training. Defaults to None.
            deterministic (Optional[bool], optional): The deterministic to use for training. Defaults to None.
            precision (Optional[str], optional): The precision to use for training. Defaults to None.
            val_interval (Optional[int], optional): Specifies the validation Interval. Defaults to None.
            val_evaluator (Optional[Union[Evaluator, Dict, List]], optional): A evaluator object
                used for computing metrics for validation. It can be a dict or a
                list of dict to build a evaluator. If specified,
                :attr:`val_dataloader` should also be specified. Defaults to None.
            default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
                execute default actions like updating model parameters and saving
                checkpoints. Default hooks are ``OptimizerHook``,
                ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook`` and
                ``CheckpointHook``. Defaults to None.
                See :meth:`register_default_hooks` for more details.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
            visualizer (Visualizer or dict, optional): A Visualizer object or a
                dict build Visualizer object. Defaults to None. If not
                specified, default config will be used.


        Returns:
            _type_: Output of training.
        """
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
            "val_interval": val_interval,
            "val_evaluator": val_evaluator,
            "param_scheduler": param_scheduler,
            "default_hooks": default_hooks,
            "custom_hooks": custom_hooks,
            "visualizer": visualizer,
        }
        update_check = self._update_config(func_args=train_args, **kwargs)

        target_folder = Path(self.work_dir) / f"{self.timestamp}_train"

        if not hasattr(self, "runner") or update_check:
            self.config.pop("work_dir")
            base_runner = self.registry.get("Runner")
            if base_runner is None:
                raise ModuleNotFoundError("Runner not found.")
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_train",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **self.config,
            )
        # TODO: Need to align outputs
        if checkpoint is not None:
            self.runner.load_checkpoint(checkpoint)
        self.dumped_config = dump_lazy_config(config=self.config, scope=self.registry.name)
        output_model = self.runner.train()

        # Get CKPT path
        if self.config.train_cfg.by_epoch:
            ckpt_path = glob.glob(str(target_folder / "epoch*.pth"))[-1]
        else:
            ckpt_path = glob.glob(str(target_folder / "iter*.pth"))[-1]
        best_ckpt_path = glob.glob(str(target_folder / "best_*.pth"))
        if len(best_ckpt_path) >= 1:
            ckpt_path = best_ckpt_path[0]
        last_ckpt = glob.glob(str(target_folder / "last_checkpoint"))[-1]
        # TODO: Clean up output
        # Copy & Remove weights file
        output_model_dir = target_folder / "models"
        output_model_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(Path(ckpt_path), output_model_dir / "weights.pth")
        shutil.copy(Path(last_ckpt), output_model_dir / "last_checkpoint")
        Path(ckpt_path).unlink()
        Path(last_ckpt).unlink()

        results = {"model": output_model, "checkpoint": str(output_model_dir / "weights.pth")}
        self.latest_model = results
        return results

    def validate(
        self,
        model: Optional[Union[torch.nn.Module, dict]] = None,
        val_dataloader: Optional[Union[DataLoader, dict]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        val_evaluator: Optional[Union[Evaluator, dict, list]] = None,
        **kwargs,
    ) -> dict:  # Metric (data_class or dict)
        val_args = {
            "model": model,
            "val_dataloader": val_dataloader,
            "val_evaluator": val_evaluator,
            "precision": precision,
        }
        update_check = self._update_config(func_args=val_args, **kwargs)
        target_folder = Path(self.work_dir) / f"{self.timestamp}_validate"
        if not hasattr(self, "runner"):
            self.config.pop("work_dir")
            base_runner = self.registry.get("Runner")
            if base_runner is None:
                raise ModuleNotFoundError("Runner not found.")
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_validate",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **self.config,
            )
        elif update_check:
            self.runner._val_dataloader = self.config["val_dataloader"]
            self.runner._val_loop = self.config["val_cfg"]
            self.runner._val_evaluator = self.config["val_evaluator"]
            self.runner._experiment_name = f"otx_validate_{self.runner.timestamp}"
        if checkpoint is not None:
            self.runner.load_checkpoint(checkpoint)

        self.dumped_config = dump_lazy_config(config=self.config, file=None, scope=self.registry.name)

        return self.runner.val()

    def test(
        self,
        model: Optional[Union[torch.nn.Module, dict]] = None,
        test_dataloader: Optional[DataLoader] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        test_evaluator: Optional[Union[Evaluator, dict, list]] = None,
        **kwargs,
    ) -> dict:  # Metric (data_class or dict)
        test_args = {
            "model": model,
            "test_dataloader": test_dataloader,
            "test_evaluator": test_evaluator,
            "precision": precision,
        }
        update_check = self._update_config(func_args=test_args, **kwargs)
        target_folder = Path(self.work_dir) / f"{self.timestamp}_test"
        if not hasattr(self, "runner"):
            self.config.pop("work_dir")
            base_runner = self.registry.get("Runner")
            if base_runner is None:
                raise ModuleNotFoundError("Runner not found.")
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_test",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **self.config,
            )
        elif update_check:
            self.runner._test_dataloader = self.config["test_dataloader"]
            self.runner._test_loop = self.config["test_cfg"]
            self.runner._test_evaluator = self.config["test_evaluator"]
            self.runner._experiment_name = f"otx_test_{self.runner.timestamp}"
        if checkpoint is not None:
            self.runner.load_checkpoint(checkpoint)

        self.dumped_config = dump_lazy_config(config=self.config, scope=self.registry.name)

        return self.runner.test()

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, dict, str]] = None,
        img: Optional[Union[str, np.ndarray, list]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        pipeline: Optional[Union[dict, list]] = None,
        **kwargs,
    ) -> list:
        raise NotImplementedError(f"{self}.predict is not implemented.")

    def export(
        self,
        model: Optional[
            Union[torch.nn.Module, str, Config]
        ] = None,  # Module with _config OR Model Config OR config-file
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: Optional[str] = None,
        codebase: Optional[str] = None,
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: Optional[str] = None,  # File path only?
        dump_features: bool = False,  # TODO
        device: str = "cpu",
        input_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> dict:  # Output: IR Models
        if not AVAILABLE:
            raise ModuleNotFoundError("MMXEngine's export is dependent on mmdeploy.")
        from mmdeploy.utils import get_backend_config, get_codebase_config, get_ir_config, load_config

        from otx.v2.adapters.torch.mmengine.mmdeploy.exporter import Exporter
        from otx.v2.adapters.torch.mmengine.mmdeploy.utils.deploy_cfg_utils import (
            patch_input_preprocessing,
            patch_input_shape,
        )

        # Configure model_cfg
        model_cfg = None
        if model is not None:
            if isinstance(model, str):
                model_cfg = Config.fromfile(model)
            elif isinstance(model, Config):
                model_cfg = copy.deepcopy(model)
            elif isinstance(model, torch.nn.Module) and hasattr(model, "_config"):
                model_cfg = model._config.get("model", model._config)
            else:
                raise NotImplementedError
        elif self.dumped_config.get("model", None) and self.dumped_config["model"] is not None:
            if isinstance(self.dumped_config["model"], dict):
                model_cfg = Config(self.dumped_config["model"])
            else:
                model_cfg = self.dumped_config["model"]
        else:
            raise ValueError("Not fount target model.")

        if checkpoint is None:
            checkpoint = self.latest_model.get("checkpoint", None)
        self.dumped_config["model"] = model_cfg
        self.dumped_config["default_scope"] = "mmengine"

        # Configure deploy_cfg
        codebase_config = None
        ir_config = None
        backend_config = None
        if deploy_config is not None:
            deploy_config_dict = load_config(deploy_config)[0]
            ir_config = get_ir_config(deploy_config_dict)
            backend_config = get_backend_config(deploy_config_dict)
            codebase_config = get_codebase_config(deploy_config_dict)
        else:
            deploy_config_dict = {}

        # CODEBASE_COFIG Update
        if codebase_config is None:
            codebase = codebase if codebase is not None else self.registry.name
            codebase_config = {"type": codebase, "task": task}
            deploy_config_dict["codebase_config"] = codebase_config
        # IR_COFIG Update
        if ir_config is None:
            ir_config = {
                "type": "onnx",
                "export_params": True,
                "keep_initializers_as_inputs": False,
                "opset_version": 11,
                "save_file": "end2end.onnx",
                "input_names": ["input"],
                "output_names": ["output"],
                "input_shape": None,
                "optimize": True,
                "dynamic_axes": {"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch"}},
            }
            deploy_config_dict["ir_config"] = ir_config
        # BACKEND_CONFIG Update
        if backend_config is None:
            backend_config = {"type": "openvino", "model_inputs": [{"opt_shapes": {"input": [1, 3, 224, 224]}}]}
            deploy_config_dict["backend_config"] = backend_config

        # Patch input's configuration
        if isinstance(deploy_config_dict, dict):
            deploy_config_dict = Config(deploy_config_dict)
        data_preprocessor = self.dumped_config.get("data_preprocessor", None)
        mean = data_preprocessor["mean"] if data_preprocessor is not None else [123.675, 116.28, 103.53]
        std = data_preprocessor["std"] if data_preprocessor is not None else [58.395, 57.12, 57.375]
        to_rgb = data_preprocessor["to_rgb"] if data_preprocessor is not None else False
        patch_input_preprocessing(deploy_cfg=deploy_config_dict, mean=mean, std=std, to_rgb=to_rgb)
        if not deploy_config_dict.backend_config.get("model_inputs", []):
            if input_shape is None:
                # TODO: Patch From self.config's test pipeline
                pass
            patch_input_shape(deploy_config_dict, input_shape=input_shape)

        export_dir = Path(self.work_dir) / f"{self.timestamp}_export"
        exporter = Exporter(
            config=self.dumped_config,
            checkpoint=str(checkpoint),
            deploy_config=deploy_config_dict,
            work_dir=str(export_dir),
            precision=precision,
            export_type=export_type,
            device=device,
        )

        return exporter.export()
