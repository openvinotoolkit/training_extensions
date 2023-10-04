"""OTX adapters.torch.mmengine.Engine module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import copy
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
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
from otx.v2.adapters.torch.mmengine.utils.runner_config import get_value_from_config, update_runner_config
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.importing import get_all_args, get_default_args
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
            arg_config = get_value_from_config(arg_key, func_args, config=self.config)
            if arg_config is not None:
                kwargs[arg_key] = arg_config

        precision: Optional[str] = func_args.get("precision", None)
        precision = self.config.get(arg_key, None) if precision is None else precision

        # Update train_cfg & val_cfg (ValLoop) & test_cfg (TestLoop)
        self.config, kwargs = update_runner_config(
            func_args=func_args,
            config=self.config,
            precision=precision,
            num_classes=num_classes,
            scope=self.registry.name,
            **kwargs,
        )

        # Update randomness
        seed = func_args.get("seed", self.config.pop("seed", None))
        deterministic = func_args.get("deterministic", self.config.pop("deterministic", None))
        if seed is not None:
            kwargs["randomness"] = {"seed": seed, "deterministic": deterministic}

        distributed = get_value_from_config("distributed", func_args, config=self.config, default=False)
        default_hooks = get_value_from_config("default_hooks", func_args, config=self.config)
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
        visualizer = get_value_from_config("visualizer", func_args, config=self.config)
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
        runner_default_args = get_default_args(Runner.__init__)
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
            msg = f"In Engine.config, remove {removed_key} that are unavailable to the Runner."
            logger.warning(msg, stacklevel=2)
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
                msg = "Runner not found."
                raise ModuleNotFoundError(msg)
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
            ckpt_path = list(Path(target_folder).glob("epoch*.pth"))[-1]
        else:
            ckpt_path = list(Path(target_folder).glob("iter*.pth"))[-1]
        best_ckpt_path = list(Path(target_folder).glob("best_*.pth"))
        if len(best_ckpt_path) >= 1:
            ckpt_path = best_ckpt_path[0]
        last_ckpt = list(Path(target_folder).glob("last_checkpoint"))[-1]
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
                msg = "Runner not found."
                raise ModuleNotFoundError(msg)
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_validate",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **self.config,
            )
        elif update_check:
            # FIXME: SLF001
            self.runner._val_dataloader = self.config["val_dataloader"]  # noqa: SLF001
            self.runner._val_loop = self.config["val_cfg"]  # noqa: SLF001
            self.runner._val_evaluator = self.config["val_evaluator"]  # noqa: SLF001
            self.runner._experiment_name = f"otx_validate_{self.runner.timestamp}"  # noqa: SLF001
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
                msg = "Runner not found."
                raise ModuleNotFoundError(msg)
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_test",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **self.config,
            )
        elif update_check:
            # FIXME: SLF001
            self.runner._test_dataloader = self.config["test_dataloader"]  # noqa: SLF001
            self.runner._test_loop = self.config["test_cfg"]  # noqa: SLF001
            self.runner._test_evaluator = self.config["test_evaluator"]  # noqa: SLF001
            self.runner._experiment_name = f"otx_test_{self.runner.timestamp}"  # noqa: SLF001
        if checkpoint is not None:
            self.runner.load_checkpoint(checkpoint)

        self.dumped_config = dump_lazy_config(config=self.config, scope=self.registry.name)

        return self.runner.test()

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
        device: str = "cpu",
        input_shape: Optional[Tuple[int, int]] = None,
    ) -> dict:  # Output: IR Models
        if not AVAILABLE:
            msg = "MMXEngine's export is dependent on mmdeploy."
            raise ModuleNotFoundError(msg)
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
                model_cfg = model._config.get("model", model._config)  # noqa: SLF001
            else:
                raise NotImplementedError
        elif self.dumped_config.get("model", None) and self.dumped_config["model"] is not None:
            if isinstance(self.dumped_config["model"], dict):
                model_cfg = Config(self.dumped_config["model"])
            else:
                model_cfg = self.dumped_config["model"]
        else:
            msg = "Not fount target model."
            raise ValueError(msg)

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
