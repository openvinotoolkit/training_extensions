"""OTX adapters.torch.mmengine.Engine module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mmengine.device import get_device
from mmengine.runner import Runner

from otx.v2.adapters.torch.mmengine.mmdeploy import AVAILABLE
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import dump_lazy_config
from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.importing import get_all_args, get_default_args
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    from mmengine.evaluator import Evaluator
    from mmengine.hooks import Hook
    from mmengine.optim import _ParamScheduler
    from mmengine.visualization import Visualizer
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

logger = get_logger()
MMENGINE_DTYPE = ("float16", "bfloat16", "float32", "float64")
# NOTE: We need to discuss where to declare the default value.
DEFAULT_CONFIG = Config({
    "val_interval": 1,
    "seed": 1234,
    "deterministic": False,
    "precision": "float32",
    "default_hooks": {
        "logger": {
            "type": "LoggerHook",
            "interval": 100,
        },
        "timer": {
            "type": "IterTimerHook",
        },
        "checkpoint": {
            "type": "CheckpointHook",
            "interval": 1,
            "save_best": "auto",
            "max_keep_ckpts": 1,
        },
    },
    "custom_hooks": [],
    "visualizer": {
        "type": "UniversalVisualizer",
        "vis_backends": [
            {"type": "LocalVisBackend"},
            {"type": "TensorboardVisBackend"},
        ],
    },
    "optimizer": {
        "type": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005,
    },
})


class MMXEngine(Engine):
    """OTX adapters.torch.mmengine.MMXEngine class.

    This class is a subclass of the otx.v2.api.core.engine.Engine class and provides additional functionality
    for training and evaluating PyTorch models using the MMEngine framework.
    """
    default_config = DEFAULT_CONFIG

    def __init__(
        self,
        work_dir: str | Path | None = None,
    ) -> None:
        """Initialize a new instance of the MMEngine class.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
        """
        super().__init__(work_dir=work_dir)
        self.runner: Runner
        self.latest_model = {"model": None, "checkpoint": None}
        self.registry = MMEngineRegistry()
        self.dumped_config = Config({})

    def _get_value_from_config(
        self,
        arg_key: str,
        positional_args: dict,
    ) -> dict | list | Config | None:
        """Get the value of a given argument key from either the positional arguments or the config.

        Args:
            arg_key (str): The key of the argument to retrieve.
            positional_args (dict): The positional arguments passed to the function.

        Returns:
            dict | list | None: The value of the argument, or the default value if not found.

        Examples:
        >>> self.default_config = Config({"max_epochs": 20})
        >>> get_value_from_config(
                arg_key="max_epochs",
                positional_args={"max_epochs": 10},
            )
        10
        >>> get_value_from_config(
                arg_key="max_epochs",
                positional_args={},
            )
        20
        """
        # Priority 1: Positional Args
        result = positional_args.get(arg_key, None)
        # Priority 2: Input Config Value
        return self.default_config.get(arg_key, None) if result is None else result

    def _update_train_config(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        arguments: dict,
        config: Config,
    ) -> None:
        """Update the training configuration with the given arguments and default configuration.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The training dataloader.
            arguments (dict): The arguments to update the configuration.
            default_config (Config): The default configuration.
            config (Config): The configuration to update.

        Raises:
            ValueError: If both `max_epochs` and `max_iters` are set.

        Returns:
            None
        """
        config["train_dataloader"] = train_dataloader
        precision = self._get_value_from_config("precision", arguments)
        max_iters = self._get_value_from_config("max_iters", arguments)
        max_epochs = self._get_value_from_config("max_epochs", arguments)
        val_interval = self._get_value_from_config("val_interval", arguments)
        if max_iters is not None and max_epochs is not None:
            msg = "Only one of `max_epochs` or `max_iters` can be set."
            raise ValueError(msg)
        if "train_cfg" not in config or config["train_cfg"] is None:
            config["train_cfg"] = {"val_interval": val_interval, "by_epoch": True}
        if max_epochs is not None:
            config["train_cfg"]["by_epoch"] = True
            config["train_cfg"]["max_epochs"] = max_epochs
        elif max_iters is not None:
            config["train_cfg"]["by_epoch"] = False
            config["train_cfg"]["max_iters"] = max_iters
        # Update Optimizer
        if "optim_wrapper" not in config or config["optim_wrapper"] is None:
            optimizer = self._get_value_from_config("optimizer", arguments)
            if get_device() not in ("cuda", "gpu", "npu", "mlu"):
                config["optim_wrapper"] = {"type": "OptimWrapper", "optimizer": optimizer}
            else:
                config["optim_wrapper"] = {
                    "type": "AmpOptimWrapper", "dtype": precision, "optimizer": optimizer,
                }


    def _update_config(
        self,
        func_args: dict,
        **kwargs,
    ) -> tuple[Config, bool]:
        """Update the configuration of the runner with the provided arguments.

        Args:
            func_args (dict): The arguments passed to the engine.
            **kwargs: Additional keyword arguments to update the configuration for mmengine.Runner.

        Returns:
            tuple[Config, bool]: Config, True if the configuration was updated, False otherwise.
        """
        update_check = not all(value is None for value in func_args.values()) or not all(
            value is None for value in kwargs.values()
        )

        runner_config = Config({})
        for key, value in kwargs.items():
            if value is None:
                continue
            runner_config[key] = value

        # Update Model & Dataloaders
        model = func_args.get("model", None)
        if model is not None:
            runner_config["model"] = model

        # train_dataloader & train_cfg & optim_wrapper
        train_dataloader = func_args.get("train_dataloader", None)
        precision = self._get_value_from_config("precision", func_args)
        if train_dataloader is not None:
            self._update_train_config(
                train_dataloader=train_dataloader,
                arguments=func_args,
                config=runner_config,
            )
        elif train_dataloader is None:
            runner_config["train_dataloader"] = None
            runner_config["train_cfg"] = None
            runner_config["optim_wrapper"] = None

        for subset in ("val", "test"):
            data_loader = func_args.get(f"{subset}_dataloader", None)
            if data_loader is not None:
                runner_config[f"{subset}_dataloader"] = data_loader
                if f"{subset}_cfg" not in runner_config or runner_config[f"{subset}_cfg"] is None:
                    runner_config[f"{subset}_cfg"] = {}
                if precision in ["float16", "fp16"]:
                    runner_config[f"{subset}_cfg"]["fp16"] = True
                # Update val_evaluator
                evaluator = self._get_value_from_config(f"{subset}_evaluator", func_args)
                runner_config[f"{subset}_evaluator"] = evaluator if evaluator is not None else {}
            else:
                runner_config[f"{subset}_dataloader"] = None
                runner_config[f"{subset}_cfg"] = None
                runner_config[f"{subset}_evaluator"] = None

        # Update randomness: seed & deterministic
        seed = self._get_value_from_config("seed", func_args)
        deterministic = self._get_value_from_config("deterministic", func_args)
        if seed is not None:
            runner_config["randomness"] = {"seed": seed, "deterministic": deterministic}

        # Update param_scheduler, default_hooks, custom_hooks, visualizer
        for runner_key in ("param_scheduler", "default_hooks", "custom_hooks", "visualizer"):
            runner_value = self._get_value_from_config(runner_key, func_args)
            if runner_value is not None:
                runner_config[runner_key] = runner_value

        # Update scope for Registry import step
        runner_config["default_scope"] = self.registry.name

        # Check Config Default is not None
        runner_default_args = get_default_args(Runner.__init__)
        for not_none_arg, default_value in runner_default_args:
            if runner_config.get(not_none_arg, None) is None:
                runner_config[not_none_arg] = default_value
        # Last Check for Runner.__init__
        runner_arg_list = get_all_args(Runner.__init__)
        removed_key = [config_key for config_key in runner_config if config_key not in runner_arg_list]
        if removed_key:
            msg = f"In Engine.config, remove {removed_key} that are unavailable to the Runner."
            logger.warning(msg, stacklevel=2)
            for config_key in removed_key:
                runner_config.pop(config_key)
        return runner_config, update_check

    def train(
        self,
        model: torch.nn.Module | dict | None = None,
        train_dataloader: DataLoader | dict | None = None,
        val_dataloader: DataLoader | dict | None = None,
        optimizer: dict | Optimizer | None = None,
        checkpoint: str | Path | None = None,
        max_iters: int | None = None,
        max_epochs: int | None = None,
        distributed: bool | None = None,
        seed: int | None = None,
        deterministic: bool | None = None,
        precision: str | None = None,
        val_interval: int | None = None,
        val_evaluator: Evaluator | (dict | list) | None = None,
        param_scheduler: _ParamScheduler | (dict | list) | None = None,
        default_hooks: dict | None = None,
        custom_hooks: list | (dict | Hook) | None = None,
        visualizer: Visualizer | dict | None = None,
        **kwargs,
    ) -> dict:
        """Train the given model using the provided data and configuration.

        Args:
            model (Optional[Union[torch.nn.Module, Dict]], optional): The models available in Engine. Defaults to None.
            train_dataloader (Optional[Union[DataLoader, Dict]], optional): Training Dataset's pipeline.
                Defaults to None.
            val_dataloader (Optional[Union[DataLoader, Dict]], optional): Validation Dataset's pipeline.
                Defaults to None.
            optimizer (Optional[Union[dict, Optimizer]], optional): optimizer for training. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
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
            param_scheduler (Optional[Union[_ParamScheduler, dict, list]]): The parameter scheduler to use for training.
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
            **kwargs (Any): This is used as an additional parameter to mmengine.Engine.

        Returns:
            dict: A dictionary containing the trained model and the path to the checkpoint file.
        """
        if isinstance(model, torch.nn.Module):
            model.train()
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
        config, update_check = self._update_config(func_args=train_args, **kwargs)

        target_folder = Path(self.work_dir) / f"{self.timestamp}_train"

        if not hasattr(self, "runner") or update_check:
            config.pop("work_dir")
            base_runner = self.registry.get("Runner")
            if base_runner is None:
                msg = "Runner not found."
                raise ModuleNotFoundError(msg)
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_train",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **config,
            )
        if checkpoint is not None:
            self.runner.load_checkpoint(checkpoint)
        self.dumped_config = dump_lazy_config(config=config, scope=self.registry.name)
        output_model = self.runner.train()

        # Get CKPT path
        if config.train_cfg.by_epoch:
            ckpt_path = list(Path(target_folder).glob("epoch*.pth"))[-1]
        else:
            ckpt_path = list(Path(target_folder).glob("iter*.pth"))[-1]
        best_ckpt_path = list(Path(target_folder).glob("best_*.pth"))
        if len(best_ckpt_path) >= 1:
            ckpt_path = best_ckpt_path[0]
        last_ckpt = list(Path(target_folder).glob("last_checkpoint"))[-1]
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
        model: torch.nn.Module | dict | None = None,
        val_dataloader: DataLoader | dict | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = None,
        val_evaluator: Evaluator | (dict | list) | None = None,
        **kwargs,
    ) -> dict:
        """Run validation on the given model using the provided validation dataloader and evaluator.

        Args:
            model (Optional[Union[torch.nn.Module, dict]]): The model to be validated.
            val_dataloader (Optional[Union[DataLoader, dict]]): The validation dataloader.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint to be loaded.
            precision (Optional[str]): The precision of the model.
            val_evaluator (Optional[Union[Evaluator, dict, list]]): The evaluator to be used for validation.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The validation results.
        """
        if isinstance(model, torch.nn.Module):
            model.eval()
        val_args = {
            "model": model,
            "val_dataloader": val_dataloader,
            "val_evaluator": val_evaluator,
            "precision": precision,
        }
        config, update_check = self._update_config(func_args=val_args, **kwargs)
        target_folder = Path(self.work_dir) / f"{self.timestamp}_validate"
        if not hasattr(self, "runner"):
            config.pop("work_dir")
            base_runner = self.registry.get("Runner")
            if base_runner is None:
                msg = "Runner not found."
                raise ModuleNotFoundError(msg)
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_validate",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **config,
            )
        elif update_check:
            self.runner._val_dataloader = config["val_dataloader"]  # noqa: SLF001
            self.runner._val_loop = config["val_cfg"]  # noqa: SLF001
            self.runner._val_evaluator = config["val_evaluator"]  # noqa: SLF001
            self.runner._experiment_name = f"otx_validate_{self.runner.timestamp}"  # noqa: SLF001
        if checkpoint is not None:
            self.runner.load_checkpoint(checkpoint)

        self.dumped_config = dump_lazy_config(config=config, file=None, scope=self.registry.name)

        return self.runner.val()

    def test(
        self,
        model: torch.nn.Module | dict | None = None,
        test_dataloader: DataLoader | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = None,
        test_evaluator: Evaluator | (dict | list) | None = None,
        **kwargs,
    ) -> dict:
        """Test the given model on the test dataset.

        Args:
            model (torch.nn.Module or dict, optional): The model to test. If None, the model
                passed to the latest will be used. Defaults to None.
            test_dataloader (DataLoader, optional): The dataloader to use for testing. Defaults to None.
            checkpoint (str or Path, optional): The path to the checkpoint to load before testing.
                If None, the checkpoint passed to the latest will be used. Defaults to None.
            precision (str, optional): The precision to use for testing. Defaults to None.
            test_evaluator (Evaluator or dict or list, optional): The evaluator(s) to use for testing.
                Defaults to None.
            **kwargs: Additional keyword arguments to update the configuration.

        Returns:
            dict: A dictionary containing the test results.
        """
        if isinstance(model, torch.nn.Module):
            model.eval()
        test_args = {
            "model": model,
            "test_dataloader": test_dataloader,
            "test_evaluator": test_evaluator,
            "precision": precision,
        }
        config, update_check = self._update_config(func_args=test_args, **kwargs)
        # This will not build if there are training-related hooks.
        config["custom_hooks"] = None
        target_folder = Path(self.work_dir) / f"{self.timestamp}_test"
        if not hasattr(self, "runner"):
            config.pop("work_dir")
            base_runner = self.registry.get("Runner")
            if base_runner is None:
                msg = "Runner not found."
                raise ModuleNotFoundError(msg)
            self.runner = base_runner(
                work_dir=str(target_folder),
                experiment_name="otx_test",
                cfg=Config({}),  # To prevent unnecessary dumps.
                **config,
            )
        elif update_check:
            self.runner._test_dataloader = config["test_dataloader"]  # noqa: SLF001
            self.runner._test_loop = config["test_cfg"]  # noqa: SLF001
            self.runner._test_evaluator = config["test_evaluator"]  # noqa: SLF001
            self.runner._experiment_name = f"otx_test_{self.runner.timestamp}"  # noqa: SLF001
        if checkpoint is not None:
            self.runner.load_checkpoint(checkpoint)

        self.dumped_config = dump_lazy_config(config=config, scope=self.registry.name)

        return self.runner.test()

    def export(
        self,
        model: torch.nn.Module | (str | Config) | None = None,  # Module with _config OR Model Config OR config-file
        checkpoint: str | Path | None = None,
        precision: str | None = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: str | None = None,
        codebase: str | None = None,
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: str | None = None,  # File path only?
        device: str = "cpu",
        input_shape: tuple[int, int] | None = None,
    ) -> dict:
        """Export the model to an intermediate representation (IR) format.

        Args:
            model (Optional[Union[torch.nn.Module, str, Config]]): The model to export.
                Can be a PyTorch module with a `_config` attribute, a model config, or a path to a config file.
                Defaults to None.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint file to use for exporting.
                Defaults to None.
            precision (Optional[str]): The precision to use for exporting.
                Can be "float16", "fp16", "float32", or "fp32". Defaults to "float32".
            task (Optional[str]): The task to use for exporting. Defaults to None.
            codebase (Optional[str]): The codebase to use for exporting. Defaults to None.
            export_type (str): The type of export to perform. Can be "ONNX" or "OPENVINO". Defaults to "OPENVINO".
            deploy_config (Optional[str]): The path to the deploy config file to use for exporting. Defaults to None.
            device (str): The device to use for exporting. Defaults to "cpu".
            input_shape (Optional[Tuple[int, int]]): The input shape to use for exporting. Defaults to None.

        Returns:
            dict: The intermediate representation (IR or onnx) models.
        """
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
        elif "model" in self.dumped_config:
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
            self._update_codebase_config(codebase, task, deploy_config_dict)

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
        data_preprocessor = self.dumped_config.get("model", {}).get("data_preprocessor", None)
        mean = data_preprocessor["mean"] if data_preprocessor is not None else [123.675, 116.28, 103.53]
        std = data_preprocessor["std"] if data_preprocessor is not None else [58.395, 57.12, 57.375]
        to_rgb = data_preprocessor.get("bgr_to_rgb", data_preprocessor.get("to_rgb", False))
        patch_input_preprocessing(deploy_cfg=deploy_config_dict, mean=mean, std=std, to_rgb=to_rgb)
        if not deploy_config_dict.backend_config.get("model_inputs", []):
            if input_shape is None:
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

    def _update_codebase_config(self, codebase: str | None, task: str | None, deploy_config_dict: dict) -> None:
        """Update specific codebase config.

        Args:
            codebase(str): mmX codebase framework
            task(str): mmdeploy task
            deploy_config_dict(dict): Config dict for deployment
        """
        codebase = codebase if codebase is not None else self.registry.name
        codebase_config = {"type": codebase, "task": task}
        deploy_config_dict["codebase_config"] = codebase_config
