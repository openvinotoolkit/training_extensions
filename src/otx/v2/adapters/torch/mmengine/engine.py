import copy
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from otx.v2.adapters.torch.mmengine.mmdeploy import is_mmdeploy_enabled
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig as Config
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import dump_lazy_config
from otx.v2.adapters.torch.mmengine.registry import MMEngineRegistry
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.importing import get_all_args, get_non_default_args
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
        work_dir: Optional[str] = None,
        config: Optional[Union[Dict, Config, str]] = None,
    ) -> None:
        super().__init__(work_dir=work_dir)
        self.runner = None
        self.latest_model = {"model": None, "checkpoint": None}
        self.registry = MMEngineRegistry()
        # self.base_runner = self.registry.get("RUNNER")
        self.initial_config(config)
        self.dumped_config = Config({})

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
        if "work_dir" not in kwargs:
            if self.work_dir is None:
                raise ValueError("Engine.work_dir is None.")
            kwargs["work_dir"] = self.work_dir

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
        max_iters = self.config.get("max_iters", None) if max_iters is None else max_iters
        max_epochs = func_args.get("max_epochs", None)
        max_epochs = self.config.get("max_epochs", None) if max_epochs is None else max_epochs
        precision = func_args.get("precision", None)
        precision = self.config.get("precision", None) if precision is None else precision
        eval_interval = func_args.get("eval_interval", 1)
        if max_iters is not None and max_epochs is not None:
            raise ValueError("Only one of `max_epochs` or `max_iters` can be set.")
        if "train_cfg" not in kwargs or kwargs["train_cfg"] is None:
            eval_interval = eval_interval if eval_interval is not None else 1
            kwargs["train_cfg"] = dict(val_interval=eval_interval, by_epoch=True)
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
            if precision in ["float16", "fp16"]:
                kwargs["val_cfg"]["fp16"] = True

            # Update val_evaluator
            if "val_evaluator" not in kwargs or kwargs["val_evaluator"] is None:
                # TODO: Need to set val_evaluator as task-agnostic way
                kwargs["val_evaluator"] = dict(type="mmpretrain.Accuracy")
        elif isinstance(self.config.get("val_dataloader", None), dict):
            # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
            logger.warning("Currently, OTX does not accept val_dataloader as a dict configuration.")
            self.config["val_dataloader"] = None
            self.config["val_cfg"] = None
            self.config["val_evaluator"] = None

        # Update test_cfg (TestLoop)
        if test_dataloader is not None:
            if "test_cfg" not in kwargs or kwargs["test_cfg"] is None:
                kwargs["test_cfg"] = dict()
            if precision in ["float16", "fp16"]:
                kwargs["test_cfg"]["fp16"] = True

            # Update test_evaluator
            if "test_evaluator" not in kwargs or kwargs["test_evaluator"] is None:
                # TODO: Need to set test_evaluator as task-agnostic way
                kwargs["test_evaluator"] = self.config.get("val_evaluator", dict(type="mmpretrain.Accuracy"))
        elif isinstance(self.config.get("test_dataloader", None), dict):
            # FIXME: This is currently not possible because it requires the use of a DatasetEntity.
            logger.warning("Currently, OTX does not accept test_dataloader as a dict configuration.")
            self.config["test_dataloader"] = None
            self.config["test_cfg"] = None
            self.config["test_evaluator"] = None

        # Update randomness
        seed = func_args.get("seed", self.config.pop("seed", None))
        deterministic = func_args.get("deterministic", self.config.pop("deterministic", None))
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
        if self.config.get("visualizer", None) is not None:
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
        for config_key in self.config.keys():
            if config_key not in runner_arg_list:
                removed_key.append(config_key)
        if removed_key:
            logger.warning(f"In Engine.config, remove {removed_key} " "that are unavailable to the Runner.")
            for config_key in removed_key:
                self.config.pop(config_key)

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
        r"""Training Functions with the MMEngine Framework.

        Args:
            model (Optional[Union[torch.nn.Module, Dict]], optional): The models available in Engine. Defaults to None.
            train_dataloader (Optional[Union[DataLoader, Dict]], optional): Training Dataset's pipeline. Defaults to None.
            val_dataloader (Optional[Union[DataLoader, Dict]], optional): Validation Dataset's pipeline. Defaults to None.
            optimizer (Optional[Union[dict, Optimizer]], optional): _description_. Defaults to None.
            max_iters (Optional[int], optional): Specifies the maximum iters of training. Defaults to None.
            max_epochs (Optional[int], optional): Specifies the maximum epoch of training. Defaults to None.
            distributed (Optional[bool], optional): Whether to use the distributed setting. Defaults to None.
            seed (Optional[int], optional): The seed to use for training. Defaults to None.
            deterministic (Optional[bool], optional): The deterministic to use for training. Defaults to None.
            precision (Optional[str], optional): The precision to use for training. Defaults to None.
            eval_interval (Optional[int], optional): Specifies the validation Interval. Defaults to None.
            custom_hooks (Optional[Union[List, Dict, Hook]], optional): Custom Hooks for mmengine.Runner. Defaults to None.

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
            "eval_interval": eval_interval,
            "custom_hooks": custom_hooks,
        }

        update_check = self.update_config(func_args=train_args, **kwargs)
        if self.runner is None or update_check:
            base_runner = self.registry.get("Runner")
            self.runner = base_runner(**self.config)
        # TODO: Need to align outputs
        config_path = Path(self.work_dir) / f"{self.runner.timestamp}" / "configs.py"
        self.dumped_config = dump_lazy_config(config=self.config, file=config_path, scope=self.registry.name)

        output_model = self.runner.train()

        # Get CKPT path
        if self.config.train_cfg.by_epoch:
            ckpt_path = glob.glob(str(Path(self.work_dir) / "epoch*.pth"))[-1]
        else:
            ckpt_path = glob.glob(str(Path(self.work_dir) / "iter*.pth"))[-1]
        best_ckpt_path = glob.glob(str(Path(self.work_dir) / "best_*.pth"))
        if len(best_ckpt_path) >= 1:
            ckpt_path = best_ckpt_path[0]
        self.latest_model = {"model": output_model, "checkpoint": ckpt_path}
        return output_model, ckpt_path

    def validate(
        self,
        model: Optional[Union[torch.nn.Module, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, float]:  # Metric (data_class or dict)
        val_args = {
            "model": model,
            "val_dataloader": val_dataloader,
            "precision": precision,
        }
        update_check = self.update_config(func_args=val_args, **kwargs)
        if self.runner is None:
            base_runner = self.registry.get("Runner")
            self.runner = base_runner(**self.config)
        elif update_check:
            self.runner._val_dataloader = self.config["val_dataloader"]
            self.runner._val_loop = self.config["val_cfg"]
            self.runner._val_evaluator = self.config["val_evaluator"]

        Path(self.config.work_dir) / f"{self.runner.timestamp}" / "configs.py"
        self.dumped_config = dump_lazy_config(config=self.config, file=None, scope=self.registry.name)

        return self.runner.val()

    def test(
        self,
        model: Optional[Union[torch.nn.Module, Dict]] = None,
        test_dataloader: Optional[DataLoader] = None,
        precision: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, float]:  # Metric (data_class or dict)
        test_args = {
            "model": model,
            "test_dataloader": test_dataloader,
            "precision": precision,
        }
        update_check = self.update_config(func_args=test_args, **kwargs)
        # self.config = dump_lazy_config(config=self.config, file=None, scope=self.registry.name)
        if self.runner is None:
            base_runner = self.registry.get("Runner")
            # self.runner = base_runner.from_cfg(self.config)
            self.runner = base_runner(**self.config)
        elif update_check:
            self.runner._test_dataloader = self.config["test_dataloader"]
            self.runner._test_loop = self.config["test_cfg"]
            self.runner._test_evaluator = self.config["test_evaluator"]

        config_path = Path(self.config.work_dir) / f"{self.runner.timestamp}" / "configs.py"
        self.dumped_config = dump_lazy_config(config=self.config, file=config_path, scope=self.registry.name)

        return self.runner.test()

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, Dict]] = None,
        dataloader: Optional[Union[DataLoader, Dict]] = None,
        checkpoint: Optional[str] = None,
    ):
        raise NotImplementedError()

    def export(
        self,
        model: Optional[
            Union[torch.nn.Module, str, Config]
        ] = None,  # Module with _build_config OR Model Config OR config-file
        checkpoint: Optional[str] = None,
        task: Optional[str] = None,
        codebase: Optional[str] = None,
        precision: str = "float32",  # ["float16", "fp16", "float32", "fp32"]
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: Optional[str] = None,  # File path only?
        dump_features: bool = False,  # TODO
        device: str = "cpu",
        input_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):  # Output: IR Models
        if not is_mmdeploy_enabled():
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
            elif isinstance(model, torch.nn.Module) and hasattr(model, "_build_config"):
                model_cfg = Config(model._build_config)
            else:
                raise NotImplementedError()
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

        # Configure deploy_cfg
        codebase_config = None
        ir_config = None
        backend_config = None
        if deploy_config is not None:
            deploy_config = load_config(deploy_config)[0]
            ir_config = get_ir_config(deploy_config)
            backend_config = get_backend_config(deploy_config)
            codebase_config = get_codebase_config(deploy_config)
        else:
            deploy_config = {}

        # CODEBASE_COFIG Update
        if codebase_config is None:
            codebase = codebase if codebase is not None else self.registry.name
            codebase_config = dict(type=codebase, task=task)
            deploy_config["codebase_config"] = codebase_config
        # IR_COFIG Update
        if ir_config is None:
            ir_config = dict(
                type="onnx",
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                save_file="end2end.onnx",
                input_names=["input"],
                output_names=["output"],
                input_shape=None,
                optimize=True,
                dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch"}},
            )
            deploy_config["ir_config"] = ir_config
        # BACKEND_CONFIG Update
        if backend_config is None:
            backend_config = dict(type="openvino", model_inputs=[dict(opt_shapes=dict(input=[1, 3, 224, 224]))])
            deploy_config["backend_config"] = backend_config

        # Patch input's configuration
        if isinstance(deploy_config, dict):
            deploy_config = Config(deploy_config)
        data_preprocessor = self.dumped_config.get("data_preprocessor", None)
        mean = data_preprocessor["mean"] if data_preprocessor is not None else [123.675, 116.28, 103.53]
        std = data_preprocessor["std"] if data_preprocessor is not None else [58.395, 57.12, 57.375]
        to_rgb = data_preprocessor["to_rgb"] if data_preprocessor is not None else False
        patch_input_preprocessing(deploy_cfg=deploy_config, mean=mean, std=std, to_rgb=to_rgb)
        if not deploy_config.backend_config.get("model_inputs", []):
            if input_shape is None:
                # TODO: Patch From self.config's test pipeline
                pass
            patch_input_shape(deploy_config, input_shape=input_shape)

        exporter = Exporter(
            config=self.dumped_config,
            checkpoint=checkpoint,
            deploy_config=deploy_config,
            work_dir=f"{self.work_dir}/openvino",
            precision=precision,
            export_type=export_type,
            device=device,
        )
        exporter.export()

        results: Dict[str, Dict[str, str]] = {"outputs": {}}

        if export_type.upper() == "ONNX":
            onnx_file = [f for f in Path(self.work_dir).iterdir() if str(f).endswith(".onnx")][0]
            results["outputs"]["onnx"] = str(Path(self.work_dir) / onnx_file)
        else:
            bin_file = [f for f in Path(self.work_dir).iterdir() if str(f).endswith(".bin")][0]
            xml_file = [f for f in Path(self.work_dir).iterdir() if str(f).endswith(".xml")][0]
            results["outputs"]["bin"] = str(Path(self.work_dir) / bin_file)
            results["outputs"]["xml"] = str(Path(self.work_dir) / xml_file)

        return results


### DONE ### CLI -> Geti~
"""

"""
