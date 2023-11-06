"""OTX adapters.torch.mmengine.mmpretrain.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mmseg.apis import MMSegInferencer

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmseg.registry import MMSegmentationRegistry
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    import numpy as np
    from mmengine.evaluator import Evaluator
    from mmengine.hooks import Hook
    from mmengine.optim import _ParamScheduler
    from mmengine.visualization import Visualizer
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

logger = get_logger()


class MMSegEngine(MMXEngine):
    """The MMPretrainEngine class is responsible for running inference on pre-trained models."""

    def __init__(
        self,
        work_dir: str | Path | None = None,
        config: dict | (Config | str) | None = None,
    ) -> None:
        """Initialize a new instance of the MMPretrainEngine class.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[Dict, Config, str]], optional): The configuration for the engine. Defaults to None.
        """
        super().__init__(work_dir=work_dir, config=config)
        self.registry = MMSegmentationRegistry()

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
        val_evaluator: Evaluator | dict | list | None = None,
        param_scheduler: _ParamScheduler | dict | list | None = None,
        default_hooks: dict | None = None,
        custom_hooks: list | dict | Hook | None = None,
        visualizer: Visualizer | dict | None = None,
        **kwargs,
    ) -> dict:
        """Train the model using the given data and hyperparameters.

        Args:
            model (torch.nn.Module | dict | None): The model to train.
            train_dataloader (DataLoader | dict | None): The dataloader for training data.
            val_dataloader (DataLoader | dict | None): The dataloader for validation data.
            optimizer (dict | Optimizer | None): The optimizer to use for training.
            checkpoint (str | Path | None): The path to save the checkpoint file.
            max_iters (int | None): The maximum number of iterations to train for.
            max_epochs (int | None): The maximum number of epochs to train for.
            distributed (bool | None): Whether to use distributed training.
            seed (int | None): The random seed to use for training.
            deterministic (bool | None): Whether to use deterministic training.
            precision (str | None): The precision to use for training.
            val_interval (int | None): The interval at which to perform validation.
            val_evaluator (Evaluator | dict | list | None): The evaluator to use for validation.
            param_scheduler (_ParamScheduler | dict | list | None): The parameter scheduler to use for training.
            default_hooks (dict | None): The default hooks to use for training.
            custom_hooks (list | dict | Hook | None): The custom hooks to use for training.
            visualizer (Visualizer | dict | None): The visualizer to use for training.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the training results.
        """
        if val_evaluator is None:
            val_evaluator = {
                "type": "IoUMetric",
                "iou_metrics": [
                    "mDice",
                ],
            }
        return super().train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            checkpoint,
            max_iters,
            max_epochs,
            distributed,
            seed,
            deterministic,
            precision,
            val_interval,
            val_evaluator,
            param_scheduler,
            default_hooks,
            custom_hooks,
            visualizer,
            **kwargs,
        )

    def predict(
        self,
        model: torch.nn.Module | (dict | str) | None = None,
        img: str | (np.ndarray | list) | None = None,
        checkpoint: str | Path | None = None,
        pipeline: dict | list | None = None,
        device: str | (torch.device | None) = None,
        task: str | None = None,  # noqa: ARG002
        batch_size: int = 1,
        **kwargs,
    ) -> list[dict]:
        """Runs inference on the given input image(s) using the specified model and checkpoint.

        Args:
            model (Optional[Union[torch.nn.Module, Dict, str]], optional): The model to use for inference. Can be a
                PyTorch module, a dictionary containing the model configuration, or a string representing the path to
                the model checkpoint file. Defaults to None.
            img (Optional[Union[str, np.ndarray, list]], optional): The input image(s) to run inference on. Can be a
                string representing the path to the image file, a NumPy array containing the image data, or a list of
                NumPy arrays containing multiple images. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): The path to the checkpoint file to use for inference.
                Defaults to None.
            pipeline (Optional[Union[Dict, List]], optional): The data pipeline to use for inference. Can be a
                dictionary containing the pipeline configuration, or a list of dictionaries containing multiple
                pipeline configurations. Defaults to None.
            device (Union[str, torch.device, None], optional): The device to use for inference. Can be a string
                representing the device name (e.g. 'cpu' or 'cuda'), a PyTorch device object, or None to use the
                default device. Defaults to None.
            task (Optional[str], optional): The type of task to perform. Defaults to None.
            batch_size (int, optional): The batch size to use for inference. Defaults to 1.
            **kwargs: Additional keyword arguments to pass to the inference function.

        Returns:
            List[Dict]: A list of dictionaries containing the inference results.
        """
        # Model config need data_pipeline of test_dataloader
        # Update pipelines
        if pipeline is None:
            pipeline = [
                {"type": "LoadImageFromFile"},
                {"type": "Resize", "scale": (544, 544)},
                {"type": "PackSegInputs", "_scope_": "mmseg"},
            ]
        config = Config({})
        if isinstance(model, torch.nn.Module) and hasattr(model, "_config"):
            config = model._config  # noqa: SLF001
        elif isinstance(model, dict) and "_config" in model:
            config = model["_config"]
        config["test_dataloader"] = {"dataset": {"pipeline": pipeline}}

        # Check if the model can use mmseg's inference api.
        if isinstance(checkpoint, Path):
            checkpoint = str(checkpoint)

        inferencer = MMSegInferencer(
            model=config,
            weights=checkpoint,
            device=device,
        )

        return inferencer(img, batch_size=batch_size, **kwargs)

    def export(
        self,
        model: torch.nn.Module | (str | Config) | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: str | None = "Segmentation",
        codebase: str | None = "mmseg",
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: str | None = None,  # File path only?
        device: str = "cpu",
        input_shape: tuple[int, int] | None = None,
        **kwargs,
    ) -> dict:
        """Export a PyTorch model to a specified format for deployment.

        Args:
            model (Optional[Union[torch.nn.Module, str, Config]]): The PyTorch model to export.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint file to use for exporting.
            precision (Optional[str]): The precision to use for exporting.
                Can be one of ["float16", "fp16", "float32", "fp32"].
            task (Optional[str]): The task for which the model is being exported. Defaults to "Classification".
            codebase (Optional[str]): The codebase for the model being exported. Defaults to "mmpretrain".
            export_type (str): The type of export to perform. Can be one of "ONNX" or "OPENVINO". Defaults to "OPENVINO"
            deploy_config (Optional[str]): The path to the deployment configuration file to use for exporting.
                File path only.
            device (str): The device to use for exporting. Defaults to "cpu".
            input_shape (Optional[Tuple[int, int]]): The input shape of the model being exported.
            **kwargs: Additional keyword arguments to pass to the export function.

        Returns:
            dict: A dictionary containing information about the exported model.
        """
        return super().export(
            model=model,
            checkpoint=checkpoint,
            precision=precision,
            task=task,
            codebase=codebase,
            export_type=export_type,
            deploy_config=deploy_config,
            device=device,
            input_shape=input_shape,
            **kwargs,
        )
