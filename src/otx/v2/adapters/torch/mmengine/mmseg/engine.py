"""OTX adapters.torch.mmengine.mmseg.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mmseg.apis import MMSegInferencer
from mmseg.registry import VISUALIZERS

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmseg.registry import MMSegmentationRegistry
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger()


class MMSegEngine(MMXEngine):
    """The MMSegmentation class is responsible for running inference on pre-trained models."""

    def __init__(self, work_dir: str | Path | None = None) -> None:
        """Initialize a new instance of the MMPretrainEngine class.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[Dict, Config, str]], optional): The configuration for the engine. Defaults to None.
        """
        super().__init__(work_dir=work_dir)
        self.registry = MMSegmentationRegistry()
        self.visualizer_cfg = {"name": "visualizer", "type": "SegLocalVisualizer"}
        self.evaluator_cfg = {"type": "IoUMetric", "iou_metrics": ["mDice"]}

    def _update_config(self, func_args: dict, **kwargs) -> tuple[Config, bool]:
        """Update the configuration of the runner with the provided arguments.

        Args:
            func_args (dict): The arguments passed to the engine.
            **kwargs: Additional keyword arguments to update the configuration for mmengine.Runner.

        Returns:
            tuple[Config, bool]: Config, True if the configuration was updated, False otherwise.
        """
        config, update_check = super()._update_config(func_args, **kwargs)
        if getattr(config, "val_dataloader", None) and not hasattr(config.val_evaluator, "type"):
            config.val_evaluator = self.evaluator_cfg
            config.val_cfg = {"type": "ValLoop"}
        if getattr(config, "test_dataloader", None) and not hasattr(config.test_evaluator, "type"):
            config.test_evaluator = self.evaluator_cfg
            config.test_cfg = {"type": "TestLoop"}
        if hasattr(config, "visualizer") and config.visualizer.type not in VISUALIZERS:
            config.visualizer = self.visualizer_cfg
        return config, update_check

    def predict(
        self,
        model: torch.nn.Module | (dict | str) | None = None,
        img: str | (np.ndarray | list) | None = None,
        checkpoint: str | Path | None = None,
        pipeline: dict | list | None = None,
        device: str | (torch.device | None) = None,
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

        inferencer = MMSegInferencer(model=config, weights=checkpoint, device=device)

        return inferencer(img, batch_size=batch_size, **kwargs)

    def export(
        self,
        model: torch.nn.Module | (str | Config) | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: str | None = "Segmentation",
        codebase: str | None = "mmseg",
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: str | None = None,
        device: str = "cpu",
        input_shape: tuple[int, int] | None = None,
        **kwargs,
    ) -> dict:
        """Export a PyTorch model to a specified format for deployment.

        Args:
            model (torch.nn.Module | str | Config | None): The PyTorch model to export.
            checkpoint (str | Path | None): The path to the checkpoint file to use for exporting.
            precision (str | None): The precision to use for exporting.
            task (str | None): The task for which the model is being exported. Defaults to "Segmentation".
            codebase (str | None): The codebase for the model being exported. Defaults to "mmseg".
            export_type (str): The type of export to perform. Can be one of "ONNX" or "OPENVINO". Defaults to "OPENVINO"
            deploy_config (str | None): The path to the deployment configuration file to use for exporting.
                File path only.
            device (str): The device to use for exporting. Defaults to "cpu".
            input_shape (tuple[int, int] | None): The input shape of the model being exported.
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
