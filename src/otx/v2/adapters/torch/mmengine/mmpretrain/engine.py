"""OTX adapters.torch.mmengine.mmpretrain.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.adapters.torch.mmengine.utils.runner_config import get_value_from_config
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger()


class MMPTEngine(MMXEngine):
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
        self.registry = MMPretrainRegistry()

    def _update_evaluator(self, evaluator: list | dict | None, num_classes: int) -> list | dict | None:
        if evaluator is None or not evaluator:
            evaluator = [{"type": "Accuracy"}]
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

    def _update_config(self, func_args: dict, **kwargs) -> bool:
        update_check = super()._update_config(func_args, **kwargs)
        num_classes = -1
        model = self.config.get("model", {})
        if isinstance(model, torch.nn.Module):
            head = model.head if hasattr(model, "head") else None
            num_classes = head.num_classes if hasattr(head, "num_classes") else -1
        else:
            head = model.get("head", {})
            num_classes = head.get("num_classes", -1)
        for subset in ("val", "test"):
            if f"{subset}_dataloader" in self.config and self.config[f"{subset}_dataloader"] is not None:
                evaluator = get_value_from_config(f"{subset}_evaluator", func_args, config=self.config)
                self.config[f"{subset}_evaluator"] = self._update_evaluator(
                    evaluator=evaluator, num_classes=num_classes,
                )

        return update_check


    def predict(
        self,
        model: torch.nn.Module | (dict | str) | None = None,
        img: str | (np.ndarray | list) | None = None,
        checkpoint: str | Path | None = None,
        pipeline: dict | list | None = None,
        device: str | (torch.device | None) = None,
        task: str | None = None,
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
        from mmengine.model import BaseModel
        from mmpretrain import ImageClassificationInferencer, inference_model

        # Model config need data_pipeline of test_dataloader
        # Update pipelines
        if pipeline is None:
            from otx.v2.adapters.torch.mmengine.mmpretrain.dataset import get_default_pipeline

            pipeline = get_default_pipeline()
        config = Config({})
        if isinstance(model, torch.nn.Module) and hasattr(model, "_config"):
            config = model._config  # noqa: SLF001
        elif isinstance(model, dict) and "_config" in model:
            config = model["_config"]
        config["test_dataloader"] = {"dataset": {"pipeline": pipeline}}
        if isinstance(model, dict):
            model.setdefault("_config", config)
        elif isinstance(model, torch.nn.Module):
            model._config = config  # noqa: SLF001

        # Check if the model can use mmpretrain's inference api.
        if isinstance(checkpoint, Path):
            checkpoint = str(checkpoint)
        metainfo = getattr(model, "_metainfo", None)
        if isinstance(model, BaseModel) and metainfo is not None and metainfo.results is not None:
            task = next(result.task for result in metainfo.results)
            inputs = {
                "model": model,
                "pretrained": checkpoint,
                "device": device,
                "inputs": img,
                "batch_size": batch_size,
            }
            if task in ("Image Caption", "Visual Grounding", "Visual Question Answering"):
                inputs["images"] = inputs.pop("inputs")
            return [inference_model(**inputs, **kwargs)]
        if task is not None and task != "Image Classification":
            raise NotImplementedError
        inferencer = ImageClassificationInferencer(
            model=model,
            pretrained=checkpoint,
            device=device,
        )

        return inferencer(img, batch_size, **kwargs)

    def export(
        self,
        model: torch.nn.Module | (str | Config) | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: str | None = "Classification",
        codebase: str | None = "mmpretrain",
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
