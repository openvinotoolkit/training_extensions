"""OTX adapters.torch.mmengine.mmdet.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmdet.registry import MMDetRegistry
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger()


class MMDetEngine(MMXEngine):
    """The MMDetEngine class is responsible for running inference on pre-trained models."""

    def __init__(
        self,
        work_dir: str | Path | None = None,
        config: dict | (Config | str) | None = None,
    ) -> None:
        """Initialize a new instance of the MMDetEngine class.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[Dict, Config, str]], optional): The configuration for the engine. Defaults to None.
        """
        super().__init__(work_dir=work_dir, config=config)
        self.registry = MMDetRegistry()

    def _update_config(self, func_args: dict, **kwargs) -> bool:
        update_check = super()._update_config(func_args, **kwargs)
        if hasattr(self.config, "val_dataloader") and not hasattr(self.config.val_evaluator, "type"):
            self.config.val_evaluator = {"type": "OTXDetMetric", "metric": "mAP"}
        if hasattr(self.config, "test_dataloader") and not hasattr(self.config.test_evaluator, "type"):
            self.config.test_evaluator = {"type": "OTXDetMetric", "metric": "mAP"}
        self.config.default_hooks.checkpoint.save_best = "pascal_voc/mAP"
        max_epochs = getattr(self.config.train_cfg, "max_epochs", None)
        if max_epochs:
            for scheduler in self.config.param_scheduler:
                if hasattr(scheduler, "end") and scheduler.end > max_epochs:
                    scheduler.end = max_epochs
                    if hasattr(scheduler, "begin") and scheduler.begin > scheduler.end:
                        scheduler.begin = scheduler.end
                if hasattr(scheduler, "begin") and scheduler.begin > max_epochs:
                    scheduler.begin = max_epochs - 1
        return update_check

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
        import cv2
        from mmcv.transforms import Compose
        from mmdet.apis import DetInferencer, inference_detector
        from mmengine.model import BaseModel

        # Model config need data_pipeline of test_dataloader
        # Update pipelines
        if pipeline is None:
            pipeline = [
                {"type": "Resize", "scale": [512, 512]},
                {"type": "PackDetInputs", "meta_keys": ['img', 'img_id', 'scale', 'img_shape', 'scale_factor']},
            ]

        cfg = Config({})
        if isinstance(model, torch.nn.Module) and hasattr(model, "cfg"):
            cfg = model.cfg
        elif isinstance(model, dict) and "cfg" in model:
            cfg = model["cfg"]
        cfg["test_dataloader"] = {"dataset": {"pipeline": pipeline}}
        if isinstance(model, dict):
            model.setdefault("cfg", cfg)
        elif isinstance(model, torch.nn.Module):
            model.cfg = cfg

        # Check if the model can use mmdet's inference api.
        if isinstance(checkpoint, Path):
            checkpoint = str(checkpoint)
        if isinstance(model, BaseModel):
            if isinstance(img, str):
                img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
            inputs = {
                "model": model,
                "imgs": img,
                "test_pipeline": Compose(pipeline),
            }
            return [inference_detector(**inputs, **kwargs)]
        inferencer = DetInferencer(
            model=model,
            weights=checkpoint,
            device=device,
        )

        return inferencer(img, batch_size, **kwargs)

    def export(
        self,
        model: torch.nn.Module | (str | Config) | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: str | None = "ObjectDetection",
        codebase: str | None = "mmdet",
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
            codebase (Optional[str]): The codebase for the model being exported. Defaults to "mmdet".
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

    def _update_codebase_config(self, codebase: str | None, task: str | None, deploy_config_dict: dict) -> None:
        """Update specific codebase config.

        Args:
            codebase(str): mmX codebase framework
            task(str): mmdeploy task
            deploy_config_dict(dict): Config dict for deployment
        """
        codebase = codebase if codebase is not None else self.registry.name
        codebase_config = {
            "type": codebase,
            "task": task,
            "post_processing": {
                "score_threshold": 0.05,
                "confidence_threshold": 0.005,  # for YOLOv3
                "iou_threshold": 0.5,
                "max_output_boxes_per_class": 200,
                "pre_top_k": 5000,
                "keep_top_k": 100,
                "background_label_id": -1,
            },
        }
        deploy_config_dict["codebase_config"] = codebase_config
