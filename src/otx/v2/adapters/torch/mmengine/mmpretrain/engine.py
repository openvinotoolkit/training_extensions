"""OTX adapters.torch.mmengine.mmpretrain.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig as Config
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


class MMPTEngine(MMXEngine):
    def __init__(
        self,
        work_dir: Optional[Union[str, Path]] = None,
        config: Optional[Union[Dict, Config, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(work_dir=work_dir, config=config)
        self.registry = MMPretrainRegistry()

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, Dict, str]] = None,
        img: Optional[Union[str, np.ndarray, list]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        pipeline: Optional[Union[Dict, List]] = None,
        device: Union[str, torch.device, None] = None,
        task: Optional[str] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> List[Dict]:
        from mmengine.model import BaseModel
        from mmpretrain import ImageClassificationInferencer, inference_model

        # Model config need data_pipeline of test_dataloader
        # Update pipelines
        if pipeline is None:
            from otx.v2.adapters.torch.mmengine.mmpretrain.dataset import get_default_pipeline

            pipeline = get_default_pipeline()
        config = Config({})
        if hasattr(model, "_config"):
            config = getattr(model, "_config")
        elif isinstance(model, dict) and "_config" in model:
            config = model["_config"]
        config["test_dataloader"] = {"dataset": {"pipeline": pipeline}}
        if isinstance(model, dict):
            model.setdefault("_config", config)
        elif isinstance(model, torch.nn.Module):
            model._config = config

        # Check if the model can use mmpretrain's inference api.
        if isinstance(checkpoint, Path):
            checkpoint = str(checkpoint)
        if (
            isinstance(model, BaseModel)
            and getattr(model, "_metainfo") is not None
            and getattr(model._metainfo, "results") is not None
        ):
            task = [result.task for result in model._metainfo.results][0]
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
        elif task is not None and task != "Image Classification":
            raise NotImplementedError()
        inferencer = ImageClassificationInferencer(
            model=model,
            pretrained=checkpoint,
            device=device,
        )

        return inferencer(img, batch_size, **kwargs)

    def export(
        self,
        model: Optional[Union[torch.nn.Module, str, Config]] = None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = "float32",  # ["float16", "fp16", "float32", "fp32"]
        task: Optional[str] = "Classification",
        codebase: Optional[str] = "mmpretrain",
        export_type: str = "OPENVINO",  # "ONNX" or "OPENVINO"
        deploy_config: Optional[str] = None,  # File path only?
        dump_features: bool = False,  # TODO
        device: str = "cpu",
        input_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        return super().export(
            model=model,
            checkpoint=checkpoint,
            precision=precision,
            task=task,
            codebase=codebase,
            export_type=export_type,
            deploy_config=deploy_config,
            dump_features=dump_features,
            device=device,
            input_shape=input_shape,
            **kwargs,
        )
