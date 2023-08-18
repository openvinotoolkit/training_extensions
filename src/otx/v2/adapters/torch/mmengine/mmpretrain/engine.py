import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmpretrain import ImageClassificationInferencer, inference_model
from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry
from otx.v2.adapters.torch.mmengine.modules.utils import CustomConfig as Config
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


class MMPTEngine(MMXEngine):
    def __init__(
        self,
        work_dir: Optional[str] = None,
        config: Optional[Union[Dict, Config, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(work_dir=work_dir, config=config)
        self.registry = MMPretrainRegistry()

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, Dict, str]] = None,
        checkpoint: Union[bool, str, Path] = True,
        img: Optional[Union[str, np.ndarray, list]] = None,
        pipeline: Optional[List[Dict]] = None,
        device: Union[str, torch.device, None] = None,
        # task: Optional[str] = None,
        batch_size: int = 1,
    ) -> List[Dict]:
        from mmpretrain import inference_model
        from mmengine.model import BaseModel

        # Model config need data_pipeline of test_dataloader
        # Update pipelines
        if pipeline is None:
            from otx.v2.adapters.torch.mmengine.mmpretrain.dataset import get_default_pipeline

            pipeline = get_default_pipeline()
        config = Config({})
        if hasattr(model, "_config"):
            config = model._config
        elif "_config" in model:
            config = model["_config"]
        config["test_dataloader"] = {"dataset": {"pipeline": pipeline}}
        if isinstance(model, dict):
            model.setdefault("_config", config)
        elif isinstance(model, torch.nn.Module):
            model._config = config

        # # Check if the model can use mmpretrain's inference api.
        # if isinstance(checkpoint, Path):
        #     checkpoint = str(checkpoint)
        # if isinstance(model, BaseModel) and hasattr(model, "_metainfo") and model._metainfo.results is not None:
        #     return inference_model(
        #         model=model,
        #         pretrained=checkpoint,
        #         device=device,
        #         inputs=img,
        #         batch_size=batch_size
        #     )
        # elif task is not None and task != "Image Classification":
        #     raise NotImplementedError()
        inferencer = ImageClassificationInferencer(
            model=model,
            pretrained=checkpoint,
            device=device,
        )

        return inferencer(img, batch_size)

    def export(
        self,
        model: Optional[Union[torch.nn.Module, str, Config]] = None,
        checkpoint: Optional[str] = None,
        task: str = "Classification",
        codebase: str = "mmpretrain",
        precision: str = "float32",  # ["float16", "fp16", "float32", "fp32"]
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
            task=task,
            codebase=codebase,
            precision=precision,
            export_type=export_type,
            deploy_config=deploy_config,
            dump_features=dump_features,
            device=device,
            input_shape=input_shape,
        )
