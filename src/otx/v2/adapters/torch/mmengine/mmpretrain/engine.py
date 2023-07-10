from typing import Dict, List, Optional, Union

import numpy as np
import torch
from mmpretrain import ImageClassificationInferencer
from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry
from otx.v2.api.utils.logger import get_logger

from mmengine.config import Config

logger = get_logger()


class MMPTEngine(MMXEngine):
    def __init__(
        self,
        config: Optional[Union[Dict, Config, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config)
        self.module_registry = MMPretrainRegistry()
        self.base_runner = self.module_registry.get("Runner")

    def predict(
        self,
        model: Optional[Union[torch.nn.Module, Dict, str]] = None,
        img: Optional[Union[str, np.ndarray, list]] = None,
        pipeline: Optional[List[Dict]] = None,
    ):
        # Update pipelines
        if pipeline is None:
            from otx.v2.adapters.torch.mmengine.mmpretrain.dataset import get_default_pipeline

            pipeline = get_default_pipeline()

        if not hasattr(model, "_config"):
            temp_config = {"test_dataloader": {"dataset": {"pipeline": pipeline}}}
            if isinstance(model, dict):
                model.setdefault("_config", Config(temp_config))
            elif isinstance(model, torch.nn.Module):
                model._config = Config(temp_config)
        inferencer = ImageClassificationInferencer(model=model)

        return inferencer(img)
