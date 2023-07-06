import torch
from mmpretrain.apis import inference_model
from otx.v2.adapters.torch.mmengine.engine import MMXEngine
from otx.v2.adapters.torch.mmengine.mmpretrain.registry import MMPretrainRegistry
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


class MMPTEngine(MMXEngine):
    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()
        self.module_registry = MMPretrainRegistry()

    def infer(self, model: torch.nn.Module, img):
        return inference_model(model, img)

    def evaluate(self, **params):
        raise NotImplementedError()
