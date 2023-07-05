import torch
from mmcls.apis import inference_model
from otx.v2.adapters.torch.mmcv.engine import MMXEngine
from otx.v2.adapters.torch.mmcv.mmcls.registry import MMCLSRegistry
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


class MMCLSEngine(MMXEngine):
    def __init__(
        self,
        **params,
    ) -> None:
        super().__init__()
        self.module_registry = MMCLSRegistry()

    def infer(self, model: torch.nn.Module, img):
        return inference_model(model, img)

    def evaluate(self, **params):
        raise NotImplementedError()
