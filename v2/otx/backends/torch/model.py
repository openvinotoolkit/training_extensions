import torch
import timm
from otx.core.model import IModel
from otx.utils.config import Config
from otx.utils.logger import get_logger
# from torch import nn

logger = get_logger()


class TorchModel(IModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)

    def build(self):
        return TorchModel.load_from_hub(self.config.hub, self.config.model)

    @staticmethod
    def load_from_hub(hub, model, pretrained=True, **kwargs):
        if hub == "timm":
            model = timm.create_model(model, pretrained=pretrained)
        elif hub == "pytorch/vision":
            model = torch.hub.load(hub, model, **kwargs)
        else:
            raise ValueError(f"not supported model hub repo {hub_path}")
        return model

    def save(self):
        pass

    def export(self):
        pass

    @staticmethod
    def rand(size):
        return torch.rand(size)