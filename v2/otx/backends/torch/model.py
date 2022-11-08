import torch
import timm
from otx.core.model import IModel
from otx.utils.logger import get_logger
# from torch import nn

logger = get_logger()


class TorchModelAdapter(IModel):
    def build(self, **kwargs):
        pretrained = True
        if self._ckpt is not None:
            pretrained = False
        model = self.load_from_hub(self.config.hub, self.config.model, pretrained=pretrained)
        if self._ckpt is not None:
            model.load_state_dict(self.load_weights_from_ckpt(self._ckpt))
        return model

    def save(self, model, path, **kwargs):
        torch.save(model, path)

    def export(self, model, path, exp_type):
        pass

    def update_config(self, config: dict):
        self._config.merge_from_dict(config)

    @staticmethod
    def load_from_hub(hub, model, pretrained=True, **kwargs):
        if hub == "timm":
            model = timm.create_model(model, pretrained=pretrained)
        elif hub == "pytorch/vision":
            model = torch.hub.load(hub, model, **kwargs)
        else:
            raise ValueError(f"not supported model hub repo {hub}")
        return model

    @staticmethod
    def load_weights_from_ckpt(ckpt):
        return torch.load(ckpt)

    @staticmethod
    def rand(size):
        return torch.rand(size)
