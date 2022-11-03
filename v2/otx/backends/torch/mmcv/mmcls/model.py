from otx.backends.torch.model import TorchModel

class ModelAdapter(TorchModel):
    def __init__(self, model_config):
        super().__init__(model_config)
