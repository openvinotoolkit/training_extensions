from efficientnet_pytorch import EfficientNet

import torch.nn as nn


class EfficientNetLikeBackbone(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        # Final linear layer
        del self.model._avg_pooling
        if self.model._global_params.include_top:
            del self.model._dropout
            del self.model._fc

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        return x
