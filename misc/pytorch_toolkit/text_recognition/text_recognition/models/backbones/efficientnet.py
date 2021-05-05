import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d, round_filters


class EfficientNetLikeBackbone(nn.Module):
    def __init__(self, model_name, in_channels=1, **kwargs):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        if in_channels != 3:
            out_channels = round_filters(32, self.model._global_params)
            self.model._conv_stem = get_same_padding_conv2d(image_size=self.model._global_params.image_size)(
                in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        # Final linear layer
        del self.model._avg_pooling
        if self.model._global_params.include_top:
            del self.model._dropout
            del self.model._fc

    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        return x
