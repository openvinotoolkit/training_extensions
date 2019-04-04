from torch import nn as nn
from torch.nn import functional as F

from ..utils import get_fine_tuning_parameters
from .backbone import make_encoder
from .modules import squash_dims, unsquash_dim


class MultiFrameBaseline(nn.Module):
    """Simple baseline that runs a classifier on each frame independently and averages logits."""

    def __init__(self, sample_duration, encoder='resnet34', n_classes=400, input_size=224, pretrained=True,
                 input_channels=3):
        """Average prediction over multiple frames"""
        super().__init__()

        # backbone
        encoder = make_encoder(encoder, input_size=input_size, input_channels=input_channels, pretrained=pretrained)
        self.resnet = encoder.features  # name is kept for compatibility with older checkpoints
        self.last_feature_size = encoder.features_shape[1]
        self.fc = nn.Linear(encoder.features_shape[0], n_classes)
        self.dropout = nn.Dropout2d(0.5)

        self.sequence_size = sample_duration
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        # (B x T x C x H x W) -> (B*T x C x H x W)
        images = squash_dims(images, (0, 1))

        features = self.resnet(images)
        # features = self.dropout(features)

        features = F.avg_pool2d(features, self.last_feature_size)  # (B*T) x C
        features = unsquash_dim(features, 0, (-1, self.sequence_size))
        ys = self.fc(features.squeeze(-1).squeeze(-1))

        return ys.mean(1)

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)
