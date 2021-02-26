import torch
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
        self.embed_size = encoder.features_shape[0]
        self.last_feature_size = encoder.features_shape[1]
        self.fc = nn.Linear(encoder.features_shape[0], n_classes)
        self.dropout = nn.Dropout2d(0.5)

        self.n_classes = n_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.sequence_size = sample_duration
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        # (B x T x C x H x W) -> (B*T x C x H x W)
        batch_size = images.shape[0]
        images = squash_dims(images, (0, 1))

        features = self.resnet(images)
        # features = self.dropout(features)

        features = F.avg_pool2d(features, self.last_feature_size)  # (B*T) x C
        features = unsquash_dim(features, 0, (batch_size, -1))
        ys = self.fc(features.squeeze(-1).squeeze(-1))

        return ys.mean(1)

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)


class MultiFrameBaselineEncoder(MultiFrameBaseline):
    def forward(self, images):
        batch_size = images.shape[0]
        if images.dim() == 5:
            # If input tensor contains temporal dimension, combine it with the batch one:
            # (B x T x C x H x W) -> (B*T x C x H x W)
            images = squash_dims(images, (0, 1))
        features = self.resnet(images)
        features = F.avg_pool2d(features, self.last_feature_size, 1)  # (B*T) x C
        features = features.squeeze(-1).squeeze(-1)
        features = self.fc(features)
        if images.dim() == 5:
            # Separate temporal and batch dimensions back.
            # (B*T x embd_size) -> (B x T x embd_size)
            features = unsquash_dim(features, 0, (batch_size, -1))
        return features

    def export_onnx(self, export_path):
        first_param = next(self.parameters())
        input_tensor = first_param.new_zeros(1, self.input_channels, self.input_size, self.input_size)
        with torch.no_grad():
            torch.onnx.export(self, (input_tensor,), export_path,
                              input_names=['image'], output_names=['features'],
                              verbose=True)


class MultiFrameBaselineDecoder(MultiFrameBaseline):
    def forward(self, features):
        return features.mean(1)

    def export_onnx(self, export_path):
        first_param = next(self.parameters())
        input_tensor = first_param.new_zeros(1, self.sequence_size, self.n_classes)
        with torch.no_grad():
            torch.onnx.export(self, (input_tensor,), export_path,
                              input_names=['features'], output_names=['logits'],
                              verbose=True)
