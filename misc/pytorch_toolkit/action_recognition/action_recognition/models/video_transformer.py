import torch
from torch import nn as nn
from torch.nn import functional as F

from .backbone import make_encoder
from .modules import Identity, squash_dims, unsquash_dim
from .modules.self_attention import DecoderBlock, PositionEncoding
from ..utils import get_fine_tuning_parameters, load_state


class VideoTransformer(nn.Module):
    def __init__(self, embed_size, sequence_size, encoder='resnet34', n_classes=400, input_size=224, pretrained=True,
                 input_channels=3, num_layers=4, layer_norm=True):
        super().__init__()

        # backbone
        encoder = make_encoder(encoder, input_size=input_size, pretrained=pretrained, input_channels=input_channels)
        self.resnet = encoder.features  # name is kept for compatibility with older checkpoints
        self.last_feature_size = encoder.features_shape[1]
        self.embed_size = embed_size

        if encoder.features_shape[0] != embed_size:
            self.reduce_conv = nn.Conv2d(encoder.features_shape[0], embed_size, 1)
        else:
            self.reduce_conv = Identity()

        self.sequence_size = sequence_size

        self.self_attention_decoder = SelfAttentionDecoder(embed_size, embed_size, [8] * num_layers,
                                                           sequence_size, layer_norm=layer_norm)
        self.fc = nn.Linear(embed_size, n_classes)
        self.dropout = nn.Dropout2d(0.8)

        self.init_weights()
        self.input_channels = input_channels
        self.input_size = input_size

    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, rgb_clip):
        """Extract the image feature vectors."""
        # (B x T x C x H x W) -> (B*T x C x H x W)
        rgb_clip = squash_dims(rgb_clip, (0, 1))

        features = self.resnet(rgb_clip)
        features = self.reduce_conv(features)

        features = F.avg_pool2d(features, 7)  # (B*T) x C
        features = unsquash_dim(features, 0, (-1, self.sequence_size))
        ys = self.self_attention_decoder(features[..., 0, 0])
        # ys = self.dropout(ys)
        ys = self.fc(ys)

        return ys.mean(1)

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)

    def load_checkpoint(self, state_dict):
        load_state(self, state_dict, 'fc')


class VideoTransformerEncoder(VideoTransformer):
    def forward(self, rgb_frame):
        features = self.resnet(rgb_frame)
        features = self.reduce_conv(features)
        features = F.avg_pool2d(features, 7)
        return features

    def export_onnx(self, export_path):
        first_param = next(self.parameters())
        input_tensor = first_param.new_zeros(1, self.input_channels, self.input_size, self.input_size)
        with torch.no_grad():
            torch.onnx.export(self, (input_tensor,), export_path, verbose=True)


class VideoTransformerDecoder(VideoTransformer):
    def forward(self, features):
        ys = self.self_attention_decoder(features)
        ys = self.fc(ys)
        return ys.mean(1)

    def export_onnx(self, export_path):
        first_param = next(self.parameters())
        input_tensor = first_param.new_zeros(1, self.sequence_size, self.embed_size)
        with torch.no_grad():
            torch.onnx.export(self, (input_tensor,), export_path, verbose=True)


class SelfAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, sequence_size, inner_hidden_factor=2, layer_norm=True):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.layers = nn.ModuleList([
            DecoderBlock(inp_size, hid_size, hid_size * inner_hidden_factor, n_head, hid_size // n_head,
                         hid_size // n_head, layer_norm=layer_norm)
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, x):
        outputs, attentions = [], []
        b, t, c = x.size()
        x = self.position_encoding(x)

        for layer in self.layers:
            x, attn = layer(x)

            outputs.append(x)
        return x
