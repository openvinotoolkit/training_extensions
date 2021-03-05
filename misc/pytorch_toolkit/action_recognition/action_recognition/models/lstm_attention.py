from torch import nn
from torch.nn import functional as F
from torchvision import models as models

from ..utils import get_fine_tuning_parameters, load_state
from .backbone import make_encoder
from .modules import (Attention, AttentionLSTM, StateInitZero, squash_dims,
                      unsquash_dim)


class VisualAttentionLSTM(nn.Module):
    """LSTM architecture with attention mechanism (https://arxiv.org/pdf/1511.04119.pdf)"""

    def __init__(self, embed_size, sequence_size, encoder='resnet34', n_classes=400, input_size=224, pretrained=True,
                 use_attention=False, num_layers=1, bidirectional=False):
        super().__init__()
        self.use_attention = use_attention

        # backbone
        encoder = make_encoder(encoder, input_size=input_size, pretrained=pretrained)
        self.resnet = encoder.features  # name is kept for compatibility with older checkpoints

        self.dropout = nn.Dropout(p=0.5)

        # self.state_init = StateInitFC(resnet1_channel_size, embed_size)
        bidirectional_mult = 2 if bidirectional else 1
        self.state_init = StateInitZero(embed_size, num_layers=num_layers * bidirectional_mult, batch_first=True)

        if use_attention:
            self.lstm = AttentionLSTM(encoder.features_shape[0], embed_size, encoder.features_shape[1] ** 2,
                                      batch_first=True, num_layers=num_layers, dropout=0.2)
        else:
            self.lstm = nn.LSTM(encoder.features_shape[0], embed_size, num_layers=num_layers, dropout=0.2,
                                batch_first=False, bidirectional=True)

        self.fc = nn.Linear(bidirectional_mult * embed_size, n_classes)
        self.out_dropout = nn.Dropout(0.5)

        self.embed_size = embed_size
        self.sequence_size = sequence_size
        self.last_feature_size = encoder.features_shape[1]

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
        features = self.dropout(features)

        features = unsquash_dim(features, 0, (-1, self.sequence_size))
        hx, cx = self.state_init(features)
        # early_features = early_features.transpose(0, 1)  # to T x B x H x W

        # no attention
        if not self.use_attention:
            features = F.avg_pool2d(squash_dims(features, (0, 1)), 7)
            features = unsquash_dim(features, 0, (-1, self.sequence_size))
            features = features.squeeze(-1).squeeze(-1).transpose(0, 1)

        ys, hidden = self.lstm(features, (hx, cx))
        ys = ys.transpose(0, 1)

        ys = self.fc(ys)
        ys = ys.mean(1)
        return ys

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)

    def load_checkpoint(self, state_dict):
        load_state(self, state_dict, 'fc')


class ResnetAttSingleInput(nn.Module):
    """ONNX Exportable variant of the LSTM-Attenion model"""

    def __init__(self, embed_size, sequence_size, n_classes=400, input_size=224, pretrained=True, resnet_size=50):
        """Load the pretrained ResNet and replace top fc layer."""
        super().__init__()

        # backbone
        resnet_cls = getattr(models, "resnet{}".format(resnet_size))
        resnet_model = resnet_cls(pretrained=pretrained)

        modules = list(resnet_model.children())[:-2]  # delete the last fc layer.
        self.resnet1 = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p=0.5)

        resnet1_channel_size = resnet_model.fc.in_features
        resnet1_spatial_size = input_size // 32
        self.last_feature_size = resnet1_spatial_size
        self.embed_size = embed_size
        self.sequence_size = sequence_size

        num_layers = 1
        self.attn = Attention(embed_size, None, self.last_feature_size * self.last_feature_size)
        self.lstm = nn.LSTM(resnet1_channel_size, embed_size, num_layers=num_layers, dropout=0.2, batch_first=False)

        self.fc = nn.Linear(embed_size, n_classes)
        self.out_dropout = nn.Dropout(0.5)

    def forward(self, images, hx, cx):
        """Extract the image feature vectors."""
        # (B x T x C x H x W) -> (B*T x C x H x W)
        # images = squash_dims(images, (0, 1))
        features = self.resnet1(images)

        # features = unsquash_dim(features, 0, (-1, self.sequence_size))
        features = unsquash_dim(features, 0, (-1, 1))
        v = squash_dims(features[0].transpose(1, 0), (2, 3))
        feature, attention = self.attn(hx[0], v, v)
        feature = feature.permute((1, 0))

        ys, (hx, cx) = self.lstm(feature.unsqueeze(0), (hx, cx))
        ys = self.fc(ys)
        ys = ys.mean(1)
        return ys, hx, cx

    def trainable_parameters(self):
        return get_fine_tuning_parameters(self)
