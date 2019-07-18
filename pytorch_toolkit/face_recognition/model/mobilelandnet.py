import torch.nn as nn
import torch
import math

from model.blocks.mobilenet_v2_blocks import InvertedResidual
from model.blocks.shared_blocks import make_activation
from .common import ModelInterface

def init_block(in_channels, out_channels, stride, activation=nn.ReLU):
    """Builds the first block of the MobileFaceNet"""
    return nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        make_activation(activation)
    )

class MobileLandNet(ModelInterface):
    def __init__(self, embedding_size=128, num_classes=1, width_multiplier=1., feature=True):
        super(MobileLandNet, self).__init__()
        assert embedding_size > 0
        assert num_classes > 0
        assert width_multiplier > 0
        self.feature = feature

# Set up of inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [2, 64, 5, 2],
            [2, 128, 1, 2],
            [4, 128, 6, 1],
            [2, 128, 1, 1]
        ]

        first_channel_num = 64
        self.features = [init_block(3, first_channel_num, 2)]

        self.features.append(nn.Conv2d(first_channel_num, first_channel_num, 3, 1, 1,
                                       groups=first_channel_num, bias=False))
        self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.ReLU())

        # Inverted Residual Blocks
        in_channel_num = first_channel_num
        size_h, size_w = MobileLandNet.get_input_res()
        size_h, size_w = size_h // 2, size_w // 2
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    size_h, size_w = size_h // s, size_w // s
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          s, t, outp_size=(size_h, size_w)))
                else:
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          1, t, outp_size=(size_h, size_w)))
                in_channel_num = output_channel

        self.s1 = nn.Sequential(*self.features)
        self.fc_channel_num = 14*14*in_channel_num + 7*7*in_channel_num*2 + embedding_size
        # feature block
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel_num, 2*in_channel_num, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(2*in_channel_num),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(2*in_channel_num, embedding_size, 7, 1, 0, bias=False),
                                   nn.BatchNorm2d(embedding_size),
                                   nn.ReLU())
        self.fc_loc = nn.Linear(self.fc_channel_num, 32)
        self.init_weights()

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.conv1(s1)
        s3 = self.conv2(s2)
        # s1 = s1.view(s1.size(0), -1)
        # s2 = s2.view(s2.size(0), -1)
        # s3 = s3.view(s3.size(0), -1)
        s1 = torch.flatten(s1, start_dim=1)
        s2 = torch.flatten(s2, start_dim=1)
        s3 = torch.flatten(s3, start_dim=1)
        out = torch.cat((s1, s2, s3), dim=1)
        out = self.fc_loc(out)
        return out

    @staticmethod
    def get_input_res():
        return 112, 112

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.

    def init_weights(self):
        """Initializes weights of the model before training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
#
# def test():
#     input = torch.randint(0, 255, (2, 3, 112, 112), dtype=torch.float32)
#     model = MobileLandNet()
#     out = model.forward(input)
#     print(out)
#
# test()