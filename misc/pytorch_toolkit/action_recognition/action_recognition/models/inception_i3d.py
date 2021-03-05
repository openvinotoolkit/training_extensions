import torch
from torch import nn
from torch.nn import functional as F

from ..utils import get_fine_tuning_parameters


def calc_same_padding(kernel_shape, stride=None):
    if stride is None:
        stride = (1,) * len(kernel_shape)
    return [(ks - 1) // 2 for ks, st in zip(kernel_shape, stride)]


def pad_same(input, kernel_size, stride=(1, 1, 1), dilation=(1, 1, 1), value=0):
    t_left, t_right = get_pad_value(input.size(2), kernel_size[0], stride[0], dilation[0])
    rows_left, rows_right = get_pad_value(input.size(3), kernel_size[1], stride[1], dilation[1])
    cols_left, cols_right = get_pad_value(input.size(4), kernel_size[2], stride[2], dilation[2])

    input = F.pad(input, (cols_left, cols_right, rows_left, rows_right, t_left, t_right), value=value)
    return input


def get_pad_value(input_size, filter_size, stride, dilation):
    effective_filter_size = (filter_size - 1) * dilation + 1
    out_size = (input_size + stride - 1) // stride
    padding_needed = max(0, (out_size - 1) * stride + effective_filter_size - input_size)

    padding_left = padding_needed // 2
    padding_right = (padding_needed - 1) // 2 + 1
    return padding_left, padding_right


class Unit3d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), use_batch_norm=True,
                 use_bias=False, use_relu=True, padding_valid=True):
        super().__init__()

        self.conv_3d = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_shape, stride=stride,
                                 padding=calc_same_padding(kernel_shape, stride) if not padding_valid else 0,
                                 bias=use_bias)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm3d(output_channels)

            # self.batch_norm.weight.data.ones_()
        if use_relu:
            self.relu = nn.ReLU()

        self.use_batch_norm = use_batch_norm
        self.use_relu = use_relu
        self.padding_valid = padding_valid

    def forward(self, x):
        x = self.conv_3d(pad_same(x, self.conv_3d.kernel_size, self.conv_3d.stride) if self.padding_valid else x)
        first_conv = x
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_relu:
            x = self.relu(x)

        return x


class InceptionBlock(nn.Module):
    def __init__(self, input_channels, branch_channels):
        super().__init__()

        self.Branch_0_Conv3d_0a_1x1 = Unit3d(input_channels, branch_channels[0], kernel_shape=(1, 1, 1))

        self.Branch_1_Conv3d_0a_1x1 = Unit3d(input_channels, branch_channels[1], kernel_shape=(1, 1, 1))
        self.Branch_1_Conv3d_0b_3x3 = Unit3d(branch_channels[1], branch_channels[2], kernel_shape=(3, 3, 3))

        self.Branch_2_Conv3d_0a_1x1 = Unit3d(input_channels, branch_channels[3], kernel_shape=(1, 1, 1))
        self.Branch_2_Conv3d_0b_3x3 = Unit3d(branch_channels[3], branch_channels[4], kernel_shape=(3, 3, 3))

        self.Branch_3_MaxPool3d_0a_3x3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                                      padding=0)
        self.Branch_3_Conv3d_0b_1x1 = Unit3d(input_channels, branch_channels[5], kernel_shape=(1, 1, 1))

    def forward(self, x, endpoint=False):
        branch0 = self.Branch_0_Conv3d_0a_1x1(x)

        branch1 = self.Branch_1_Conv3d_0a_1x1(x)
        branch1 = self.Branch_1_Conv3d_0b_3x3(branch1)

        branch2 = self.Branch_2_Conv3d_0a_1x1(x)
        branch2 = self.Branch_2_Conv3d_0b_3x3(branch2)

        branch3 = self.Branch_3_MaxPool3d_0a_3x3(
            pad_same(x, self.Branch_3_MaxPool3d_0a_3x3.kernel_size, self.Branch_3_MaxPool3d_0a_3x3.stride, value=-999))
        branch3 = self.Branch_3_Conv3d_0b_1x1(branch3)
        inner_endpoint = branch3

        if not endpoint:
            return torch.cat((branch0, branch1, branch2, branch3), dim=1)
        else:
            return torch.cat((branch0, branch1, branch2, branch3), dim=1), inner_endpoint


class InceptionI3D(nn.Module):
    def __init__(self, input_channels=3, num_classes=400, pretrain=False, dropout_rate=0.):
        super().__init__()

        assert pretrain is False, "Pretrain is not implemented"

        self.Conv3d_1a_7x7 = Unit3d(input_channels, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2), padding_valid=True)

        self.MaxPool3d_2a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                             padding=0)
        self.Conv3d_2b_1x1 = Unit3d(64, 64, kernel_shape=(1, 1, 1))
        self.Conv3d_2c_3x3 = Unit3d(64, 192, kernel_shape=(3, 3, 3))

        self.MaxPool3d_3a_3x3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                             padding=0)
        self.Mixed_3b = InceptionBlock(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionBlock(256, [128, 128, 192, 32, 96, 64])

        self.MaxPool3d_4a_3x3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                             padding=0)
        self.Mixed_4b = InceptionBlock(480, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionBlock(512, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionBlock(512, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionBlock(512, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionBlock(528, [256, 160, 320, 32, 128, 128])

        self.MaxPool3d_5a_2x2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                             padding=0)
        self.Mixed_5b = InceptionBlock(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionBlock(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout3d(dropout_rate)
        self.logits = Unit3d(1024, num_classes, kernel_shape=(1, 1, 1), use_batch_norm=False, use_relu=False,
                             use_bias=True)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.Conv3d_1a_7x7(x)

        x = self.MaxPool3d_2a_3x3(pad_same(x, self.MaxPool3d_2a_3x3.kernel_size, self.MaxPool3d_2a_3x3.stride))
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)

        x = self.MaxPool3d_3a_3x3(pad_same(x, self.MaxPool3d_3a_3x3.kernel_size, self.MaxPool3d_3a_3x3.stride))
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)

        x = self.MaxPool3d_4a_3x3(pad_same(x, self.MaxPool3d_4a_3x3.kernel_size, self.MaxPool3d_4a_3x3.stride))
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)

        x = self.MaxPool3d_5a_2x2(pad_same(x, self.MaxPool3d_5a_2x2.kernel_size, self.MaxPool3d_5a_2x2.stride))
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        logits = self.logits(x)

        result = logits.mean(dim=2).squeeze(-1).squeeze(-1)
        return result

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)
