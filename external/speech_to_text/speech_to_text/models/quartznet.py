# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn as nn
from .utils import init_weights


class GroupShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        size = x.size()
        channels_per_group = int(size[1]) // self.groups
        x = x.view(-1, self.groups, channels_per_group, size[-1])
        x = x.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.groups * channels_per_group, size[-1])
        return x


class DWConv1d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=11, stride=1, dilation=1, padding=0, bias=False, groups=1):
        super().__init__()
        self.dw = nn.Conv1d(channels_in, channels_in, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, groups=channels_in)
        self.pw = nn.Conv1d(channels_in, channels_out, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, groups=groups)
        self.shuffle = GroupShuffle(groups) if groups > 1 else nn.Sequential()

    def forward(self, x):
        return self.shuffle(self.pw(self.dw(x)))


class QNetSubBlock(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size=11,
            stride=1,
            dilation=1,
            padding=0,
            bias=False,
            groups=1,
            dropout_prob=0.2,
            conv_type=DWConv1d
    ):
        super().__init__()
        self.conv = conv_type(
            channels_in,
            channels_out,
            kernel_size = kernel_size,
            stride = stride,
            dilation = dilation,
            padding = padding,
            bias = bias,
            groups = groups
        )
        self.norm = nn.BatchNorm1d(channels_out, eps=1e-3, momentum=0.1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Sequential()

    def forward(self, x, residual=None):
        x = self.conv(x)
        x = self.norm(x)
        if residual is not None:
             x = x + residual
        x = self.act(x)
        x = self.dropout(x)
        return x


class QNetBlock(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size=11,
            stride=1,
            dilation=1,
            padding=None,
            bias=False,
            groups=1,
            dropout_prob=0.2,
            repeat=5,
            separable=True,
            **kwargs
    ):
        super().__init__()
        padding = (kernel_size * dilation - 1) // 2 if padding is None else padding
        # residual connection
        self.residual = None
        if repeat > 1:
            self.residual = nn.Sequential(
                nn.Conv1d(channels_in, channels_out, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels_out, eps=1e-3, momentum=0.1)
            )
        # sub-blocks
        conv_type = DWConv1d if separable else nn.Conv1d
        self.layers = nn.ModuleList()
        for i in range(repeat):
            self.layers.append(
                QNetSubBlock(
                    channels_in = channels_in,
                    channels_out = channels_out,
                    kernel_size = kernel_size,
                    stride = stride,
                    dilation = dilation,
                    padding = padding,
                    bias = bias,
                    groups = groups,
                    dropout_prob = dropout_prob,
                    conv_type = conv_type
                )
            )
            channels_in = channels_out

    def forward(self, x):
        residual = None if self.residual is None else self.residual(x)
        for m in self.layers[:-1]:
            x = m(x)
        return self.layers[-1](x, residual)


class QuartzNet(nn.Module):
    def __init__(self, n_mels, vocab_size, cfg, output_softmax=False):
        super().__init__()
        # params of tne model
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.output_softmax = output_softmax
        # build model
        self.stride = 1
        layers = []
        channels_in = n_mels
        for params in cfg:
            for b in range(params["n_blocks"]):
                self.stride *= params["stride"]
                params["channels_in"] = channels_in
                layers.append(
                    QNetBlock(**params)
                )
                channels_in = params["channels_out"]
        self.layers = nn.Sequential(*layers)
        self.predictor = nn.Conv1d(channels_in, vocab_size, kernel_size=1, bias=True)
        self.apply(lambda x: init_weights(x))

    def forward(self, x):
        x = self.layers(x)
        x = self.predictor(x)
        x = x.permute(0, 2, 1)
        if self.output_softmax:
            x = x.softmax(-1)
        return x
