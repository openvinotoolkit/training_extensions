# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import math

import torch.nn as nn
import torch

from .attention import LayerNorm


class NormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, i, activation=None, batch_norm=False):
        super().__init__()

        kernel_size = 2 * i + 1
        dilation = int(math.sqrt(i + 1))
        padding = i * dilation

        if batch_norm:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                  dilation=(dilation, 1), stride=1, padding=(padding, 0), bias=False)
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.utils.weight_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                          dilation=(dilation, 1), stride=1, padding=(padding, 0), bias=False))
            self.norm = None

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.activation:
            x = self.activation(x)
        return x

    def remove_weight_norm(self):
        if self.norm is None:
            nn.utils.remove_weight_norm(self.conv)


class HighwayNetworkConv(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Conv1d(size, size, 1)
        self.W2 = nn.Conv1d(size, size, 1)
        self.W1.bias.data.fill_(0.)
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * self.act(x1) + (1. - g) * x

        return y


class Residual2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k, batch_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.bath_norm = batch_norm

        if batch_norm:
            for i in range(1, k + 1):
                self.convs.append(NormConv2d(hidden_channels, hidden_channels, i * 2 + 1, nn.PReLU(init=0.2), batch_norm=True))
                self.shortcuts.append(nn.Conv2d(hidden_channels, hidden_channels, 1))

            self.conv_first = nn.Conv2d(in_channels, hidden_channels, 1)
            self.conv_last = nn.Conv2d(hidden_channels, out_channels, 1)
        else:
            for i in range(1, k + 1):
                self.convs.append(NormConv2d(hidden_channels, hidden_channels, i, nn.LeakyReLU(0.2)))
                self.shortcuts.append(nn.utils.weight_norm(nn.Conv2d(hidden_channels, hidden_channels, 1)))

            self.conv_first = nn.utils.weight_norm(nn.Conv2d(in_channels, hidden_channels, 1))
            self.conv_last = nn.utils.weight_norm(nn.Conv2d(hidden_channels, out_channels, 1))

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x_mask = x_mask.unsqueeze(1)
        x = self.conv_first(x)
        for m1, m2 in zip(self.shortcuts, self.convs):
            x = m1(x * x_mask) + m2(x * x_mask)
        x = self.conv_last(x * x_mask)
        x = x.squeeze(1)
        return x

    def remove_weight_norm(self):
        if self.bath_norm:
            return
        nn.utils.remove_weight_norm(self.conv_first)
        nn.utils.remove_weight_norm(self.conv_last)

        for layer in self.shortcuts:
            nn.utils.remove_weight_norm(layer)

        for layer in self.convs:
            layer.remove_weight_norm()


class ResStack(nn.Module):
    def __init__(self, in_channel, out_channels, K, use_layer_norm=False):
        super(ResStack, self).__init__()

        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(in_channel, in_channel,
                              kernel_size=2 * i + 1,
                              dilation=int(math.sqrt(i + 1)),
                              padding=i * int(math.sqrt(i + 1))),
                    LayerNorm(in_channel),
                    nn.PReLU(init=0.2),
                    nn.Conv1d(in_channel, out_channels, kernel_size=1),
                    LayerNorm(out_channels),
                    nn.PReLU(init=0.2),
                )
                for i in range(K)
            ])

            self.shortcuts = nn.ModuleList([
                nn.Conv1d(in_channel, out_channels, kernel_size=1)
                for i in range(K)
            ])
        else:
            self.blocks = nn.ModuleList([
                                            nn.Sequential(
                                                nn.PReLU(init=0.2),
                                                nn.utils.weight_norm(nn.Conv1d(in_channel, in_channel,
                                                                               kernel_size=2 * i + 1,
                                                                               dilation=int(math.sqrt(i + 1)),
                                                                               padding=i * int(math.sqrt(i + 1)))),
                                                nn.PReLU(init=0.2),
                                                nn.utils.weight_norm(nn.Conv1d(in_channel, out_channels, kernel_size=1)),
                                            )
                                            for i in range(K)
                                            ])

            self.shortcuts = nn.ModuleList([
                                               nn.utils.weight_norm(nn.Conv1d(in_channel, out_channels, kernel_size=1))
                                               for i in range(K)
                                               ])

    def forward(self, x, x_mask):
        conv_bank = []
        # Convolution Bank
        for block, shortcut in zip(self.blocks, self.shortcuts):
            c = block(x * x_mask) + shortcut(x * x_mask)  # Convolution
            conv_bank.append(c)

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        return conv_bank * x_mask

    def remove_weight_norm(self):
        if self.use_layer_norm:
            return

        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])
            nn.utils.remove_weight_norm(shortcut)


class ResStack2Stage(nn.Module):
    def __init__(self, in_channel, out_channels, K, use_layer_norm=False):
        super(ResStack2Stage, self).__init__()
        assert K % 2 == 0

        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.blocks = nn.ModuleList([
                                            nn.Sequential(
                                                nn.Conv1d(in_channel, in_channel,
                                                           kernel_size=2 * i + 1,
                                                           dilation=int(math.sqrt(i + 1)),
                                                           padding=i * int(math.sqrt(i + 1))),
                                                LayerNorm(in_channel),
                                                nn.PReLU(init=0.2),
                                                nn.Conv1d(in_channel, out_channels, kernel_size=1),
                                                LayerNorm(out_channels),
                                                nn.PReLU(init=0.2),
                                            )
                                            for i in range(1, K + 1)
                                            ])

            self.shortcuts = nn.ModuleList([nn.Conv1d(in_channel, out_channels, kernel_size=1) for i in range(1, K + 1) ])

            self.proj1 = nn.Conv1d(out_channels * K // 2, out_channels, 1)

            self.proj2 = nn.Conv1d(out_channels * K // 2, out_channels, 1)
        else:
            self.blocks = nn.ModuleList([
                                            nn.Sequential(
                                                nn.LeakyReLU(0.2),
                                                nn.utils.weight_norm(nn.Conv1d(in_channel, in_channel,
                                                                               kernel_size=2 * i + 1,
                                                                               dilation=int(math.sqrt(i + 1)),
                                                                               padding=i * int(math.sqrt(i + 1)))),
                                                nn.LeakyReLU(0.2),
                                                nn.utils.weight_norm(nn.Conv1d(in_channel, out_channels, kernel_size=1)),
                                            )
                                            for i in range(1, K + 1)
                                            ])

            self.shortcuts = nn.ModuleList([
                                               nn.utils.weight_norm(nn.Conv1d(in_channel, out_channels, kernel_size=1))
                                               for i in range(1, K + 1)
                                               ])

            self.proj1 = nn.utils.weight_norm(nn.Conv1d(out_channels * K // 2, out_channels, 1))

            self.proj2 = nn.utils.weight_norm(nn.Conv1d(out_channels * K // 2, out_channels, 1))

    def forward(self, x, x_mask):
        conv_bank = []
        # Convolution Bank
        K = len(self.blocks)
        for k in range(K // 2):
            block = self.blocks[k]
            shortcut = self.shortcuts[k]
            c = block(x * x_mask) + shortcut(x * x_mask)  # Convolution
            conv_bank.append(c)
        conv_bank = torch.cat(conv_bank, dim=1)
        x = x + self.proj1(conv_bank * x_mask)

        conv_bank = []
        for k in range(K // 2):
            block = self.blocks[K // 2 + k]
            shortcut = self.shortcuts[K // 2 + k]
            c = block(x * x_mask) + shortcut(x * x_mask)  # Convolution
            conv_bank.append(c)
        conv_bank = torch.cat(conv_bank, dim=1)
        x = x + self.proj2(conv_bank * x_mask)

        return x * x_mask

    def remove_weight_norm(self):
        if self.use_layer_norm:
            return

        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])
            nn.utils.remove_weight_norm(shortcut)
        nn.utils.remove_weight_norm(self.proj1)
        nn.utils.remove_weight_norm(self.proj2)


class ResSequenceDilated(nn.Module):
    def __init__(self, in_channel, out_channels, K):
        super(ResSequenceDilated, self).__init__()

        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channel, out_channels, 1))
        dilations = [min(4, int(math.sqrt(1.7 ** i))) for i in range(1, K)]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels,
                                               kernel_size=2 * i + 1,
                                               dilation=dilations[i - 1],
                                               padding=i * dilations[i - 1])),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=1)),
            )
            for i in range(1, K)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=1))
            for i in range(1, K)
        ])

    def forward(self, x, x_mask):
        x = self.conv1(x)

        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = block(x * x_mask) + shortcut(x * x_mask)

        return x * x_mask

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])
            nn.utils.remove_weight_norm(shortcut)


class ResSequence(nn.Module):
    def __init__(self, in_channel, out_channels, K, use_layer_norm=False):
        super(ResSequence, self).__init__()

        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.conv1 = nn.Conv1d(in_channel, out_channels, 1)

            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels,
                              kernel_size=2 * i + 1, dilation=int(math.sqrt(i + 1)),
                              padding=i * int(math.sqrt(i + 1))),
                    LayerNorm(out_channels),
                    nn.PReLU(init=0.2),
                    nn.Conv1d(out_channels, out_channels, kernel_size=1),
                    LayerNorm(out_channels),
                    nn.PReLU(init=0.2),
                )
                for i in range(1, K)
            ])

            self.shortcuts = nn.ModuleList([
                nn.Conv1d(out_channels, out_channels, kernel_size=1)
                for i in range(1, K)
            ])
        else:
            self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channel, out_channels, 1))

            self.blocks = nn.ModuleList([
                                            nn.Sequential(
                                                nn.LeakyReLU(0.2),
                                                nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels,
                                                                               kernel_size=2 * i + 1,
                                                                               dilation=int(math.sqrt(i + 1)),
                                                                               padding=i * int(math.sqrt(i + 1)))),
                                                nn.LeakyReLU(0.2),
                                                nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=1)),
                                            )
                                            for i in range(K)
                                            ])

            self.shortcuts = nn.ModuleList([
                                               nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=1))
                                               for i in range(K)
                                               ])

    def forward(self, x, x_mask):
        x = self.conv1(x)

        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = block(x * x_mask) + shortcut(x * x_mask)

        return x * x_mask

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])
            nn.utils.remove_weight_norm(shortcut)


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels=256, kernel_size=3, p_dropout=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2))
        self.conv_2 = nn.utils.weight_norm(
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2))
        self.conv_3 = nn.utils.weight_norm(
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2))
        self.proj = nn.utils.weight_norm(nn.Conv1d(filter_channels, 1, 1))

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        # x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        # x = self.norm_2(x)
        x = self.drop(x)

        x = self.conv_3(x * x_mask)
        x = torch.relu(x)
        # x = self.norm_3(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)

        return x * x_mask

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_1)
        nn.utils.remove_weight_norm(self.conv_2)
        nn.utils.remove_weight_norm(self.conv_3)
        nn.utils.remove_weight_norm(self.proj)
