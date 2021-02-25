# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import gc
import torch
import torch.nn as nn


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(conv, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(5, 5, 3), stride=(stride, stride, 1), padding=(2, 2, 1),
                               bias=False, groups=groups)

    # pylint: disable=W0221
    def forward(self, x):
        out = x
        out = self.conv1(out)
        return out

class ResNextBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None, width=4, compression=2):
        super(ResNextBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        conv_groups = out_channels // (width*compression)

        self.conv_pre = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//compression,
                                  kernel_size=1, stride=1, padding=0, bias=False, groups=1)
        self.conv1 = conv(in_channels=out_channels//compression, out_channels=out_channels//compression,
                          stride=stride, groups=conv_groups)

        self.relu1 = nn.LeakyReLU(2e-2, inplace=True)
        self.relu2 = nn.LeakyReLU(2e-2, inplace=True)
        self.relu3 = nn.LeakyReLU(2e-2, inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression)
        self.norm3 = nn.InstanceNorm3d(num_features=out_channels)
        self.conv_post = nn.Conv3d(in_channels=out_channels // compression, out_channels=out_channels,
                                   kernel_size=1, padding=0, bias=False, groups=1)

    # pylint: disable=W0221
    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm3(out)

        out = x + out
        out = self.relu3(out)

        return out

class shuffle(nn.Module):
    def __init__(self, ratio):
        super(shuffle, self).__init__()
        self.ratio = ratio

    # pylint: disable=W0221
    def forward(self, x):
        batch_size, in_channels, d, h, w = x.shape
        out_channels = in_channels // (self.ratio*self.ratio*self.ratio)
        out = x.view(batch_size*out_channels, self.ratio, self.ratio, self.ratio, d, h, w)
        out = out.permute(0, 4, 1, 5, 2, 6, 3)

        return out.contiguous().view(batch_size, out_channels, d*self.ratio, h*self.ratio, w*self.ratio)


class re_shuffle(nn.Module):
    def __init__(self, ratio):
        super(re_shuffle, self).__init__()
        self.ratio = ratio

    # pylint: disable=W0221
    def forward(self, x):
        batch_size, in_channels, d, h, w = x.shape

        out_channels = in_channels * self.ratio * self.ratio * self.ratio
        out = x.view(batch_size*in_channels, d//self.ratio, self.ratio,
                     h//self.ratio, self.ratio, w//self.ratio, self.ratio)
        out = out.permute(0, 2, 4, 6, 1, 3, 5)
        out = out.contiguous().view(batch_size, out_channels, d//self.ratio, h//self.ratio, w//self.ratio)
        return out

class UpsamplingPixelShuffle(nn.Module):
    def __init__(self, input_channels, output_channels, ratio=2):
        super(UpsamplingPixelShuffle, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv3d(in_channels=input_channels, out_channels=output_channels*int(ratio**3),
                              kernel_size=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(2e-2, inplace=True)
        self.shuffle = shuffle(ratio=ratio)

    # pylint: disable=W0221
    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return out

# pylint: disable=R0201,R0902,R0915,W0621
class UNet(nn.Module):
    def __init__(self, depth, encoder_layers, number_of_channels, number_of_outputs):
        super(UNet, self).__init__()
        print('UNet {}'.format(number_of_channels))

        self.encoder_layers = encoder_layers

        self.number_of_channels = number_of_channels
        self.number_of_outputs = number_of_outputs
        self.depth = depth

        self.conv_input = 0

        self.encoder_convs = nn.ModuleList()

        self.upsampling = nn.ModuleList()

        self.decoder_convs = nn.ModuleList()

        self.decoder_convs1x1 = nn.ModuleList()

        self.attention_convs = nn.ModuleList()

        self.upsampling_distance = nn.ModuleList()

        self.conv_input = nn.Conv3d(in_channels=1, out_channels=self.number_of_channels[0], kernel_size=(7, 7, 3),
                                    stride=1, padding=(3, 3, 1), bias=False)
        self.norm_input = nn.InstanceNorm3d(num_features=self.number_of_channels[0])

        conv_first_list = []
        for _ in range(self.encoder_layers[0]):
            conv_first_list.append(ResNextBottleneck(in_channels=self.number_of_channels[0],
                                                     out_channels=self.number_of_channels[0], stride=1))

        self.conv_first = nn.Sequential(*conv_first_list)

        self.conv_middle = nn.Conv3d(in_channels=self.number_of_channels[-1], out_channels=self.number_of_channels[-1],
                                     kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(2e-2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv_output = nn.Conv3d(in_channels=self.number_of_channels[0], out_channels=self.number_of_outputs-1,
                                     kernel_size=1, stride=1, padding=0, bias=True, groups=self.number_of_outputs-1)

        self.softmax = nn.Softmax(dim=1)
        self.construct_dencoder_convs(depth=depth, number_of_channels=number_of_channels)
        self.construct_encoder_convs(depth=depth, number_of_channels=number_of_channels)
        self.construct_upsampling_convs(depth=depth, number_of_channels=number_of_channels)

    def _make_encoder_layer(self, in_channels, channels, blocks, stride=1, block=ResNextBottleneck):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=channels, kernel_size=2, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(in_channels=channels, out_channels=channels, stride=1, downsample=downsample))

        for _ in range(1, blocks):
            layers.append(block(in_channels=channels, out_channels=channels, stride=1))


        return nn.Sequential(*layers)

    def construct_encoder_convs(self, depth, number_of_channels):
        for i in range(depth-1):
            conv = self._make_encoder_layer(in_channels=number_of_channels[i], channels=number_of_channels[i+1],
                                            blocks=self.encoder_layers[i+1], stride=2, block=ResNextBottleneck)
            self.encoder_convs.append(conv)

    def construct_dencoder_convs(self, depth, number_of_channels):
        for i in range(depth):

            conv_list = []
            for _ in range(self.encoder_layers[i]):
                conv_list.append(ResNextBottleneck(in_channels=number_of_channels[i],
                                                   out_channels=number_of_channels[i], stride=1))

            conv = nn.Sequential(
                *conv_list
            )

            conv1x1 = nn.Conv3d(in_channels=2*number_of_channels[i], out_channels=number_of_channels[i],
                                kernel_size=1, bias=False)
            self.decoder_convs.append(conv)
            self.decoder_convs1x1.append(conv1x1)

    def construct_upsampling_convs(self, depth, number_of_channels):
        for i in range(depth-1):
            conv = UpsamplingPixelShuffle(input_channels=number_of_channels[i+1], output_channels=number_of_channels[i])
            self.upsampling.append(conv)

    # pylint: disable=W0221
    def forward(self, x):
        skip_connections = []
        gc.collect()
        input = x[0]

        conv = self.conv_input(input)
        conv = self.norm_input(conv)
        conv = self.conv_first(conv)

        for i in range(self.depth-1):
            skip_connections.append(conv)
            conv = self.encoder_convs[i](conv)

        for i in reversed(range(self.depth-1)):
            conv = self.upsampling[i](conv)
            conv = self.relu(conv)

            conc = torch.cat([skip_connections[i], conv], dim=1)
            conv = self.decoder_convs1x1[i](conc)
            conv = self.relu(conv)
            conv = self.decoder_convs[i](conv)

        out_logits = self.conv_output(conv)

        out_logits = self.sigmoid(out_logits)

        return [out_logits,]
