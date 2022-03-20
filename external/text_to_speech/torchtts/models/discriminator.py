# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, k_size):
        super(Discriminator, self).__init__()
        kernel_size_d = (5, k_size)
        stride_d_2 = (2, 2)
        stride_d_1 = (1, 1)
        padding_d = (2, k_size // 2)

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 8, kernel_size=1)),
            nn.utils.weight_norm(nn.Conv2d(8, 16, kernel_size=1, stride=stride_d_2, groups=2)),
            nn.utils.weight_norm(nn.Conv2d(16, 32, kernel_size=1, stride=stride_d_1, groups=8)),
            nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=1, stride=stride_d_2, groups=16)),
            nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=1, stride=stride_d_1, groups=32)),
            nn.utils.weight_norm(nn.Conv2d(128, 128, kernel_size=1)),
            nn.utils.weight_norm(nn.Conv2d(128, 1, kernel_size=1)),
        ])

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.utils.weight_norm(nn.Conv2d(1, 8, kernel_size=(7, 7), stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(p=0.1),
                nn.utils.weight_norm(
                    nn.Conv2d(8, 16, kernel_size=kernel_size_d, stride=stride_d_2, padding=padding_d, groups=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(p=0.1),
                nn.utils.weight_norm(
                    nn.Conv2d(16, 32, kernel_size=kernel_size_d, stride=stride_d_1, padding=padding_d, groups=8)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(32, 64, kernel_size=kernel_size_d, stride=stride_d_2, padding=padding_d, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(p=0.1),
                nn.utils.weight_norm(
                    nn.Conv2d(64, 128, kernel_size=kernel_size_d, stride=stride_d_1, padding=padding_d, groups=32)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()

        for i, module in enumerate(self.discriminator):
            x = module(x) + self.shortcuts[i](x)
            if (i > 0 and x.shape[2] > 20) or i == len(self.discriminator) - 1:
                features.append(x)
        if len(features) == 1:
            return None, features[-1]

        return features[:-1], features[-1]


class DiscriminatorFreq(nn.Module):
    def __init__(self, kernel_size):
        super(DiscriminatorFreq, self).__init__()
        kernel_size_v = (kernel_size, 5)
        padding_v = (kernel_size // 2, 2)

        stride_d_1 = (1, 1)
        stride_d_2 = (2, 2)

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 8, kernel_size=1)),
            nn.utils.weight_norm(nn.Conv2d(8, 16, kernel_size=1, stride=stride_d_2, groups=2)),
            nn.utils.weight_norm(nn.Conv2d(16, 32, kernel_size=1, stride=stride_d_1, groups=8)),
            nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=1, stride=stride_d_2, groups=16)),
            nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=1, stride=stride_d_1, groups=32)),
            nn.utils.weight_norm(nn.Conv2d(128, 128, kernel_size=1)),
            nn.utils.weight_norm(nn.Conv2d(128, 1, kernel_size=1)),
        ])


        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.utils.weight_norm(nn.Conv2d(1, 8, kernel_size=(7, 7), stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(8, 16, kernel_size=kernel_size_v, stride=stride_d_2, padding=padding_v, groups=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(16, 32, kernel_size=kernel_size_v, stride=stride_d_1, padding=padding_v, groups=8)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(32, 64, kernel_size=kernel_size_v, stride=stride_d_2, padding=padding_v, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(
                    nn.Conv2d(64, 128, kernel_size=kernel_size_v, stride=stride_d_1, padding=padding_v, groups=32)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for i, module in enumerate(self.discriminator):
            x = module(x) + self.shortcuts[i](x)
            if (i > 0 and x.shape[2] > 20) or i == len(self.discriminator) - 1:
                features.append(x)
        if len(features) == 1:
            return None, features[-1]
        return features[:-1], features[-1]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, discriminator_kernel_sizes=None):
        super(MultiScaleDiscriminator, self).__init__()

        if discriminator_kernel_sizes is None:
            discriminator_kernel_sizes = [21, 11, 9]
        self.discriminators = nn.ModuleList(
            [Discriminator(kernel_size) for kernel_size in discriminator_kernel_sizes]
        )

        self.discriminators_freq = nn.ModuleList(
            [DiscriminatorFreq(kernel_size) for kernel_size in discriminator_kernel_sizes]
        )

        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool2d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
             for _ in range(len(discriminator_kernel_sizes) - 1)]
        )

    def forward(self, x):
        ret = list()

        for pool, disc_f, disc_t in zip(self.pooling, self.discriminators_freq, self.discriminators):
            x = pool(x)
            ret.append(disc_t(x))
            ret.append(disc_f(x))

        return ret


if __name__ == '__main__':
    model = MultiScaleDiscriminator()

    x = torch.randn(3, 1, 80, 128)
    print('Input shape ', x.shape)

    data = model(x)
    for features, score in data:
        print('Score shape ', score.shape)
        if features is None:
            continue
        for feat in features:
            print(' Activation shape', feat.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
