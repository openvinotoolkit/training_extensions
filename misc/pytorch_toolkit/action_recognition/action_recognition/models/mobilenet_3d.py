import math

import torch.nn as nn


class MobileNet(nn.Module):
    @staticmethod
    def conv_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv3d(
                inp,
                oup,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def conv_dw(inp, oup, stride):
        return nn.Sequential(
            nn.Conv3d(
                inp,
                inp,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=inp,
                bias=False),
            nn.BatchNorm3d(inp),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                inp,
                oup,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )

    def __init__(self, sample_size, sample_duration, num_classes=400, last_fc=True):
        super(MobileNet, self).__init__()

        self.last_fc = last_fc

        self.model = nn.Sequential(
            self.conv_bn(3, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 1),
        )

        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)

        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)

        return x


class DepthWiseBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(DepthWiseBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            inp,
            inp,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=inp,
            bias=False)
        self.bn1 = nn.BatchNorm3d(inp)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            inp,
            oup,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn2 = nn.BatchNorm3d(oup)
        self.inplanes = inp
        self.outplanes = oup
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        out = self.relu(out)

        return out


class MobileNetResidual(nn.Module):
    @staticmethod
    def conv_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv3d(
                inp,
                oup,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            nn.BatchNorm3d(oup),
            nn.ReLU(inplace=True)
        )

    def __init__(self, sample_size, sample_duration, num_classes=400, last_fc=True):
        super(MobileNetResidual, self).__init__()

        self.last_fc = last_fc

        self.model = nn.Sequential(
            self.conv_bn(3, 32, 2),
            DepthWiseBlock(32, 64, 1),
            DepthWiseBlock(64, 128, (1, 2, 2)),
            DepthWiseBlock(128, 128, 1),
            DepthWiseBlock(128, 256, 2),
            DepthWiseBlock(256, 256, 1),
            DepthWiseBlock(256, 512, 2),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 512, 1),
            DepthWiseBlock(512, 1024, 2),
            DepthWiseBlock(1024, 1024, 1),
        )

        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)

        x = self.avgpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)

        return x
