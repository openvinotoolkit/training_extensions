from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import load_state


class RMBlock(nn.Module):
    def __init__(self, input_planes, squeeze_planes, output_planes, downsample=False, dropout_ratio=0.1,
                 activation=nn.ELU):
        super(RMBlock, self).__init__()
        self.downsample = downsample
        self.input_planes = input_planes
        self.output_planes = output_planes

        self.squeeze_conv = nn.Conv2d(input_planes, squeeze_planes, kernel_size=1, bias=False)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)

        self.dw_conv = nn.Conv2d(squeeze_planes, squeeze_planes, groups=squeeze_planes, kernel_size=3, padding=1,
                                 stride=2 if downsample else 1, bias=False)
        self.dw_bn = nn.BatchNorm2d(squeeze_planes)

        self.expand_conv = nn.Conv2d(squeeze_planes, output_planes, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(output_planes)

        self.activation = activation(inplace=True)
        self.dropout_ratio = dropout_ratio

        if self.downsample:
            self.skip_conv = nn.Conv2d(input_planes, output_planes, kernel_size=1, bias=False)
            self.skip_conv_bn = nn.BatchNorm2d(output_planes)

        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.activation(self.squeeze_bn(self.squeeze_conv(x)))
        out = self.activation(self.dw_bn(self.dw_conv(out)))
        out = self.expand_bn(self.expand_conv(out))
        if self.dropout_ratio > 0:
            out = F.dropout(out, p=self.dropout_ratio, training=self.training, inplace=True)
        if self.downsample:
            residual = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
            residual = self.skip_conv(residual)
            residual = self.skip_conv_bn(residual)
        out += residual
        return self.activation(out)


class RMNetBody(nn.Module):
    def __init__(self, block=RMBlock, blocks_per_stage=(None, 4, 8, 10, 11), trunk_width=(32, 32, 64, 128, 256),
                 bottleneck_width=(None, 8, 16, 32, 64)):
        super(RMNetBody, self).__init__()
        assert len(blocks_per_stage) == len(trunk_width) == len(bottleneck_width)
        self.dim_out = trunk_width[-1]

        stages = [nn.Sequential(OrderedDict([
            ('data_bn', nn.BatchNorm2d(3)),
            ('conv1', nn.Conv2d(3, trunk_width[0], kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(trunk_width[0])),
            ('relu1', nn.ReLU(inplace=True))
        ])),
        ]
        for i, (blocks_num, w, wb) in enumerate(zip(blocks_per_stage, trunk_width, bottleneck_width)):
            # Zeroth stage is already added.
            if i == 0:
                continue
            stage = []
            # Do not downscale input to the first stage.
            if i > 1:
                stage.append(block(trunk_width[i - 1], wb, w, downsample=True))
            for _ in range(blocks_num):
                stage.append(block(w, wb, w))
            stages.append(nn.Sequential(*stage))

        self.stages = nn.Sequential(OrderedDict([
            ('stage_{}'.format(i), stage) for i, stage in enumerate(stages)
        ]))

        self.init_weights()

    def init_weights(self):
        m = self.stages[0][0]  # ['data_bn']
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
        m = self.stages[0][1]  # ['conv1']
        nn.init.kaiming_normal(m.weight, mode='fan_out')
        m = self.stages[0][2]  # ['bn1']
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
        # All other blocks should be initialized internally during instantiation.

    def forward(self, x):
        return self.stages(x)


class RMNetClassifier(nn.Module):
    def __init__(self, num_classes, body=RMNetBody, dropout_ratio=0.1, pretrained=None):
        super(RMNetClassifier, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.backbone = body()
        self.extra_conv_bn_relu = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
                                                nn.BatchNorm2d(512), nn.ELU())
        self.extra_conv_bn_relu_2 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
                                                  nn.BatchNorm2d(1024), nn.ReLU())
        self.fc = nn.Conv2d(1024, num_classes, 1, stride=1, padding=0)

        if pretrained:
            checkpoint = torch.load(pretrained)
            load_state(self, checkpoint)
            # self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.backbone(x)
        x = self.extra_conv_bn_relu(x)
        x = self.extra_conv_bn_relu_2(x)
        x = F.avg_pool2d(x, (4, 4))
        x = self.fc(x)
        x = x.view(-1, x.size(1))
        return x
