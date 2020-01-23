# https://github.com/d-li14/mobilenetv3.pytorch

"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch.nn as nn
import math

from segmentoly.rcnn.backbones.backbone import Backbone

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, kernel_size, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_fixed = nn.AvgPool2d(kernel_size)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        b, c, _, _ = x.size()
        if x.shape[2:] == self.kernel_size:
            y = self.avg_pool_fixed(x).view(b, c)
        else:
            y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, shape):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim, kernel_size=shape) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim, kernel_size=shape) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(Backbone):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1., shape=None):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        dims_out = [input_channel]
        scales_out = [2]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            scale = scales_out[-1] *  2 if s == 2 else scales_out[-1]
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs,
                                shape=(shape[0] // scale, shape[1] // scale)))
            scales_out.append(scale)
            dims_out.append(output_channel)
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        #building last several layers
        self.conv = nn.Sequential(
            conv_1x1_bn(input_channel, _make_divisible(exp_size * width_mult, 8)),
            SELayer(
                _make_divisible(exp_size * width_mult, 8)) if mode == 'small' else nn.Sequential()
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.classifier = nn.Sequential(
            nn.Linear(_make_divisible(exp_size * width_mult, 8), output_channel),
            nn.BatchNorm1d(output_channel) if mode == 'small' else nn.Sequential(),
            h_swish(),
            nn.Linear(output_channel, num_classes),
            nn.BatchNorm1d(num_classes) if mode == 'small' else nn.Sequential(),
            h_swish() if mode == 'small' else nn.Sequential()
        )

        self._initialize_weights()

        self._all_dims_out = tuple(dims_out)
        self._all_scales_out = tuple(scales_out)

    @property
    def stages(self):
        return self.features

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.conv(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x

    def _initialize_weights(self):
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


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 0, 0, 1],
        [3,  64,  24, 0, 0, 2],
        [3,  72,  24, 0, 0, 1],
        [5,  72,  40, 1, 0, 2],
        [5, 120,  40, 1, 0, 1],
        [5, 120,  40, 1, 0, 1],
        [3, 240,  80, 0, 1, 2],
        [3, 200,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 184,  80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 672, 160, 1, 1, 2],
        [5, 960, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3,  16,  16, 1, 0, 2],
        [3,  72,  24, 0, 0, 2],
        [3,  88,  24, 0, 0, 1],
        [5,  96,  40, 1, 1, 2],
        [5, 240,  40, 1, 1, 1],
        [5, 240,  40, 1, 1, 1],
        [5, 120,  48, 1, 1, 1],
        [5, 144,  48, 1, 1, 1],
        [5, 288,  96, 1, 1, 2],
        [5, 576,  96, 1, 1, 1],
        [5, 576,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)