"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# ICNet implementation attempting to closely follow the original authors' model at:
# https://github.com/hszhao/ICNet
# Important differences:
# 1) Upsampling is nearest-neighbour instead of bilinear since it is impossible
#    to export bilinear upsampling to ONNX yet
# 2) Weight initialization is omitted because it caused mIoU degradation on CamVid

from collections import OrderedDict
from pkg_resources import parse_version

from numpy import lcm
import torch.nn as nn
import torch
import torch.nn.functional as F

from examples.common.example_logger import logger
from nncf.utils import is_tracing_state


class ConvBN(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.05)  # Corresponds to momentum 0.95 in Caffe notation

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.convbn = ConvBN(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.convbn(inputs)
        x = self.relu(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, reduce_channels, increase_channels, dilation=1, stride=1):
        super().__init__()
        nonshrinking_padding = dilation
        self.conv_1x1_reduce_bnrelu = ConvBNReLU(in_channels, out_channels=reduce_channels,
                                                 kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)
        self.conv_3x3_bnrelu = ConvBNReLU(in_channels=reduce_channels, out_channels=reduce_channels,
                                          kernel_size=3, stride=1, padding=nonshrinking_padding,
                                          dilation=dilation, bias=False)
        self.conv_1x1_increase_bn = ConvBN(in_channels=reduce_channels, out_channels=increase_channels,
                                           kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.need_proj = (in_channels != increase_channels)
        if self.need_proj:
            self.conv_1x1_proj_bn = ConvBN(in_channels, out_channels=increase_channels,
                                           kernel_size=1, stride=stride, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        fx = self.conv_1x1_reduce_bnrelu(inputs)
        fx = self.conv_3x3_bnrelu(fx)
        fx = self.conv_1x1_increase_bn(fx)
        x = inputs
        if self.need_proj:
            x = self.conv_1x1_proj_bn(x)
        out = fx + x
        out = self.relu(out)
        return out


class ICNetBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Naming conventions below are chosen to correspond to the
        # icnet_cityscapes_bnnomerge.prototxt file in the original ICNet Github
        # repository. Although ICNet low-resolution branch layers 'conv3', 'conv4' and 'conv5',
        # are based upon ResNet50, they rather correspond to ResNet50 layers
        # 'conv2', 'conv3' and 'conv4' respectively.

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_1_3x3_s2', ConvBNReLU(in_channels, out_channels=32, kernel_size=3,
                                          stride=2, padding=1, dilation=1, bias=False)),
            ('conv1_2_3x3', ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3,
                                       stride=1, padding=1, dilation=1, bias=False)),
            ('conv1_3_3x3', ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3,
                                       stride=1, padding=1, dilation=1, bias=False)),
            ]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2_1', ResNetBlock(64, 32, 128)),
            ('conv2_2', ResNetBlock(128, 32, 128)),
            ('conv2_3', ResNetBlock(128, 32, 128))
            ]))
        self.conv3_1 = ResNetBlock(128, 64, 256, stride=2)
        self.conv3_rest = nn.Sequential(OrderedDict([
            ('conv3_2', ResNetBlock(256, 64, 256)),
            ('conv3_3', ResNetBlock(256, 64, 256)),
            ('conv3_4', ResNetBlock(256, 64, 256))
            ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4_1', ResNetBlock(256, 128, 512, dilation=2)),
            ('conv4_2', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_3', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_4', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_5', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_6', ResNetBlock(512, 128, 512, dilation=2)),
        ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4_1', ResNetBlock(256, 128, 512, dilation=2)),
            ('conv4_2', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_3', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_4', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_5', ResNetBlock(512, 128, 512, dilation=2)),
            ('conv4_6', ResNetBlock(512, 128, 512, dilation=2)),
        ]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('conv5_1', ResNetBlock(512, 256, 1024, dilation=4)),
            ('conv5_2', ResNetBlock(1024, 256, 1024, dilation=4)),
            ('conv5_3', ResNetBlock(1024, 256, 1024, dilation=4)),
        ]))

    def forward(self):
        pass


def get_backbone(backbone, in_channels):
    if backbone == 'icnet':
        return ICNetBackbone(in_channels)
    raise NotImplementedError


class PyramidPooling(nn.Module):
    def __init__(self, input_size_hw, bin_dimensions=None, mode='sum'):
        super().__init__()

        if mode not in ['sum', 'cat']:
            raise NotImplementedError

        self.mode = mode
        self.input_size_hw = input_size_hw
        #self.sampling_params = {'mode': 'bilinear', 'align_corners': True}
        self.sampling_params = {'mode': 'nearest'}
        if bin_dimensions is None:
            self.bin_dimensions = [1, 2, 3, 6]
        else:
            self.bin_dimensions = bin_dimensions

        # ONNX only supports exporting adaptive_avg_pool2d if the input tensor
        # height and width are exact multiples of the output_size (i.e. bin dimensions).
        # Inference-time pad calculation is also impossible to export to ONNX, therefore
        # the required padding parameters are pre-calculated here, at init.
        self.paddings = {}
        for dim in self.bin_dimensions:
            pad_h = (dim - (input_size_hw[0] % dim)) % dim
            pad_w = (dim - (input_size_hw[1] % dim)) % dim
            self.paddings[dim] = [0, pad_w, 0, pad_h]

    def forward(self, inputs):
        x = inputs.clone()

        for dim in self.bin_dimensions:
            padded_input = F.pad(inputs, self.paddings[dim], mode='constant', value=0)
            pooled_feature = F.adaptive_avg_pool2d(padded_input, dim)
            pooled_feature = F.interpolate(pooled_feature, self.input_size_hw, **self.sampling_params)
            if self.mode == 'sum':
                x += pooled_feature
            elif self.mode == 'cat':
                x = torch.cat(pooled_feature)
            else:
                raise NotImplementedError

        return x


class CascadeFeatureFusion(nn.Module):
    def __init__(self, in_channels_lowres, in_channels_highres, highres_size_hw, num_classes):
        super().__init__()
        #self.sampling_params = {'mode': 'bilinear', 'align_corners': True}
        self.sampling_params = {'mode': 'nearest'}

        self.conv = ConvBN(in_channels_lowres, out_channels=128,
                           kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv_proj = ConvBN(in_channels_highres, out_channels=128,
                                kernel_size=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(in_channels_lowres, out_channels=num_classes,
                                    kernel_size=1, padding=0, dilation=1, bias=True)
        self.highres_size_hw = highres_size_hw

    def forward(self, lowres_input, highres_input):
        upsampled = F.interpolate(lowres_input, self.highres_size_hw, **self.sampling_params)
        lr = self.conv(upsampled)
        hr = self.conv_proj(highres_input)
        x = lr + hr
        x = self.relu(x)
        if self.training:
            aux_labels = self.classifier(upsampled)
            return x, aux_labels
        return x


class ICNet(nn.Module):
    def __init__(
            self,
            input_size_hw,
            in_channels=3,
            n_classes=20,
            backbone='icnet'
    ):
        super().__init__()
        self._input_size_hw = input_size_hw

        self._input_size_hw_ds2 = (self._input_size_hw[0] // 2, self._input_size_hw[1] // 2)
        self._input_size_hw_ds4 = (self._input_size_hw[0] // 4, self._input_size_hw[1] // 4)
        self._input_size_hw_ds8 = (self._input_size_hw[0] // 8, self._input_size_hw[1] // 8)
        self._input_size_hw_ds16 = (self._input_size_hw[0] // 16, self._input_size_hw[1] // 16)
        self._input_size_hw_ds32 = (self._input_size_hw[0] // 32, self._input_size_hw[1] // 32)


        #self.sampling_params = {'mode': 'bilinear', 'align_corners': True}
        self.sampling_params = {'mode': 'nearest'}


        self.backbone = get_backbone(backbone, in_channels)

        self.highres_conv = nn.Sequential(OrderedDict([
            ('conv1_sub1', ConvBNReLU(in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)),
            ('conv2_sub1', ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)),
            ('conv3_sub1', ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False))
        ]))

        # 'conv5_4_k1' is applied immediately after pyramid pooling and before
        # cascade feature fusion
        self.conv5_4_k1 = ConvBNReLU(in_channels=1024, out_channels=256,
                                     kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # Using pyramid pooling in 'sum' mode instead of 'cat' as in PSPNet,
        # probably because in ICNet it is immediately followed by 1x1 reduce
        # convolution anyway
        self.ppm = PyramidPooling(self._input_size_hw_ds32)
        self.cff42 = CascadeFeatureFusion(in_channels_lowres=256, in_channels_highres=256,
                                          highres_size_hw=self._input_size_hw_ds16, num_classes=n_classes)
        self.cff421 = CascadeFeatureFusion(in_channels_lowres=128, in_channels_highres=32,
                                           highres_size_hw=self._input_size_hw_ds8, num_classes=n_classes)
        self.conv6_cls = nn.Conv2d(128, out_channels=n_classes,
                                   kernel_size=1, padding=0, dilation=1, bias=True)


        required_alignment = 32
        for bin_dim in self.ppm.bin_dimensions:
            required_alignment = lcm(required_alignment, bin_dim)
        if (input_size_hw[0] % required_alignment) or (input_size_hw[1] % required_alignment):
            raise ValueError("ICNet may only operate on {}-aligned input resolutions".format(required_alignment))
        # Weight initialization
        # for module in self.modules():
        #     if isinstance(module, nn.Conv2d):
        #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        #         if module.bias is not None:
        #             module.bias.data.zero_()
        #     elif isinstance(module, nn.BatchNorm2d):
        #         nn.init.constant_(module.weight, 1)
        #         nn.init.constant_(module.bias, 0)

    def highres_branch(self, inputs):
        x = self.highres_conv(inputs)
        return x

    def mediumres_branch(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.maxpool(x)
        x = self.backbone.conv2(x)
        x = self.backbone.conv3_1(x)
        return x

    def lowres_branch(self, inputs):
        x = self.backbone.conv3_rest(inputs)
        x = self.backbone.conv4(x)
        x = self.backbone.conv5(x)
        x = self.ppm(x)
        x = self.conv5_4_k1(x)
        return x

    def forward(self, inputs):
        data_sub1 = inputs
        features_sub1 = self.highres_branch(data_sub1)

        data_sub2 = F.interpolate(data_sub1, self._input_size_hw_ds2, **self.sampling_params)
        features_sub2 = self.mediumres_branch(data_sub2)

        # Contrary to the ICNet paper Fig.2 , the low-resolution branch does not receive separate
        # 4x-downsampled image input, but instead reuses feature maps from the medium-resolution
        # branch.

        data_sub4 = F.interpolate(features_sub2, self._input_size_hw_ds32, **self.sampling_params)
        features_sub4 = self.lowres_branch(data_sub4)

        if self.training:
            fused_features_sub42, label_scores_ds16 = self.cff42(features_sub4, features_sub2)
            fused_features_sub421, label_scores_ds8 = self.cff421(fused_features_sub42, features_sub1)

            fused_features_ds4 = F.interpolate(fused_features_sub421, self._input_size_hw_ds4, **self.sampling_params)
            label_scores_ds4 = self.conv6_cls(fused_features_ds4)

            return OrderedDict([("ds4", label_scores_ds4),
                                ("ds8", label_scores_ds8),
                                ("ds16", label_scores_ds16)])

        fused_features_sub42 = self.cff42(features_sub4, features_sub2)
        fused_features_sub421 = self.cff421(fused_features_sub42, features_sub1)

        fused_features_ds4 = F.interpolate(fused_features_sub421, self._input_size_hw_ds4, **self.sampling_params)
        label_scores_ds4 = self.conv6_cls(fused_features_ds4)
        label_scores = F.interpolate(label_scores_ds4, self._input_size_hw, **self.sampling_params)
        if is_tracing_state() and parse_version(torch.__version__) >= parse_version("1.1.0"):
            # While exporting, add extra post-processing layers into the graph
            # so that the model outputs class probabilities instead of class scores
            softmaxed = F.softmax(label_scores, dim=1)
            return softmaxed
        return label_scores


def icnet(num_classes, pretrained=False, **kwargs):
    model = ICNet(n_classes=num_classes, **kwargs)

    if pretrained:
        logger.warning("ICNet has no pretrained weights")

    return model
