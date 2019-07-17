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

import torch
import torch.nn as nn

from .base import FeatureExtractor, duplicate
from .deformable_conv import ConvOffset2d
from .group_norm import GroupNorm
from ..utils.weights import xavier_fill, msra_fill, get_group_gn


class PANet(FeatureExtractor):
    """Base class for PANet includes common methods for its features"""
    def __init__(self, group_norm=True):
        """
        :param group_norm: if True, 2d convolutions are continued by group normalization layer
        """
        super().__init__()
        self.group_norm = group_norm

    def forward(self, *inputs, use_stub=False):
        raise NotImplementedError

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, ConvOffset2d)):
                msra_fill(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                xavier_fill(m.weight)
                nn.init.constant_(m.bias, 0)

    def _conv2d_block(self, dim_in, dim_out, kernel, stride, padding, bias):
        if self.group_norm:
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                GroupNorm(get_group_gn(dim_out), dim_out, eps=1e-5),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True)
            )

    def _linear_block(self, dim_in, dim_out):
        if self.group_norm:
            return nn.Sequential(
                nn.Linear(dim_in, dim_out),
                GroupNorm(get_group_gn(dim_out), dim_out, eps=1e-5)
            )
        else:
            return nn.Linear(dim_in, dim_out)

    def _gn_relu_block(self, channels):
        if self.group_norm:
            return nn.Sequential(
                GroupNorm(get_group_gn(channels), channels, eps=1e-5),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.ReLU(inplace=True)


class BottomUpPathAugmentation(PANet):
    """Feature bottom-up-path-augmentation with usual 2d convolutions 3x3"""
    def __init__(self, output_levels, dims_in, scales_in, dim_out, group_norm):
        """
        Initialization with the next parameters:
        :param output_levels: number of levels (feature maps) from FPN
        :param dims_in: number of channels in input feature maps
        :param scales_in: scale of each input feature map
        :param dim_out: number of channels in output feature maps
        :param group_norm: if True, 2d convolutions are continued by group normalization layer
        """
        super().__init__(group_norm)
        self.output_levels = output_levels
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        self.scales_in = scales_in
        self.scales_out = scales_in
        self.dims_in = dims_in
        self.dim_out = dim_out
        dims_out = dim_out
        if not isinstance(dims_out, (tuple, list)):
            dims_out = duplicate(dims_out, len(dims_in))
        self.dims_out = dims_out

        for i in range(self.output_levels - 2):
            self.conv1.append(
                self._conv2d_block(dims_in[i], dim_out, kernel=3, stride=2, padding=1, bias=False)
            )
            self.conv2.append(
                self._conv2d_block(dims_in[i], dim_out, kernel=3, stride=1, padding=1, bias=False)
            )
        self._init_weights()

    def forward(self, x, use_stub=False):
        """
        :param x: (list) FPN output in order [P2, P3, P4, P5, P6]
        :return: list [N2, N3, N4, N5, N6], where N2 == P2, N6 = max pooling from N5.
                 Ni = ((Ni-1 -> conv1 -> ReLU) + Pi) -> conv2 -> ReLU if 2 < i < 6
        """
        out = []
        out.append(x[0])  # N2 = P2 from FPN
        for i in range(self.output_levels - 2):  # Except 2 levels: P2 and P6
            out.append(self.conv1[i](out[-1]))
            out[-1] = out[-1] + x[i + 1]
            out[-1] = self.conv2[i](out[-1])
        # Apply max pooling to N5
        out.append(self.maxpool(out[-1]))
        return out


class BottomUpPathAugmentationWithDeformConv(PANet):
    """Feature bottom-up-path-augmentation where 2d convolutions 3x3 are replaced by
    deformable convolutions with the same parameters (kernel_size, stride, padding, bias)
    """
    def __init__(self, output_levels, dims_in, scales_in, dim_out, group_norm, num_deformable_groups=1):
        """
        Initialization with the next parameters:
        :param output_levels: number of levels (feature maps) from FPN
        :param dims_in: number of channels in input feature maps
        :param dim_out: number of channels in output feature maps
        :param group_norm: if set, 2d convolutions are continued by group normalization layer
        :param num_deformable_groups:
        """
        super().__init__(group_norm)
        self.output_levels = output_levels
        self.num_deformable_groups = num_deformable_groups

        self.offset1 = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.gn_relu1 = nn.ModuleList()

        self.offset2 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.gn_relu2 = nn.ModuleList()

        self.scales_in = scales_in
        self.scales_out = scales_in
        self.dims_in = dims_in
        self.dim_out = dim_out
        dims_out = dim_out
        if not isinstance(dims_out, (tuple, list)):
            dims_out = duplicate(dims_out, len(dims_in))
        self.dims_out = dims_out

        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        deform_out_dim = self.num_deformable_groups * 2 * 3 * 3

        for i in range(self.output_levels - 2):
            # Conv1 block
            self.offset1.append(nn.Conv2d(dims_in[i], deform_out_dim, kernel_size=3, stride=2, padding=1, bias=False))
            self.conv1.append(ConvOffset2d(dims_in[i], dim_out, kernel_size=(3, 3), stride=2, padding=1,
                                           num_deformable_groups=self.num_deformable_groups))
            self.gn_relu1.append(self._gn_relu_block(dim_out))

            # Conv2 block
            self.offset2.append(nn.Conv2d(dims_in[i], deform_out_dim, kernel_size=3, stride=1, padding=1, bias=False))
            self.conv2.append(ConvOffset2d(dims_in[i], dim_out, kernel_size=(3, 3), stride=1, padding=1,
                                           num_deformable_groups=self.num_deformable_groups))
            self.gn_relu2.append(self._gn_relu_block(dim_out))

        self._init_weights()

    def _init_weights(self):
        super()._init_weights()
        for i in range(self.output_levels - 2):
            nn.init.constant_(self.offset1[i].weight, 0)
            nn.init.constant_(self.offset2[i].weight, 0)

    def forward(self, x):
        """
        :param x: (list) FPN output in order [P2, P3, P4, P5, P6]
        :return: list [N2, N3, N4, N5, N6], where N2 == P2, N6 = max pooling from N5.
                 Ni = ((Ni-1 -> offset1) -> conv1 -> ReLU + Pi) -> offset2 -> conv2 -> ReLU if 2 < i < 6
        """
        out = []
        out.append(x[0])  # N2 = P2 from FPN
        for i in range(self.output_levels - 2):  # Except 2 levels: P2 and P6
            offset = self.offset1[i](out[-1])
            out.append(self.conv1[i](out[-1], offset))
            out[-1] = self.gn_relu1[i](out[-1])

            out[-1] = out[-1] + x[i + 1]

            offset = self.offset2[i](out[-1])
            out[-1] = self.conv2[i](out[-1], offset)
            out[-1] = self.gn_relu2[i](out[-1])
        # Apply max pooling to N5
        out.append(self.maxpool(out[-1]))
        return out



class BboxHead(PANet):
    """Detection head with features from PANet"""
    def __init__(self, dim_in, dim_out, resolution_in, cls_num, cls_agnostic_bbox_regression=False,
                 afp_levels_num=4, heavier_head=False, conv_head_dim=256, num_convs=4,
                 group_norm=False):
        """
        Initialization with the next parameters:
        :param dim_in: channels number in input tensor
        :param dim_out: channels number in output tensor
        :param resolution_in: spatial resolution of input tensor
        :param cls_num: number of classes to train
        :param cls_agnostic_bbox_regression: Use a class agnostic bounding box
               regressor instead of the default per-class regressor
        :param afp_levels_num: number of levels (feature maps) from FPN or bottom-up-path-augmentation
        :param heavier_head: if True, heavier-head feature from PANet is switched on
        :param conv_head_dim: channels number in convolutions used in heavier_head
        :param num_convs: number of convolutions in heavier_head, by default equal to 4
        :param group_norm: if True, 2d convolutions are continued by group normalization layer
        """
        super().__init__(group_norm)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.resolution_in = resolution_in
        self.cls_num = cls_num
        self.cls_agnostic_bbox_regression = cls_agnostic_bbox_regression

        self.heavier_head = heavier_head
        self.levels_num = afp_levels_num

        # Define parameters for heavier_head
        if heavier_head:
            self.num_convs = num_convs
            self.conv_head_dim = conv_head_dim

        self.fc1 = nn.ModuleList()
        self.define_fc1()

        self.fc2 = None
        self.define_fc2()

        if self.heavier_head:
            self.fc = nn.Linear(dim_in * resolution_in * resolution_in, dim_out)

        self.cls_score = nn.Linear(dim_out, cls_num)
        box_out_dims = 4 * (1 if cls_agnostic_bbox_regression else cls_num)
        self.bbox_pred = nn.Linear(dim_out, box_out_dims)

        self._init_weights()

    def define_fc1(self):
        for _ in range(self.levels_num):
            if not self.heavier_head:
                self.fc1.append(
                    self._linear_block(self.dim_in * self.resolution_in * self.resolution_in, self.dim_out)
                )
            else:
                self.fc1.append(
                    self._conv2d_block(self.dim_in, self.conv_head_dim, kernel=3, stride=1, padding=1, bias=False)
                )

    def define_fc2(self):
        if not self.heavier_head:
            if self.group_norm:
                self.fc2 = nn.Sequential(
                    nn.Linear(self.dim_out, self.dim_out),
                    GroupNorm(get_group_gn(self.dim_out), self.dim_out, eps=1e-5),
                    nn.ReLU(inplace=True)
                )
            else:
                self.fc2 = nn.Sequential(
                    nn.Linear(self.dim_out, self.dim_out),
                    nn.ReLU(inplace=True)
                )
        else:
            module_list = []
            for i in range(self.num_convs - 1):
                module_list.extend(
                    self._conv2d_block(self.dim_in, self.conv_head_dim, kernel=3, stride=1, padding=1, bias=False)
                )
            self.fc2 = nn.Sequential(*module_list)

    def _init_weights(self):
        super()._init_weights()
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        batch_size = int(x.shape[1])
        for i in range(self.levels_num):
            if self.heavier_head:
                y = self.fc1[i](x[i])
            else:
                y = nn.functional.relu(self.fc1[i](x[i].view(batch_size, -1)), inplace=True)

            if i == 0:
                pooled_feature = y
            else:
                pooled_feature = torch.max(pooled_feature, y)

        x = self.fc2(pooled_feature)

        if self.heavier_head:
            x = nn.functional.relu(self.fc(x.view(batch_size, -1)), inplace=True)

        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = nn.functional.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


class MaskHead(PANet):
    """Segmentation head with features from PANet"""
    def __init__(self, dim_in, num_cls, dim_internal=256,
                 afp_levels_num=4, fully_connected_fusion=True, in_resolution=14,
                 group_norm=False):
        """
        Initialization with the next parameters:
        :param dim_in: number of channels in an input tensor
        :param num_cls: number of classes
        :param dim_internal: number of output channels in convolutions
        :param afp_levels_num: number of levels (feature maps) from FPN or bottom-up-path-augmentation
        :param fully_connected_fusion: if True, fully-connected-fusion feature from PANet is switched on
        :param in_resolution: spatial resolution of input tensor
        :param group_norm: if True, 2d convolutions are continued by group normalization layer
        """
        super().__init__(group_norm)
        self.dim_in = dim_in
        self.dim_out = dim_internal
        self.out_resolution = in_resolution * 2

        self.fully_connected_fusion = fully_connected_fusion
        self.levels_num = afp_levels_num

        # conv1 for every feature map after ROIAlign
        self.conv1 = nn.ModuleList()
        for i in range(self.levels_num):
            self.conv1.append(self._conv2d_block(dim_in, dim_internal, kernel=3, stride=1, padding=1, bias=False))

        # Other convs for mask head excluding conv1 as in original Mask-RCNN
        self.conv2 = self._conv2d_block(dim_internal, dim_internal, kernel=3, stride=1, padding=1, bias=False)
        self.conv3 = self._conv2d_block(dim_internal, dim_internal, kernel=3, stride=1, padding=1, bias=False)
        self.conv4 = self._conv2d_block(dim_internal, dim_internal, kernel=3, stride=1, padding=1, bias=False)

        self.upconv = nn.ConvTranspose2d(dim_internal, dim_internal, kernel_size=2, stride=2, padding=0)
        self.segm = nn.Conv2d(dim_internal, num_cls, 1, 1, 0)

        if self.fully_connected_fusion:
            self.conv4_fc = self._conv2d_block(dim_internal, dim_internal, kernel=3, stride=1, padding=1, bias=False)
            dim_in = dim_internal
            dim_internal = dim_internal // 2
            self.conv5_fc = self._conv2d_block(dim_in, dim_internal, kernel=3, stride=1, padding=1, bias=False)
            self.fc = nn.Sequential(
                nn.Linear(dim_internal * in_resolution * in_resolution, self.out_resolution * self.out_resolution),
                nn.ReLU(inplace=True)
            )

        self._init_weights()

    def forward(self, x):
        pooled_feature = self.conv1[0](x[0])
        for i in range(1, self.levels_num):
            pooled_feature = torch.max(pooled_feature, self.conv1[i](x[i]))

        x = self.conv2(pooled_feature)
        x = self.conv3(x)

        if self.fully_connected_fusion:
            y = self.conv4_fc(x)

        x = self.conv4(x)
        x = nn.functional.relu(self.upconv(x), inplace=True)
        x = self.segm(x)

        if self.fully_connected_fusion:
            y = self.conv5_fc(y)
            y = y.view(int(y.size(0)), -1)
            y = self.fc(y)
            y = y.view(int(y.size(0)), 1, self.out_resolution, self.out_resolution)
            x = x + y

        if not self.training:
            x = torch.sigmoid(x)
        return x
