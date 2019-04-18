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

from collections import OrderedDict

from torch import nn

from .backbone import Backbone
from ..group_norm import GroupNorm
from ...utils.weights import get_group_gn


class ResBlock(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 stride_1x1=True, track_running_batch_stat=True):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if stride_1x1 else (1, stride)

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = nn.BatchNorm2d(innerplanes, track_running_stats=track_running_batch_stat)

        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
                               padding=1 * dilation, dilation=dilation, groups=group)
        self.bn2 = nn.BatchNorm2d(innerplanes, track_running_stats=track_running_batch_stat)

        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes, track_running_stats=track_running_batch_stat)

        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes, track_running_stats=track_running_batch_stat)
            )
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for submodule in self.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_uniform_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
            elif isinstance(submodule, nn.BatchNorm2d):
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)
                if submodule.weight is not None:
                    nn.init.constant_(submodule.weight, 1)
                if submodule.running_mean is not None:
                    nn.init.constant_(submodule.running_mean, 0)
                if submodule.running_var is not None:
                    nn.init.constant_(submodule.running_var, 1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBlockWithGN(ResBlock):
    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 stride_1x1=False, track_running_batch_stat=True):
        super().__init__(inplanes, outplanes, innerplanes, stride, dilation, group,
                         stride_1x1, track_running_batch_stat)
        self.bn1 = GroupNorm(get_group_gn(innerplanes), innerplanes, eps=1e-5, affine=True)
        self.bn2 = GroupNorm(get_group_gn(innerplanes), innerplanes, eps=1e-5, affine=True)
        self.bn3 = GroupNorm(get_group_gn(outplanes), outplanes, eps=1e-5, affine=True)
        if self.downsample is not None:
            self.downsample[1] = GroupNorm(get_group_gn(outplanes), outplanes, eps=1e-5, affine=True)


class ResBlockWithFusedBN(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, stride_1x1=True):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if stride_1x1 else (1, stride)

        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=True)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=True,
                               padding=1 * dilation, dilation=dilation, groups=group)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=True)

        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for submodule in self.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_uniform_(submodule.weight)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBody(Backbone):
    def __init__(self, block_counts, res_block=ResBlock, num_groups=1, width_per_group=64, res5_dilation=1):
        super().__init__()
        self.block_counts = block_counts
        self.num_groups = num_groups
        self.width_per_group = width_per_group
        self.res5_dilation = res5_dilation

        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2
        stage_dims_out = []

        dim_in = 64
        stages = [nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, dim_in, 7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(dim_in)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])),
        ]

        stage_dims_out.append(dim_in)
        dim_bottleneck = num_groups * width_per_group
        stage, dim_in = self.add_stage(res_block, dim_in, 256, dim_bottleneck, block_counts[0],
                                       dilation=1, stride_init=1, groups_num=num_groups)
        stages.append(stage)
        stage_dims_out.append(dim_in)

        stage, dim_in = self.add_stage(res_block, dim_in, 512, dim_bottleneck * 2, block_counts[1],
                                       dilation=1, stride_init=2, groups_num=num_groups)
        stages.append(stage)
        stage_dims_out.append(dim_in)

        stage, dim_in = self.add_stage(res_block, dim_in, 1024, dim_bottleneck * 4, block_counts[2],
                                       dilation=1, stride_init=2, groups_num=num_groups)
        stages.append(stage)
        stage_dims_out.append(dim_in)

        if len(block_counts) == 4:
            stride_init = 2 if res5_dilation == 1 else 1
            stage, dim_in = self.add_stage(res_block, dim_in, 2048, dim_bottleneck * 8, block_counts[3],
                                           dilation=res5_dilation, stride_init=stride_init, groups_num=num_groups)
            stages.append(stage)
            stage_dims_out.append(dim_in)

        self.stages = nn.Sequential(OrderedDict([
            ('stage_{}'.format(i), stage) for i, stage in enumerate(stages)
        ]))

        self.dims_in = (3,)
        self.scales_in = (1,)
        self._all_dims_out = tuple(stage_dims_out)
        self._all_scales_out = (4, ) + tuple(2 ** (i + 1) for i in range(1, len(stages)))
        self.set_output_stages(range(len(stages)))

    @staticmethod
    def add_stage(res_block, inplanes, outplanes, innerplanes, nblocks, stride_init=2, dilation=1, groups_num=1):
        res_blocks = []
        stride = stride_init
        for _ in range(nblocks):
            res_blocks.append(res_block(inplanes, outplanes, innerplanes, stride, dilation, groups_num))
            inplanes = outplanes
            stride = 1

        return nn.Sequential(*res_blocks), outplanes


class ResNet(ResNetBody):
    def __init__(self, base_arch='ResNet50', fused_batch_norms=False, group_norm=False, stride_1x1=True,
                 num_groups=1, width_per_group=64, res5_dilation=1):
        if group_norm:
            res_block_type = ResBlockWithGN
        else:
            res_block_type = ResBlockWithFusedBN if fused_batch_norms else ResBlock

        def res_block(*args, **kwargs):
            return res_block_type(*args, **kwargs, stride_1x1=stride_1x1)

        if base_arch == 'ResNet50':
            block_counts = (3, 4, 6, 3)
        elif base_arch == 'ResNet101':
            block_counts = (3, 4, 23, 3)
        elif base_arch == 'ResNet152':
            block_counts = (3, 8, 36, 3)
        else:
            raise ValueError('Invalid ResNet architecture "{}".'.format(base_arch))

        super().__init__(block_counts=block_counts, res_block=res_block, num_groups=num_groups,
                         width_per_group=width_per_group, res5_dilation=res5_dilation)
        if group_norm:
            dim_in = self.dims_out[0]
            self.stages.stage_0.bn1 = GroupNorm(get_group_gn(dim_in), dim_in, eps=1e-5, affine=True)
