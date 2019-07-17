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

from .panet_mask_rcnn_base import PANetMaskRCNN
from ..backbones.resnet import ResNet
from ..panet import BboxHead as PANetBboxHead


class ResNeXt101PANetMaskRCNN(PANetMaskRCNN):
    def __init__(self, cls_num, force_max_output_size=False, heavier_head=False, deformable_conv=True, **kwargs):
        backbone = ResNet(base_arch='ResNet101', num_groups=32, width_per_group=8)
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(5))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, force_max_output_size=force_max_output_size,
                         heavier_head=heavier_head,
                         deformable_conv=deformable_conv, **kwargs)
        self.mask_head = self.add_segmentation_head(self.bupa.dims_out, self.cls_num, afp_levels_num=4,
                                                    fully_connected_fusion=True, group_norm=True)
        self.detection_head = self.BboxHead(self.bupa.dims_out[0], 1024, PANetMaskRCNN.detection_roi_featuremap_resolution,
                                            self.cls_num,
                                            cls_agnostic_bbox_regression=False,
                                            afp_levels_num=4,
                                            heavier_head=heavier_head, group_norm=False)

    class BboxHead(PANetBboxHead):
        """BboxHead from PANet without ReLu after fc1"""
        def __init__(self, dim_in, dim_out, resolution_in, cls_num, cls_agnostic_bbox_regression=False,
                     afp_levels_num=4, heavier_head=False, conv_head_dim=256, num_convs=4,
                     group_norm=False):
            super().__init__(dim_in, dim_out, resolution_in, cls_num, cls_agnostic_bbox_regression,
                             afp_levels_num, heavier_head, conv_head_dim, num_convs, group_norm)

        def forward(self, x):
            batch_size = int(x[0].shape[0])
            for i in range(self.levels_num):
                if self.heavier_head:
                    y = self.fc1[i](x[i])
                else:
                    y = self.fc1[i](x[i].view(batch_size, -1))

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
