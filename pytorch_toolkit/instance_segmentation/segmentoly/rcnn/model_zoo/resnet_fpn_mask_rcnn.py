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

import torch.nn as nn

from .fpn_mask_rcnn_base import FPNMaskRCNN, BboxHead, DetectionOutput, MaskHead
from ..backbones.resnet import ResNet
from ..prior_box import PriorBox
from ..proposal_gt_matcher import ProposalGTMatcher


class ResNet50FPNMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, force_max_output_size=False, **kwargs):
        backbone = ResNet(base_arch='ResNet50')
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(5))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, force_max_output_size=force_max_output_size, **kwargs)


class ResNet50FPNMaskRCNNDemo(ResNet50FPNMaskRCNN):
    def __init__(self, cls_num, force_max_output_size=False, **kwargs):
        super().__init__(cls_num, force_max_output_size=force_max_output_size, **kwargs)
        self.data_normalizer = nn.BatchNorm2d(3, eps=0, affine=True, track_running_stats=True)
        self.data_normalizer.weight[0] = 1.0
        self.data_normalizer.weight[1] = 1.0
        self.data_normalizer.weight[2] = 1.0
        self.data_normalizer.bias[0] = -102.9801
        self.data_normalizer.bias[1] = -115.9465
        self.data_normalizer.bias[2] = -122.7717

    @property
    def pre_nms_rois_count(self):
        return 2000 if self.training else 100

    @property
    def post_nms_rois_count(self):
        return 2000 if self.training else 100

    def preprocess_data(self, im_data, im_info, size_divisor=32):
        im_data, im_info = super().preprocess_data(im_data, im_info, size_divisor)
        im_data = self.data_normalizer(im_data)
        return im_data, im_info


class ResNet101FPNMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, **kwargs):
        backbone = ResNet(base_arch='ResNet101')
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(5))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, **kwargs)


class ResNeXt101FPNMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, **kwargs):
        backbone = ResNet(base_arch='ResNet101', num_groups=32, width_per_group=8, stride_1x1=False)
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(5))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, **kwargs)


class ResNeXt10164x4dFPNMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, **kwargs):
        backbone = ResNet(base_arch='ResNet101', num_groups=64, width_per_group=4, stride_1x1=False)
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(5))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, **kwargs)


class ResNeXt152FPNMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, **kwargs):
        backbone = ResNet(base_arch='ResNet152', num_groups=32, width_per_group=8, stride_1x1=False)
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(5))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, **kwargs)


class ResNeXt152sFPNMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, **kwargs):
        backbone = ResNet(base_arch='ResNet152', num_groups=32, width_per_group=8)
        backbone.freeze_stages_params(range(2))
        backbone.freeze_stages_bns(range(5))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, **kwargs)


class ResNet50FPNGNMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, **kwargs):
        backbone = ResNet(base_arch='ResNet50', group_norm=True)
        backbone.freeze_stages_params(range(2))
        backbone.set_output_stages((1, 2, 3, 4))
        super().__init__(cls_num, backbone, group_norm=True, **kwargs)


class ResNet50FPNMaskRCNNLightSegmHead(ResNet50FPNMaskRCNN):
    segmentation_roi_featuremap_resolution = 7

    def __init__(self, cls_num, **kwargs):
        super().__init__(cls_num, **kwargs)
        r = self.segmentation_roi_featuremap_resolution
        self.proposal_gt_matcher = ProposalGTMatcher(positive_threshold=0.5, negative_threshold=0.5,
                                                     positive_fraction=0.25, batch_size=256,
                                                     target_mask_size=(2 * r, 2 * r))

    @staticmethod
    def add_priors_generator():
        prior_boxes = nn.ModuleList()
        widths = [[49.0, 33.0, 25.0], [89.0, 65.0, 49.0], [185.0, 129.0, 89.0], [361.0, 257.0, 185.0],
                  [729.0, 513.0, 361.0]]
        heights = [[25.0, 33.0, 49.0], [49.0, 65.0, 89.0], [97.0, 129.0, 147.0], [177.0, 257.0, 369.0],
                   [369.0, 513.0, 721.0]]
        scale_factor = 0.4
        for ws, hs in zip(widths, heights):
            if scale_factor != 1.0:
                for i in range(len(ws)):
                    ws[i] *= scale_factor
                for i in range(len(hs)):
                    hs[i] *= scale_factor
            prior_boxes.append(PriorBox(widths=ws, heights=hs, flatten=True, use_cache=True))
        priors_per_level_num = list([priors.priors_num() for priors in prior_boxes])
        assert priors_per_level_num[1:] == priors_per_level_num[:-1]
        priors_num = priors_per_level_num[0]
        return prior_boxes, priors_num

    @staticmethod
    def add_segmentation_head(features_dim_in, cls_num, group_norm):
        # ROI-wise segmentation part.
        assert features_dim_in[1:] == features_dim_in[:-1]
        mask_head = MaskHead(features_dim_in[0], 4, cls_num, 128, 1, group_norm=group_norm)
        return mask_head


class ResNet50FPNMaskRCNNLightSegmHeadDemo(ResNet50FPNMaskRCNNLightSegmHead):
    def __init__(self, cls_num, **kwargs):
        super().__init__(cls_num, **kwargs)
        self.data_normalizer = nn.BatchNorm2d(3, eps=0, affine=True, track_running_stats=True)
        self.data_normalizer.weight[0] = 1.0
        self.data_normalizer.weight[1] = 1.0
        self.data_normalizer.weight[2] = 1.0
        self.data_normalizer.bias[0] = -102.9801
        self.data_normalizer.bias[1] = -115.9465
        self.data_normalizer.bias[2] = -122.7717

    @property
    def pre_nms_rois_count(self):
        return 2000 if self.training else 100

    @property
    def post_nms_rois_count(self):
        return 2000 if self.training else 100

    def preprocess_data(self, im_data, im_info, size_divisor=32):
        im_data, im_info = super().preprocess_data(im_data, im_info, size_divisor)
        im_data = self.data_normalizer(im_data)
        return im_data, im_info
