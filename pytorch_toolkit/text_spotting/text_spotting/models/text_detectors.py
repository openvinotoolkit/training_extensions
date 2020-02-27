"""
 Copyright (c) 2020 Intel Corporation

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
from segmentoly.rcnn.model_zoo.instance_segmentation_security_0050 import \
    BottomUpPathAugmentationBN, FPN, PriorBox, MaskHeadBN, RPNLite, BboxHead3FC, DetectionOutput
from .backbones import get_backbone
from .mask_rcnn import str_to_class as str_to_text_spotter_class
from .mask_rcnn.proposal_gt_matcher import ProposalGTMatcher
from .text_recognition_heads import str_to_class as str_to_text_recognition_head_class


def make_text_detector(class_name=None,
                       backbone=None,
                       bupa_dim_out=0,
                       fpn_dim_out=128,
                       rpn_dim=128,
                       mask_head_dim=128,
                       prior_boxes_sizes=None,
                       segm_roi_featuremap_resolution=7,
                       use_text_masking=False,
                       text_recognition_head=None):
    class TextDetector(str_to_text_spotter_class[class_name]):
        segmentation_roi_featuremap_resolution = segm_roi_featuremap_resolution
        mask_text = use_text_masking

        def __init__(self, cls_num, **kwargs):
            super().__init__(cls_num, get_backbone(**backbone), **kwargs)

            self.bupa = None
            if bupa_dim_out:
                self.bupa = BottomUpPathAugmentationBN(output_levels=5, dims_in=self.fpn.dims_out,
                                                       scales_in=self.fpn.scales_out,
                                                       dim_out=bupa_dim_out, group_norm=False)

            r = self.segmentation_roi_featuremap_resolution
            self.proposal_gt_matcher = ProposalGTMatcher(positive_threshold=0.5,
                                                         negative_threshold=0.5,
                                                         positive_fraction=0.25, batch_size=256,
                                                         target_mask_size=(2 * r, 2 * r))

        @staticmethod
        def add_fpn(dims_in, scales_in, **kwargs):
            return FPN(dims_in, scales_in, fpn_dim_out, fpn_dim_out, group_norm=False)

        @staticmethod
        def add_priors_generator():
            prior_boxes = nn.ModuleList()
            widths = prior_boxes_sizes['widths']
            heights = prior_boxes_sizes['heights']

            scale_factor = 1.0
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
        def add_segmentation_head(features_dim_in, cls_num, **kwargs):
            # ROI-wise segmentation part.
            assert features_dim_in[1:] == features_dim_in[:-1]
            mask_head = MaskHeadBN(features_dim_in[0], 6, cls_num, mask_head_dim, 1)
            return mask_head

        @staticmethod
        def add_rpn(priors_num, features_dim_in):
            # RPN is shared between FPN levels.
            assert features_dim_in[1:] == features_dim_in[:-1]
            rpn = RPNLite(features_dim_in[0], rpn_dim, priors_num, 'sigmoid')
            return rpn

        @staticmethod
        def add_detection_head(features_dim_in, cls_num, fc_detection_head=True, **kwargs):
            # ROI-wise detection part.
            assert features_dim_in[1:] == features_dim_in[:-1]
            dim_out = 512
            detection_head = BboxHead3FC(features_dim_in[0], dim_out, 7, cls_num,
                                         cls_agnostic_bbox_regression=False,
                                         fc_as_conv=not fc_detection_head)
            detection_output = DetectionOutput(cls_num, nms_threshold=0.5, score_threshold=0.05,
                                               post_nms_count=100, max_detections_per_image=100)
            return detection_head, detection_output

        @staticmethod
        def add_text_recogn_head():
            return str_to_text_recognition_head_class[text_recognition_head['name']](
                **text_recognition_head['param']
            )

        @property
        def pre_nms_rois_count(self):
            return 2000 if self.training else 300

        @property
        def post_nms_rois_count(self):
            return 2000 if self.training else 300

        def forward_fpn(self, feature_pyramid):
            x = self.fpn(feature_pyramid)
            return self.bupa(x) if self.bupa else x

    return TextDetector
