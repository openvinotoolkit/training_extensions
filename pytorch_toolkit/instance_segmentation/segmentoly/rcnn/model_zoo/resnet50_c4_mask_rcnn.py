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
from torch import nn

from ..backbones.backbone import freeze_params_recursive, freeze_mode_recursive
from ..backbones.resnet import ResNetBody, ResBlock
from ..detection_output import DetectionOutput
from ..losses import rpn_loss_cls, rpn_loss_reg, detection_loss_cls, detection_loss_reg, accuracy, mask_loss
from ..prior_box import PriorBox
from ..proposal import generate_proposals
from ..proposal_gt_matcher import ProposalGTMatcher
from ..roi_feature_extractor import extract_roi_features, topk_rois
from ..rpn import RPN
from ..rpn_gt_matcher import RPNGTMatcher


class BboxHead(nn.Module):
    def __init__(self, dim_in, cls_num, cls_agnostic_bbox_regression=False):
        super().__init__()
        self.cls_num = cls_num
        self.cls_agnostic_bbox_regression = cls_agnostic_bbox_regression

        self.cls_score = nn.Conv2d(dim_in, cls_num, 1)
        box_out_dims = 4 * (1 if cls_agnostic_bbox_regression else cls_num)
        self.bbox_pred = nn.Conv2d(dim_in, box_out_dims, 1)
        self._init_weights()
        # Freeze parameters of an affine transforms in Batch Norm layers.
        freeze_params_recursive(self, freeze=True, types=(nn.BatchNorm2d, ))

    def _init_weights(self):
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def train(self, mode=True):
        super().train(mode=mode)
        # Always keep Batch Norms in eval mode.
        freeze_mode_recursive(self, train_mode=False, types=(nn.BatchNorm2d, ))

    def forward(self, x):
        batch_size = int(x.size(0))
        cls_score = self.cls_score(x).view(batch_size, -1)
        if not self.training:
            cls_score = nn.functional.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x).view(batch_size, -1)
        return cls_score, bbox_pred


class MaskHead(nn.Module):
    def __init__(self, dim_in, dim_internal, num_cls):
        super().__init__()
        self.upconv5 = nn.ConvTranspose2d(dim_in, dim_internal, 2, 2, 0)
        self.segm = nn.Conv2d(dim_internal, num_cls, 1, 1, 0)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.upconv5.weight)
        nn.init.constant_(self.upconv5.bias, 0)
        nn.init.kaiming_uniform_(self.segm.weight)
        nn.init.constant_(self.segm.bias, 0)

    def forward(self, x):
        x = self.upconv5(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.segm(x)
        if not self.training:
            x = torch.sigmoid(x)
        return x


class ResNet50C4MaskRCNN(nn.Module):
    roi_featuremap_resolution = 14

    def __init__(self, cls_num, force_max_output_size=False, **kwargs):
        super().__init__()
        self.cls_num = cls_num
        self.force_max_output_size = force_max_output_size

        self.backbone = ResNetBody(block_counts=(3, 4, 6), res_block=ResBlock, num_groups=1,
                                   width_per_group=64, res5_dilation=1)
        self.backbone.freeze_stages_params(range(2))
        self.backbone.freeze_stages_bns(range(5))
        self.backbone.set_output_stages((3, ))

        dim = self.backbone.dims_out[0]

        widths = [46.0, 92.0, 184.0, 368.0, 736.0, 32.0, 64.0, 128.0, 256.0, 512.0, 22.0, 44.0, 88.0, 176.0, 352.0]
        heights = [24.0, 48.0, 96.0, 192.0, 384.0, 32.0, 64.0, 128.0, 256.0, 512.0, 44.0, 88.0, 176.0, 352.0, 704.0]
        self.prior_boxes = PriorBox(widths=widths, heights=heights, flatten=True)

        self.rpn = RPN(dim, 1024, self.prior_boxes.priors_num(), 'sigmoid')

        self.common_detection_mask_head, head_dim = ResNetBody.add_stage(ResBlock, dim, 2048, 512, 3, stride_init=2)
        self.global_pooling = nn.AvgPool2d(7)

        self.cls_agnostic_bbox_regression = False
        self.detection_head = BboxHead(head_dim, cls_num,
                                       cls_agnostic_bbox_regression=self.cls_agnostic_bbox_regression)
        self.detection_output = DetectionOutput(cls_num, nms_threshold=0.5, score_threshold=0.05)
        self.mask_head = MaskHead(head_dim, 256, cls_num)

        # For training only.
        self.rpn_gt_matcher = RPNGTMatcher(straddle_threshold=0, positive_threshold=0.7,
                                           negative_threshold=0.3, positive_fraction=0.5,
                                           batch_size=256)
        self.proposal_gt_matcher = ProposalGTMatcher(positive_threshold=0.5, negative_threshold=0.5,
                                                     positive_fraction=0.25, batch_size=512, target_mask_size=(14, 14))

    @property
    def pre_nms_rois_count(self):
        return 12000 if self.training else 6000

    @property
    def post_nms_rois_count(self):
        return 2000 if self.training else 1000

    def preprocess_data(self, im_data, im_info, size_divisor=1):
        with torch.no_grad():
            if isinstance(im_data, list):
                im_data = self.pad_image_data(im_data, size_divisor)
            if isinstance(im_info, list):
                im_info = torch.stack(im_info, dim=0)
        if im_data.device != im_info.device:
            im_info = im_info.to(im_data.device)
        return im_data, im_info

    @staticmethod
    def pad_image_data(image_blobs, size_divisor):
        target_height = max([entry.shape[-2] for entry in image_blobs])
        target_width = max([entry.shape[-1] for entry in image_blobs])
        target_height = (target_height + size_divisor - 1) // size_divisor * size_divisor
        target_width = (target_width + size_divisor - 1) // size_divisor * size_divisor
        for i, image_blob in enumerate(image_blobs):
            image_blobs[i] = torch.nn.functional.pad(image_blob, (0, target_width - image_blob.shape[-1],
                                                                  0, target_height - image_blob.shape[-2]))
        return torch.stack(image_blobs, dim=0)

    def forward(self, im_data, im_info, gt_boxes=None, gt_labels=None, gt_is_ignored=None, gt_masks=None, **kwargs):
        if self.training:
            # In case of training return a dict rather than a list.
            return_values = {}
        else:
            return_values = []

        im_data, im_info = self.preprocess_data(im_data, im_info, 16)
        batch_size = im_data.shape[0]

        backbone_features = self.backbone(im_data)[0]

        rpn_output = self.rpn(backbone_features)

        with torch.no_grad():
            priors = self.prior_boxes(backbone_features, im_data,
                                      stride_x=self.backbone.scales_out[0],
                                      stride_y=self.backbone.scales_out[0])

        if self.training:
            with torch.no_grad():
                rpn_cls_targets, rpn_reg_targets = self.rpn_gt_matcher(priors, gt_boxes, gt_labels, gt_is_ignored, im_info)
            rpn_box_deltas, rpn_cls_scores = rpn_output[:2]
            loss_rpn_cls, accuracy_rpn_cls, precision_rpn_cls, recall_rpn_cls = rpn_loss_cls(rpn_cls_targets, rpn_cls_scores)
            loss_rpn_reg = rpn_loss_reg(rpn_cls_targets, rpn_reg_targets, rpn_box_deltas)
            return_values['losses/rpn/cls'] = loss_rpn_cls.unsqueeze(0)
            return_values['losses/rpn/reg'] = loss_rpn_reg.unsqueeze(0)
            return_values['metrics/rpn/cls_accuracy'] = accuracy_rpn_cls.unsqueeze(0)
            return_values['metrics/rpn/cls_precision'] = precision_rpn_cls.unsqueeze(0)
            return_values['metrics/rpn/cls_recall'] = recall_rpn_cls.unsqueeze(0)
            return_values['losses/rpn'] = (loss_rpn_cls + loss_rpn_reg).unsqueeze(0)

        with torch.no_grad():
            rois, rois_probs = generate_proposals(priors, rpn_output, im_info,
                                                  pre_nms_count=self.pre_nms_rois_count,
                                                  post_nms_count=self.post_nms_rois_count,
                                                  force_max_output_size=self.force_max_output_size)
            if batch_size == 1:
                rois[0] = topk_rois(rois[0], rois_probs[0], max_rois=self.post_nms_rois_count,
                                    use_stub=not self.training)

        if self.training:
            with torch.no_grad():
                rois, roi_cls_targets, roi_reg_targets, roi_mask_targets = \
                    self.proposal_gt_matcher(rois, gt_boxes, gt_labels, gt_masks)
                # Sanity checks.
                assert len(rois) == len(roi_cls_targets)
                assert len(rois) == len(roi_reg_targets)
                for im_rois, im_roi_cls_targets, im_roi_reg_targets in zip(rois, roi_cls_targets, roi_reg_targets):
                    assert im_rois.shape[0] == im_roi_cls_targets.shape[0]
                    assert im_rois.shape[0] == im_roi_reg_targets.shape[0]

        roi_features, rois = self.extract_roi_features(rois, backbone_features)

        common_roi_features = self.common_detection_mask_head(roi_features)
        assert common_roi_features.shape[0] == roi_features.shape[0]

        detection_roi_features = self.global_pooling(common_roi_features)
        raw_cls_score, raw_bbox_pred = self.detection_head(detection_roi_features)
        assert raw_cls_score.shape[0] == roi_features.shape[0]
        assert raw_bbox_pred.shape[0] == roi_features.shape[0]

        if self.training:
            loss_cls = detection_loss_cls(raw_cls_score, roi_cls_targets)
            all_targets = torch.cat(roi_cls_targets)
            valid_mask = all_targets >= 0
            accuracy_cls = accuracy(torch.argmax(raw_cls_score[valid_mask], dim=1), all_targets[valid_mask])
            loss_reg = detection_loss_reg(raw_bbox_pred, roi_cls_targets, roi_reg_targets,
                                          self.cls_agnostic_bbox_regression)
            return_values['losses/detection/cls'] = loss_cls.unsqueeze(0)
            return_values['losses/detection/reg'] = loss_reg.unsqueeze(0)
            return_values['metrics/detection/cls_accuracy'] = accuracy_cls.unsqueeze(0)

            with torch.no_grad():
                positive_indices = (all_targets > 0).nonzero().view(-1)
                positive_mask_targets = torch.cat(roi_mask_targets, dim=0).index_select(0, positive_indices)
                positive_cls_targets = all_targets.index_select(0, positive_indices)
            mask_roi_features = common_roi_features.index_select(0, positive_indices)
            raw_mask_output = self.mask_head(mask_roi_features)
            assert raw_mask_output.shape[0] == positive_indices.shape[0]

            loss_mask = mask_loss(raw_mask_output, positive_cls_targets, positive_mask_targets)
            return_values['losses/mask'] = loss_mask.unsqueeze(0)
        else:
            with torch.no_grad():
                if len(rois) == 1:
                    rois = rois[0]
                self.detection_output.force_max_output_size = self.force_max_output_size
                boxes, classes, scores, batch_ids = self.detection_output(rois, raw_bbox_pred, raw_cls_score, im_info)
            if boxes.numel() > 0:
                if batch_size > 1:
                    rois_to_segment = list([boxes.index_select(0, (batch_ids == image_id).nonzero().reshape(-1))
                                            for image_id in range(batch_size)])
                else:
                    rois_to_segment = [boxes, ]
                roi_mask_features, rois_to_segment = self.extract_roi_features(rois_to_segment, backbone_features)
                roi_mask_features = self.common_detection_mask_head(roi_mask_features)
                raw_mask_output = self.mask_head(roi_mask_features)
                return_values.extend((boxes, classes, scores, batch_ids, raw_mask_output))
            else:
                # Gathering empty tensors could be an issue in DataParallel, so explicitly return Nones.
                return_values.extend((None, None, None, None, None))

        if self.training:
            loss_total = loss_rpn_cls + loss_rpn_reg + loss_cls + loss_reg + loss_mask
            return_values['losses/TOTAL'] = loss_total.unsqueeze(0)
            return return_values, loss_total
        else:
            # Dummy operation on outputs for Model Optimizer.
            for ret_val in return_values:
                if ret_val is not None:
                    ret_val += 0
            return return_values

    def extract_roi_features(self, rois, features):
        roi_features, rois = extract_roi_features(rois, [features, ],
                                                  pyramid_scales=tuple(self.backbone.scales_out),
                                                  output_size=self.roi_featuremap_resolution,
                                                  sampling_ratio=0,
                                                  distribute_rois_between_levels=True,
                                                  preserve_rois_order=True,
                                                  use_stub=not self.training)
        roi_features = roi_features[0]
        rois = rois[0]
        return roi_features, rois
