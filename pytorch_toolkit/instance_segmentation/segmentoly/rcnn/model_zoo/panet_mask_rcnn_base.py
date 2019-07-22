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

from .fpn_mask_rcnn_base import FPNMaskRCNN
from .. import panet
from ..detection_output import DetectionOutput
from ..losses import detection_loss_cls, detection_loss_reg, accuracy, mask_loss
from ..roi_feature_extractor import extract_roi_features
from ...utils.profile import timed


class PANetMaskRCNN(FPNMaskRCNN):
    def __init__(self, cls_num, backbone, force_max_output_size=False, afp_levels_num=4, heavier_head=False,
                 group_norm=False, fully_connected_fusion=True, deformable_conv=False, **kwargs):
        super().__init__(cls_num, backbone, force_max_output_size, group_norm, **kwargs)

        if deformable_conv:
            self.bupa = panet.BottomUpPathAugmentationWithDeformConv(output_levels=5, dims_in=self.fpn.dims_out,
                                                                     scales_in=self.fpn.scales_out,
                                                                     dim_out=256, group_norm=True)
        else:
            self.bupa = panet.BottomUpPathAugmentation(output_levels=5, dims_in=self.fpn.dims_out,
                                                       scales_in=self.fpn.scales_out, dim_out=256, group_norm=True)

        self.detection_head, self.detection_output = self.add_detection_head(self.bupa.dims_out, self.cls_num,
                                                                             afp_levels_num, heavier_head, group_norm)
        self.mask_head = self.add_segmentation_head(self.bupa.dims_out, self.cls_num, afp_levels_num,
                                                    fully_connected_fusion, group_norm)

    @property
    def feature_pyramid_scales(self):
        return self.bupa.scales_out

    @staticmethod
    def add_detection_head(features_dim_in, cls_num, afp_levels_num=4, heavier_head=False, group_norm=False, **kwargs):
        # ROI-wise detection part.
        assert features_dim_in[1:] == features_dim_in[:-1]
        detection_head = panet.BboxHead(features_dim_in[0], 1024, PANetMaskRCNN.detection_roi_featuremap_resolution,
                                        cls_num,
                                        cls_agnostic_bbox_regression=False,
                                        afp_levels_num=afp_levels_num,
                                        heavier_head=heavier_head, group_norm=group_norm)
        detection_output = DetectionOutput(cls_num, nms_threshold=0.5, score_threshold=0.05)
        return detection_head, detection_output

    @staticmethod
    def add_segmentation_head(features_dim_in, cls_num, afp_levels_num=4,
                              fully_connected_fusion=False, group_norm=False):
        # ROI-wise segmentation part.
        assert features_dim_in[1:] == features_dim_in[:-1]
        mask_head = panet.MaskHead(features_dim_in[0], cls_num, 256,
                                   afp_levels_num=afp_levels_num,
                                   fully_connected_fusion=fully_connected_fusion,
                                   in_resolution=PANetMaskRCNN.segmentation_roi_featuremap_resolution,
                                   group_norm=group_norm)
        return mask_head

    @timed
    def forward(self, im_data, im_info, gt_boxes=None, gt_labels=None, gt_is_ignored=None, gt_masks=None,
                batch_idx=None, **kwargs):
        if self.training:
            # In case of training return a dict rather than a list.
            return_values = {}
        else:
            return_values = []

        im_data, im_info = self.preprocess_data(im_data, im_info, self.input_size_divisor)
        batch_size = im_data.shape[0]

        backbone_features = self.forward_backbone(im_data)
        backbone_features = self.forward_fpn(backbone_features)
        backbone_features = self.forward_bupa(backbone_features)

        with torch.no_grad():
            priors_pyramid = self.generate_priors(backbone_features, im_data)
        rpn_cls_targets, rpn_reg_targets = self.get_rpn_targets(priors_pyramid, gt_boxes, gt_labels,
                                                                gt_is_ignored, im_info)
        rois, rois_probs, rpn_metrics, rpn_loss = self.forward_rpn(priors_pyramid, backbone_features, im_info,
                                                                   rpn_cls_targets, rpn_reg_targets, batch_size)
        return_values = self.update_return_values(return_values, rpn_metrics)
        rois, rois_probs = self.process_proposals(rois, rois_probs, batch_size)
        rois, roi_cls_targets, roi_reg_targets, roi_mask_targets = self.get_targets(rois, gt_boxes, gt_labels, gt_masks)

        # Last pyramid level is used only for RPN part of the net, so, remove it now.
        backbone_features = backbone_features[:-1]
        instance_heads_output, instance_heads_loss = self.forward_instance_heads(im_info, backbone_features, rois,
                                                                                 roi_cls_targets, roi_reg_targets,
                                                                                 roi_mask_targets,
                                                                                 batch_idx=batch_idx)
        return_values = self.update_return_values(return_values, instance_heads_output)

        if self.training:
            total_loss = rpn_loss + instance_heads_loss
            return_values['losses/TOTAL'] = total_loss.detach()
            return return_values, total_loss
        else:
            # Dummy operation on outputs for Model Optimizer.
            for ret_val in return_values:
                if ret_val is not None:
                    ret_val += 0
            return return_values

    @timed
    def forward_bupa(self, feature_pyramid):
        return self.bupa(feature_pyramid)

    @timed
    def forward_instance_heads(self, im_info, feature_pyramid, rois,
                               roi_cls_targets=None, roi_reg_targets=None,
                               roi_mask_targets=None, batch_size=1, batch_idx=None):
        detection_roi_features, rois = self.extract_roi_features(rois, feature_pyramid,
                                                                 output_size=self.detection_roi_featuremap_resolution)
        raw_cls_score, raw_bbox_pred = self.forward_detection_head(detection_roi_features)

        if self.training:
            return_values = {}
            loss_cls = detection_loss_cls(raw_cls_score, roi_cls_targets)
            all_targets = torch.cat(roi_cls_targets)
            valid_mask = all_targets >= 0
            accuracy_cls = accuracy(torch.argmax(raw_cls_score[valid_mask], dim=1), all_targets[valid_mask])
            loss_reg = detection_loss_reg(raw_bbox_pred, roi_cls_targets, roi_reg_targets,
                                          self.detection_head.cls_agnostic_bbox_regression)
            return_values['losses/detection/cls'] = loss_cls.detach().unsqueeze(0)
            return_values['losses/detection/reg'] = loss_reg.detach().unsqueeze(0)
            return_values['metrics/detection/cls_accuracy'] = accuracy_cls.detach().unsqueeze(0)

            with torch.no_grad():
                positive_indices = (all_targets > 0).nonzero().view(-1)

            if len(positive_indices) > 0:
                with torch.no_grad():
                    positive_mask_targets = torch.cat(roi_mask_targets, dim=0).index_select(0, positive_indices)
                    positive_cls_targets = all_targets.index_select(0, positive_indices)
                mask_roi_features, rois = self.extract_roi_features(rois, feature_pyramid,
                                                                    output_size=self.segmentation_roi_featuremap_resolution)
                for i, roi_feats in enumerate(mask_roi_features):
                    mask_roi_features[i] = roi_feats[positive_indices]
                # mask_roi_features = mask_roi_features[positive_indices]
                raw_mask_output = self.forward_mask_head(mask_roi_features)
                assert raw_mask_output.shape[0] == positive_indices.shape[0]

                assert raw_mask_output.shape[0] == positive_cls_targets.shape[0]
                assert raw_mask_output.shape[0] == positive_mask_targets.shape[0]
                loss_mask = mask_loss(raw_mask_output, positive_cls_targets, positive_mask_targets)
                return_values['losses/mask'] = loss_mask.detach().unsqueeze(0)
                loss = (loss_cls + loss_reg + loss_mask).unsqueeze(0)
            else:
                return_values['losses/mask'] = torch.zeros_like(loss_cls).detach().unsqueeze(0)
                loss = (loss_cls + loss_reg).unsqueeze(0)
        else:
            return_values = []
            with torch.no_grad():
                boxes, classes, scores, batch_ids = self.forward_detection_output(rois, raw_bbox_pred,
                                                                                  raw_cls_score, im_info,
                                                                                  batch_idx)

            if sum(len(im_boxes) for im_boxes in boxes) == 0:
                return_values.extend((None, None, None, None, None))
            else:
                if batch_idx is not None and len(batch_idx) > 1:
                    rois = list([boxes.index_select(0, (batch_ids == image_id).nonzero().reshape(-1))
                                 for image_id in batch_idx])
                else:
                    rois = [boxes, ]
                # Extract features for every detected box preserving the order.
                mask_roi_features, _ = self.extract_roi_features(rois, feature_pyramid,
                                                                 output_size=self.segmentation_roi_featuremap_resolution)
                raw_mask_output = self.forward_mask_head(mask_roi_features)
                return_values.extend((boxes, classes, scores, batch_ids, raw_mask_output))
            loss = None
        return return_values, loss

    @timed
    def extract_roi_features(self, rois, feature_pyramid, output_size=7, preserve_order=True):
        feature_map_roi_features = []
        feature_map_rois = []
        for feature_map, scale in zip(feature_pyramid, self.feature_pyramid_scales):
            rf, r = extract_roi_features(rois, [feature_map, ],
                                         pyramid_scales=(scale, ),
                                         output_size=output_size,
                                         sampling_ratio=2,
                                         distribute_rois_between_levels=True,
                                         preserve_rois_order=preserve_order,
                                         use_stub=not self.training)
            feature_map_roi_features.append(rf)
            feature_map_rois.append(r)

        # from  image -> level -> roi features  to  level -> image -> roi features
        # roi_features = list(zip(*feature_map_roi_features))
        roi_features = feature_map_roi_features
        for i, level_roi_feats in enumerate(roi_features):
            roi_features[i] = torch.cat(level_roi_feats, dim=0)

        return feature_map_roi_features, rois

