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

import itertools

import torch
import torch.nn as nn
from segmentoly.rcnn.losses import detection_loss_cls, detection_loss_reg, accuracy, mask_loss
from segmentoly.rcnn.model_zoo.fpn_mask_rcnn_base import FPNMaskRCNN, timed, extract_roi_features


class FPNTextMaskRCNN(FPNMaskRCNN):
    weight_text_loss = 0.25
    mask_text = False

    def __init__(self, cls_num, backbone, force_max_output_size=False,
                 group_norm=False, fc_detection_head=True, num_chars=38, **kwargs):
        super().__init__(cls_num, backbone, force_max_output_size,
                         group_norm, fc_detection_head, **kwargs)

        self.num_chars = num_chars
        self.text_recogn_head = self.add_text_recogn_head()

        self.export_mode = False

    def add_text_recogn_head(*args, **kwargs):
        raise NotImplementedError

    @timed
    def forward(self, im_data, im_info, gt_boxes=None, gt_labels=None, gt_is_ignored=None,
                gt_masks=None,
                batch_idx=None, gt_texts=None, **kwargs):
        if self.training:
            # In case of training return a dict rather than a list.
            return_values = {}
        else:
            return_values = []

        im_data, im_info = self.preprocess_data(im_data, im_info, self.input_size_divisor)
        batch_size = im_data.shape[0]

        backbone_features = self.forward_backbone(im_data)
        fpn_features = self.forward_fpn(backbone_features)
        with torch.no_grad():
            priors_pyramid = self.generate_priors(fpn_features, im_data)
        rpn_cls_targets, rpn_reg_targets = self.get_rpn_targets(priors_pyramid, gt_boxes, gt_labels,
                                                                gt_is_ignored, im_info)
        rois, rois_probs, rpn_metrics, rpn_loss = self.forward_rpn(priors_pyramid,
                                                                   fpn_features, im_info,
                                                                   rpn_cls_targets, rpn_reg_targets,
                                                                   batch_size)
        return_values = self.update_return_values(return_values, rpn_metrics)
        rois, rois_probs = self.process_proposals(rois, rois_probs, batch_size)
        rois, roi_cls_targets, roi_reg_targets, roi_mask_targets, roi_text_targets = self.get_targets(
            rois, gt_boxes, gt_labels, gt_masks, gt_texts)

        # Last pyramid level is used only for RPN part of the net, so, remove it now.
        fpn_features = fpn_features[:-1]
        instance_heads_output, instance_heads_loss = self.forward_instance_heads(im_info,
                                                                                 fpn_features,
                                                                                 backbone_features,
                                                                                 rois,
                                                                                 roi_cls_targets,
                                                                                 roi_reg_targets,
                                                                                 roi_mask_targets,
                                                                                 roi_text_targets,
                                                                                 batch_idx=batch_idx)
        return_values = self.update_return_values(return_values, instance_heads_output)

        if self.training:
            total_loss = rpn_loss + instance_heads_loss
            return_values['losses/TOTAL'] = total_loss.detach()
            return return_values, total_loss
        else:
            return return_values

    def extract_text_features(self, **kwargs):
        text_roi_features, rois = self.extract_roi_features(
            kwargs['rois'],
            kwargs['feature_pyramid'],
            output_size=2 * self.segmentation_roi_featuremap_resolution
        )
        return text_roi_features, rois

    @timed
    def get_targets(self, rois, gt_boxes, gt_labels, gt_masks, gt_texts=None):
        roi_cls_targets, roi_reg_targets, roi_mask_targets, roi_text_targets = None, None, None, None
        if self.training:
            with torch.no_grad():
                rois, roi_cls_targets, roi_reg_targets, roi_mask_targets, roi_text_targets = self.proposal_gt_matcher(
                    rois, gt_boxes, gt_labels, gt_masks, gt_texts)
                # Sanity checks.
                assert len(rois) == len(roi_cls_targets)
                assert len(rois) == len(roi_reg_targets)
                for im_rois, im_roi_cls_targets, im_roi_reg_targets in zip(rois, roi_cls_targets,
                                                                           roi_reg_targets):
                    assert im_rois.shape[0] == im_roi_cls_targets.shape[0]
                    assert im_rois.shape[0] == im_roi_reg_targets.shape[0]

        return rois, roi_cls_targets, roi_reg_targets, roi_mask_targets, roi_text_targets

    @timed
    def forward_text_recogn_head(self, roi_features, text_targets=None, masks=None):
        return self.text_recogn_head(roi_features, text_targets, masks)

    def dummy_detections(self, device):
        boxes = torch.zeros((1, 4), device=device, dtype=torch.float32)
        classes = torch.zeros(1, device=device, dtype=torch.long)
        scores = torch.zeros(1, device=device, dtype=torch.float32)
        batch_ids = torch.zeros(1, device=device, dtype=torch.long)
        raw_mask_output = torch.zeros((1, self.cls_num, self.mask_resolution, self.mask_resolution),
                                      device=device, dtype=torch.float32)
        raw_text_output = self.text_recogn_head.dummy_forward()
        raw_text_output = raw_text_output.to(raw_mask_output.device)
        return boxes, classes, scores, batch_ids, raw_mask_output, raw_text_output

    @timed
    def forward_instance_heads(self, im_info, feature_pyramid, backbone_features, rois,
                               roi_cls_targets=None, roi_reg_targets=None,
                               roi_mask_targets=None, roi_text_targets=None,
                               batch_size=1, batch_idx=None):
        detection_roi_features, rois = self.extract_roi_features(rois, feature_pyramid,
                                                                 output_size=self.detection_roi_featuremap_resolution)
        raw_cls_score, raw_bbox_pred = self.forward_detection_head(detection_roi_features)

        if self.training:
            return_values = {}
            loss_cls = detection_loss_cls(raw_cls_score, roi_cls_targets)
            all_targets = torch.cat(roi_cls_targets)
            valid_mask = all_targets >= 0
            accuracy_cls = accuracy(torch.argmax(raw_cls_score[valid_mask], dim=1),
                                    all_targets[valid_mask])
            loss_reg = detection_loss_reg(raw_bbox_pred, roi_cls_targets, roi_reg_targets,
                                          self.detection_head.cls_agnostic_bbox_regression)
            return_values['losses/detection/cls'] = loss_cls.detach().unsqueeze(0)
            return_values['losses/detection/reg'] = loss_reg.detach().unsqueeze(0)
            return_values['metrics/detection/cls_accuracy'] = accuracy_cls.detach().unsqueeze(0)

            with torch.no_grad():
                positive_indices = (all_targets > 0).nonzero().view(-1)

            if len(positive_indices) > 0:
                with torch.no_grad():
                    positive_mask_targets = torch.cat(roi_mask_targets, dim=0).index_select(0,
                                                                                            positive_indices)
                    positive_cls_targets = all_targets.index_select(0, positive_indices)
                    all_text_targets = list(itertools.chain.from_iterable(roi_text_targets))
                    positive_text_indices = torch.tensor(
                        [i for i in positive_indices if all_text_targets[i]])
                    positive_text_targets = [all_text_targets[i] for i in positive_text_indices]

                mask_roi_features, rois = self.extract_roi_features(rois, feature_pyramid,
                                                                    output_size=self.segmentation_roi_featuremap_resolution)
                mask_roi_features = mask_roi_features[positive_indices]
                raw_mask_output = self.forward_mask_head(mask_roi_features)

                loss_mask = mask_loss(raw_mask_output, positive_cls_targets, positive_mask_targets)
                return_values['losses/mask'] = loss_mask.detach().unsqueeze(0)

                loss_text, accuracy_text = torch.tensor(0.0, device=im_info.device), torch.tensor(
                    0.0, device=im_info.device)
                if len(positive_text_indices) > 0:
                    text_roi_features, rois = self.extract_text_features(rois=rois,
                                                                         backbone_features=backbone_features,
                                                                         feature_pyramid=feature_pyramid)
                    text_roi_features = text_roi_features[positive_text_indices]

                    text_mask = None
                    if self.mask_text:
                        text_mask = positive_mask_targets
                        postitve_text_indices_in_positive_indices = torch.tensor(
                            [i for i, j in enumerate(positive_indices) if all_text_targets[j]])
                        text_mask = text_mask[postitve_text_indices_in_positive_indices]
                        text_mask = text_mask.unsqueeze(1)

                    loss_text, accuracy_text = self.forward_text_recogn_head(text_roi_features,
                                                                             positive_text_targets,
                                                                             masks=text_mask)
                    loss_text *= self.weight_text_loss

                return_values['losses/text'] = loss_text.detach().unsqueeze(0)
                return_values['metrics/TEXT/accuracy'] = accuracy_text.unsqueeze(0)
                loss = (loss_cls + loss_reg + loss_mask + loss_text).unsqueeze(0)
            else:
                return_values['losses/mask'] = torch.zeros_like(loss_cls).detach().unsqueeze(0)
                return_values['losses/text'] = torch.zeros_like(loss_cls).detach().unsqueeze(0)
                loss = (loss_cls + loss_reg).unsqueeze(0)
        else:
            return_values = []
            with torch.no_grad():
                boxes, classes, scores, batch_ids = self.forward_detection_output(rois,
                                                                                  raw_bbox_pred,
                                                                                  raw_cls_score,
                                                                                  im_info,
                                                                                  batch_idx)

            if sum(len(im_boxes) for im_boxes in boxes) == 0:
                return_values.extend(self.dummy_detections(im_info.device))
            else:
                if batch_idx is not None and len(batch_idx) > 1:
                    rois = list(
                        [boxes.index_select(0, (batch_ids == image_id).nonzero().reshape(-1))
                         for image_id in batch_idx])
                else:
                    rois = [boxes, ]
                # Extract features for every detected box preserving the order.
                mask_roi_features, _ = self.extract_roi_features(rois, feature_pyramid,
                                                                 output_size=self.segmentation_roi_featuremap_resolution)
                raw_mask_output = self.forward_mask_head(mask_roi_features)
                text_roi_features, _ = self.extract_text_features(rois=rois,
                                                                  backbone_features=backbone_features,
                                                                  feature_pyramid=feature_pyramid)

                text_mask = None
                if self.mask_text:
                    text_mask = raw_mask_output[:, 1, :, :]
                    text_mask = text_mask.unsqueeze(1)

                if self.export_mode:
                    raw_text_output = text_roi_features
                else:
                    raw_text_output = self.forward_text_recogn_head(text_roi_features,
                                                                    masks=text_mask)
                    raw_text_output = raw_text_output.permute((1, 0, 2))

                # Following stuff is workaround for model optimizer.
                delta = 0.0000000001
                boxes = boxes + delta
                classes = (classes.float() + delta).long()
                scores = scores + delta
                raw_text_output = raw_text_output + delta

                return_values.extend(
                    (boxes, classes, scores, batch_ids, raw_mask_output, raw_text_output))
            loss = None
        return return_values, loss


class FusedTextFeatures(nn.Module):

    def __init__(self, input_dims, input_scales, out_dim, out_scale, indexes):
        super().__init__()

        assert len(input_dims) == len(input_scales)

        input_dims = [input_dims[i] for i in indexes]
        input_scales = [input_scales[i] for i in indexes]

        self.layers = []
        for i, (scale, dim) in enumerate(zip(input_scales, input_dims)):
            if scale == out_scale:
                self.layers.append(nn.Conv2d(dim, out_dim, 1, 1))
            elif scale > out_scale:
                assert scale % out_scale == 0
                self.layers.append(nn.Sequential(
                    nn.Conv2d(dim, out_dim, 1, 1),
                    nn.UpsamplingBilinear2d(scale_factor=scale // out_scale)
                ))
            else:
                raise NotImplementedError

        self.layers = nn.ModuleList(self.layers)

    def forward(self, inputs):
        assert isinstance(inputs, list)
        outputs_sum = 0

        for i, input in enumerate(inputs):
            outputs_sum += self.layers[i](input)

        return outputs_sum


class FPNTextMaskRCNNWithBackboneTextFeatures(FPNTextMaskRCNN):

    def __init__(self, cls_num, backbone, force_max_output_size=False,
                 group_norm=False, fc_detection_head=True, num_chars=38, **kwargs):
        super().__init__(cls_num, backbone, force_max_output_size,
                         group_norm, fc_detection_head, num_chars, **kwargs)

        self.backbone_outputs = [0, 1]
        self.out_scale = 4

        self.fused_text_features = FusedTextFeatures(backbone.dims_out, backbone.scales_out,
                                                     self.text_recogn_head.encoder_dim_input,
                                                     out_scale=self.out_scale,
                                                     indexes=self.backbone_outputs)

    def extract_text_features(self, **kwargs):
        features = [kwargs['backbone_features'][i] for i in self.backbone_outputs]
        fused_text_features = self.fused_text_features(features)

        text_roi_features, rois = extract_roi_features(
            kwargs['rois'],
            [fused_text_features],
            pyramid_scales=[self.out_scale],
            output_size=self.segmentation_roi_featuremap_resolution * 2,
            sampling_ratio=2,
            distribute_rois_between_levels=True,
            preserve_rois_order=True,
            use_stub=not self.training

        )
        text_roi_features = torch.cat([x for x in text_roi_features if x is not None], dim=0)
        return text_roi_features, rois
