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

from collections import defaultdict, Mapping

import torch
from torch import nn

from ..detection_output import DetectionOutput
from ..fpn import FPN
from ..group_norm import GroupNorm
from ..losses import rpn_loss_cls, rpn_loss_reg, detection_loss_cls, detection_loss_reg, accuracy, mask_loss
from ..prior_box import PriorBox
from ..proposal import generate_proposals
from ..proposal_gt_matcher import ProposalGTMatcher
from ..roi_feature_extractor import extract_roi_features, topk_rois
from ..rpn import RPN
from ..rpn_gt_matcher import RPNGTMatcher
from ...utils.profile import timed, print_timing_stats, Timer
from ...utils.weights import xavier_fill, msra_fill, get_group_gn


class BboxHead(nn.Module):
    """Add a ReLU MLP with two hidden layers."""

    def __init__(self, dim_in, dim_out, resolution_in, cls_num, cls_agnostic_bbox_regression=False, fc_as_conv=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.resolution_in = resolution_in
        self.cls_num = cls_num
        self.cls_agnostic_bbox_regression = cls_agnostic_bbox_regression

        box_out_dims = 4 * (1 if cls_agnostic_bbox_regression else cls_num)
        if fc_as_conv:
            self.fc1 = nn.Conv2d(dim_in, dim_out, resolution_in)
            self.fc2 = nn.Conv2d(dim_out, dim_out, 1)
            self.cls_score = nn.Conv2d(dim_out, cls_num, 1)
            self.bbox_pred = nn.Conv2d(dim_out, box_out_dims, 1)
        else:
            self.fc1 = nn.Linear(dim_in * resolution_in * resolution_in, dim_out)
            self.fc2 = nn.Linear(dim_out, dim_out)
            self.cls_score = nn.Linear(dim_out, cls_num)
            self.bbox_pred = nn.Linear(dim_out, box_out_dims)

        self._init_weights()

    def _init_weights(self):
        xavier_fill(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        xavier_fill(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def get_score_and_prediction(self, x):
        batch_size = int(x.size(0))
        cls_score = self.cls_score(x).view(batch_size, -1)
        if not self.training:
            cls_score = nn.functional.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x).view(batch_size, -1)
        return cls_score, bbox_pred

    def forward(self, x):
        if isinstance(self.fc1, nn.Linear):
            batch_size = int(x.size(0))
            x = x.view(batch_size, -1)
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        return self.get_score_and_prediction(x)


class BboxHeadWithGN(BboxHead):
    """Replace linear layers by 2d convolutions with GroupNorm"""

    def __init__(self, dim_in, dim_out, resolution_in, cls_num, cls_agnostic_bbox_regression=False,
                 conv_num=4, dim_internal=256):
        super().__init__(dim_in, dim_out, resolution_in, cls_num, cls_agnostic_bbox_regression)
        module_list = []
        for i in range(conv_num):
            module_list.extend([
                nn.Conv2d(dim_in, dim_internal, kernel_size=3, stride=1, padding=1, bias=False),
                GroupNorm(get_group_gn(dim_internal), dim_internal, eps=1e-5),
                nn.ReLU(inplace=True)
            ])
        self.convs = nn.Sequential(*module_list)

        self._init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                msra_fill(m.weight)
        self.fc2 = None

    def forward(self, x):
        batch_size = int(x.size(0))
        x = self.convs(x)
        x = nn.functional.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        return self.get_score_and_prediction(x)


class MaskHead(nn.Module):
    """X * (conv 3x3) -> convT 2x2."""

    def __init__(self, dim_in, num_convs, num_cls, dim_internal=256, dilation=2, group_norm=False):
        super().__init__()
        self.dim_in = dim_in
        self.num_convs = num_convs
        self.dim_out = dim_internal
        self.group_norm = group_norm

        module_list = []
        for i in range(num_convs):
            if self.group_norm:
                module_list.extend([
                    nn.Conv2d(dim_in, dim_internal, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
                    GroupNorm(get_group_gn(dim_internal), dim_internal, eps=1e-5),
                    nn.ReLU(inplace=True)
                ])
            else:
                module_list.extend([
                    nn.Conv2d(dim_in, dim_internal, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
                    nn.ReLU(inplace=True)
                ])
            dim_in = dim_internal
        self.conv_fcn = nn.Sequential(*module_list)

        self.upconv = nn.ConvTranspose2d(dim_internal, dim_internal, kernel_size=2, stride=2, padding=0)
        self.segm = nn.Conv2d(dim_internal, num_cls, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                msra_fill(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_fcn(x)
        x = self.upconv(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.segm(x)
        if not self.training:
            x = torch.sigmoid(x)
        return x


class FPNMaskRCNN(nn.Module):
    detection_roi_featuremap_resolution = 7
    segmentation_roi_featuremap_resolution = 14
    input_size_divisor = 32

    def __init__(self, cls_num, backbone, force_max_output_size=False,
                 group_norm=False, fc_detection_head=True, **kwargs):
        super().__init__()
        self.cls_num = cls_num
        self.force_max_output_size = force_max_output_size

        self.backbone = backbone

        self.fpn = self.add_fpn(self.backbone.dims_out, self.backbone.scales_out, group_norm=group_norm)

        self.prior_boxes, priors_num = self.add_priors_generator()
        self.rpn = self.add_rpn(priors_num, self.fpn.dims_out)

        self.detection_head, self.detection_output = self.add_detection_head(self.fpn.dims_out, self.cls_num,
                                                                             group_norm=group_norm,
                                                                             fc_detection_head=fc_detection_head)
        self.mask_head = self.add_segmentation_head(self.fpn.dims_out, self.cls_num, group_norm=group_norm)

        # For training only.
        self.rpn_batch_size = 256
        self.rpn_gt_matcher = RPNGTMatcher(straddle_threshold=0, positive_threshold=0.7,
                                           negative_threshold=0.3, positive_fraction=0.5,
                                           batch_size=self.rpn_batch_size)
        self.mask_resolution = 2 * self.segmentation_roi_featuremap_resolution
        self.proposal_gt_matcher = ProposalGTMatcher(positive_threshold=0.5, negative_threshold=0.5,
                                                     positive_fraction=0.25, batch_size=512,
                                                     target_mask_size=(self.mask_resolution, self.mask_resolution))

        self._timers = defaultdict(Timer)

    @property
    def feature_pyramid_scales(self):
        return self.fpn.scales_out

    @property
    def pre_nms_rois_count(self):
        return 2000 if self.training else 1000

    @property
    def post_nms_rois_count(self):
        return 2000 if self.training else 1000

    @staticmethod
    def add_priors_generator():
        prior_boxes = nn.ModuleList()
        widths = [[49.0, 33.0, 25.0], [89.0, 65.0, 49.0], [185.0, 129.0, 89.0], [361.0, 257.0, 185.0],
                  [729.0, 513.0, 361.0]]
        heights = [[25.0, 33.0, 49.0], [49.0, 65.0, 89.0], [97.0, 129.0, 147.0], [177.0, 257.0, 369.0],
                   [369.0, 513.0, 721.0]]
        for ws, hs in zip(widths, heights):
            prior_boxes.append(PriorBox(widths=ws, heights=hs, flatten=True))
        priors_per_level_num = list([priors.priors_num() for priors in prior_boxes])
        assert priors_per_level_num[1:] == priors_per_level_num[:-1]
        priors_num = priors_per_level_num[0]
        return prior_boxes, priors_num

    @staticmethod
    def add_fpn(dims_in, scales_in, group_norm=False, **kwargs):
        return FPN(dims_in, scales_in, 256, 256, group_norm=group_norm)

    @staticmethod
    def add_rpn(priors_num, features_dim_in):
        # RPN is shared between FPN levels.
        assert features_dim_in[1:] == features_dim_in[:-1]
        rpn = RPN(features_dim_in[0], 256, priors_num, 'sigmoid')
        return rpn

    @staticmethod
    def add_detection_head(features_dim_in, cls_num, group_norm=False, fc_detection_head=True):
        # ROI-wise detection part.
        assert features_dim_in[1:] == features_dim_in[:-1]
        dim_out = 1024
        if group_norm:
            detection_head = BboxHeadWithGN(features_dim_in[0], dim_out, 7, cls_num,
                                            cls_agnostic_bbox_regression=False, conv_num=4, dim_internal=256)
        else:
            detection_head = BboxHead(features_dim_in[0], dim_out, 7, cls_num,
                                      cls_agnostic_bbox_regression=False, fc_as_conv=not fc_detection_head)
        detection_output = DetectionOutput(cls_num, nms_threshold=0.5, score_threshold=0.05)
        return detection_head, detection_output

    @staticmethod
    def add_segmentation_head(features_dim_in, cls_num, group_norm):
        # ROI-wise segmentation part.
        assert features_dim_in[1:] == features_dim_in[:-1]
        mask_head = MaskHead(features_dim_in[0], 4, cls_num, 256, 1, group_norm=group_norm)
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
    def forward_backbone(self, im_data):
        return self.backbone(im_data)

    @timed
    def forward_fpn(self, feature_pyramid):
        return self.fpn(feature_pyramid)

    @staticmethod
    def update_return_values(return_values, extra_values):
        if isinstance(return_values, list):
            return_values.extend(extra_values)
        else:
            for k, v in extra_values.items():
                if isinstance(v, Mapping):
                    return_values[k] = FPNMaskRCNN.update_return_values(return_values.get(k, {}), v)
                else:
                    return_values[k] = v
        return return_values

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

    @timed
    def preprocess_data(self, im_data, im_info, size_divisor=32):
        with torch.no_grad():
            if isinstance(im_data, list):
                im_data = self.pad_image_data(im_data, size_divisor)
            if isinstance(im_info, list):
                im_info = torch.stack(im_info, dim=0)
        if im_data.device != im_info.device:
            im_info = im_info.to(im_data.device)
        return im_data, im_info

    @timed
    def generate_priors(self, feature_pyramid, im_data):
        priors_pyramid = []
        for prior_boxes_generator, feature_map, scale in \
                zip(self.prior_boxes, feature_pyramid, self.feature_pyramid_scales):
            priors_pyramid.append(prior_boxes_generator(feature_map, im_data,
                                                        stride_x=scale, stride_y=scale))
        return priors_pyramid

    @timed
    def get_rpn_targets(self, priors_pyramid, gt_boxes, gt_labels, gt_is_ignored, im_info):
        rpn_cls_targets, rpn_reg_targets = None, None
        if self.training:
            with torch.no_grad():
                all_priors = torch.cat(priors_pyramid, dim=0)
                rpn_cls_targets, rpn_reg_targets = self.rpn_gt_matcher(all_priors, gt_boxes, gt_labels,
                                                                       gt_is_ignored, im_info)
                priors_per_level_num = list([len(p) for p in priors_pyramid])
                rpn_cls_targets = list(zip(*[cls.split(priors_per_level_num) for cls in rpn_cls_targets]))
                rpn_reg_targets = list(zip(*[reg.split(priors_per_level_num) for reg in rpn_reg_targets]))
        return rpn_cls_targets, rpn_reg_targets

    @timed
    def forward_rpn(self, priors_pyramid, features_pyramid, im_info,
                    rpn_cls_targets=None, rpn_reg_targets=None, batch_size=1):
        rois = []
        rois_probs = []
        pyramid_level_cls_losses = []
        pyramid_level_reg_losses = []
        metrics = {}
        for pyramid_level, (priors, feature_map) in enumerate(zip(priors_pyramid, features_pyramid)):
            rpn_output = self.rpn(feature_map)
            if self.training:
                rpn_box_deltas, rpn_cls_scores = rpn_output[:2]
                loss_rpn_cls, accuracy_rpn_cls, precision_rpn_cls, recall_rpn_cls = \
                    rpn_loss_cls(rpn_cls_targets[pyramid_level], rpn_cls_scores, reduction='sum')
                loss_rpn_cls /= self.rpn_batch_size * batch_size
                loss_rpn_reg = rpn_loss_reg(rpn_cls_targets[pyramid_level], rpn_reg_targets[pyramid_level],
                                            rpn_box_deltas, reduction='sum')
                loss_rpn_reg /= self.rpn_batch_size * batch_size

                level = '/{}'.format(pyramid_level)
                metrics.update({
                    'losses/rpn/cls' + level: loss_rpn_cls.detach().unsqueeze(0),
                    'losses/rpn/reg' + level: loss_rpn_reg.detach().unsqueeze(0),
                    'metrics/rpn/cls_accuracy' + level: accuracy_rpn_cls.detach().unsqueeze(0),
                    'metrics/rpn/cls_precision' + level: precision_rpn_cls.detach().unsqueeze(0),
                    'metrics/rpn/cls_recall' + level: recall_rpn_cls.detach().unsqueeze(0)
                })
                pyramid_level_cls_losses.append(loss_rpn_cls)
                pyramid_level_reg_losses.append(loss_rpn_reg)

            with torch.no_grad():
                proposal_rois, proposal_rois_probs = generate_proposals(priors, rpn_output, im_info,
                                                                        pre_nms_count=self.pre_nms_rois_count,
                                                                        post_nms_count=self.post_nms_rois_count,
                                                                        force_max_output_size=self.force_max_output_size)
                rois.append(proposal_rois)
                rois_probs.append(proposal_rois_probs)

        loss = None
        if pyramid_level_cls_losses:
            cls_loss = sum(pyramid_level_cls_losses)
            reg_loss = sum(pyramid_level_reg_losses)
            loss = cls_loss + reg_loss
            metrics['losses/rpn/cls'] = cls_loss.detach().unsqueeze(0)
            metrics['losses/rpn/reg'] = reg_loss.detach().unsqueeze(0)
            metrics['losses/rpn'] = loss.detach().unsqueeze(0)

        return rois, rois_probs, metrics, loss

    @timed
    def process_proposals(self, rois, rois_probs, batch_size):
        # Transpose from feature pyramid level -> batch element representation to
        # batch element -> feature pyramid level one.
        # And retain only post_nms_count top scoring proposals among all pyramid levels for one image.
        rois = list(zip(*rois))
        rois_probs = list(zip(*rois_probs))
        assert len(rois) == batch_size, '{} != {}'.format(len(rois), batch_size)
        assert len(rois_probs) == batch_size, '{} != {}'.format(len(rois_probs), batch_size)
        for i, (image_rois, image_rois_probs) in enumerate(zip(rois, rois_probs)):
            image_rois = torch.cat(image_rois, dim=0)
            image_rois_probs = torch.cat(image_rois_probs, dim=0)
            rois[i] = topk_rois(image_rois, image_rois_probs, max_rois=self.post_nms_rois_count,
                                use_stub=not self.training)
            rois_probs[i] = image_rois_probs
        return rois, rois_probs

    @timed
    def get_targets(self, rois, gt_boxes, gt_labels, gt_masks):
        roi_cls_targets, roi_reg_targets, roi_mask_targets = None, None, None
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
        return rois, roi_cls_targets, roi_reg_targets, roi_mask_targets

    @timed
    def forward_detection_head(self, roi_features):
        return self.detection_head(roi_features)

    @timed
    def forward_mask_head(self, roi_features):
        return self.mask_head(roi_features)

    @timed
    def forward_detection_output(self, rois, raw_bbox_pred, raw_cls_score, im_info, batch_idx):
        if len(rois) == 1:
            rois = rois[0]
        self.detection_output.force_max_output_size = self.force_max_output_size
        boxes, classes, scores, batch_ids = self.detection_output(rois, raw_bbox_pred, raw_cls_score, im_info, batch_idx)
        return boxes, classes, scores, batch_ids

    def dummy_detections(self, device):
        boxes = torch.zeros((1, 4), device=device, dtype=torch.float32)
        classes = torch.zeros(1, device=device, dtype=torch.long)
        scores = torch.zeros(1, device=device, dtype=torch.float32)
        batch_ids = torch.zeros(1, device=device, dtype=torch.long)
        raw_mask_output = torch.zeros((1, self.cls_num, self.mask_resolution, self.mask_resolution),
                                      device=device, dtype=torch.float32)
        return boxes, classes, scores, batch_ids, raw_mask_output

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
                mask_roi_features = mask_roi_features[positive_indices]
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
                return_values.extend(self.dummy_detections(im_info.device))
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
        roi_features, rois = extract_roi_features(rois, feature_pyramid,
                                                  pyramid_scales=tuple(self.feature_pyramid_scales),
                                                  output_size=output_size,
                                                  sampling_ratio=2,
                                                  distribute_rois_between_levels=True,
                                                  preserve_rois_order=preserve_order,
                                                  use_stub=not self.training)
        roi_features = torch.cat([x for x in roi_features if x is not None], dim=0)
        return roi_features, rois

    def print_timing(self):
        print_timing_stats(self._timers)
