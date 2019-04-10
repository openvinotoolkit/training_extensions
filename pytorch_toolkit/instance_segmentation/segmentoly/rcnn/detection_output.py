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

from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .nms_function import nms
from ..utils.boxes import bbox_transform, clip_boxes_to_image
from ..utils.profile import Timer, DummyTimer


class DetectionOutputFunction(torch.autograd.Function):
    _timers = defaultdict(DummyTimer)

    @staticmethod
    def symbolic(g, all_rois, all_box_deltas, all_cls_scores, im_info, batch_idx, num_classes,
                 post_nms_count=2000, nms_threshold=0.7, score_threshold=0.01, max_detections_per_image=100,
                 class_agnostic_box_regression=False, force_max_output_size=False):
        return g.op('ExperimentalDetectronDetectionOutput', all_rois, all_box_deltas, all_cls_scores, im_info,
                    num_classes_i=num_classes,
                    post_nms_count_i=post_nms_count, nms_threshold_f=nms_threshold, score_threshold_f=score_threshold,
                    max_detections_per_image_i=max_detections_per_image,
                    class_agnostic_box_regression_i=int(class_agnostic_box_regression),
                    deltas_weights_f=[10, 10, 5, 5],
                    max_delta_log_wh_f=np.log(1000. / 16.),
                    outputs=4)

    @staticmethod
    def forward(ctx, all_rois, all_box_deltas, all_cls_scores, im_info, batch_idx, num_classes,
                post_nms_count=2000, nms_threshold=0.7, score_threshold=0.01, max_detections_per_image=100,
                class_agnostic_box_regression=False, force_max_output_size=False):
        if isinstance(all_rois, torch.Tensor):
            all_rois = [all_rois, ]
        device = all_box_deltas.device
        out_boxes = []
        out_scores = []
        out_classes = []

        images_in_batch = im_info.size(0)
        slice_point_start = 0
        for image_idx in range(images_in_batch):
            im_boxes = [torch.empty(0, 4, dtype=all_box_deltas.dtype, device=device), ]
            im_scores = [torch.empty(0, dtype=all_box_deltas.dtype, device=device), ]
            im_classes = [torch.empty(0, dtype=torch.long, device=device), ]
            with DetectionOutputFunction._timers['detection_output.batch_mask']:
                rois = all_rois[image_idx]
                image_rois_num = rois.shape[0]
                slice_point_end = slice_point_start + image_rois_num
                box_deltas = all_box_deltas[slice_point_start:slice_point_end]
                cls_scores = all_cls_scores[slice_point_start:slice_point_end]
                slice_point_start = slice_point_end

            with DetectionOutputFunction._timers['detection_output.apply_deltas']:
                box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
                if class_agnostic_box_regression:
                    # Remove predictions for bg class (compatible with MSRA code)
                    box_deltas = box_deltas[:, -4:]
                pred_boxes = bbox_transform(rois, box_deltas, weights=(10., 10., 5., 5.))
                pred_boxes = clip_boxes_to_image(pred_boxes.view(-1, 4),
                                                 width=int(im_info[image_idx, 1]),
                                                 height=int(im_info[image_idx, 0])).view(*pred_boxes.shape)
                if class_agnostic_box_regression:
                    pred_boxes = pred_boxes.repeat(1, num_classes + 1)

            with DetectionOutputFunction._timers['detection_output.per_class_proc']:
                # Apply threshold on detection probabilities and apply NMS
                score_mask = torch.transpose(cls_scores > score_threshold, 1, 0).contiguous()
                # Skip j = 0, because it's the background class
                for j in range(1, num_classes):
                    with DetectionOutputFunction._timers['detection_output.score_thresholding']:
                        mask = score_mask[j]
                        valid_boxes = torch.nonzero(mask).view(-1)
                        if len(valid_boxes) == 0:
                            continue
                        scores = cls_scores[valid_boxes, j]
                        boxes = pred_boxes[valid_boxes, j * 4:(j + 1) * 4]

                    if nms_threshold > 0:
                        with DetectionOutputFunction._timers['detection_output.nms']:
                            boxes, scores = nms(boxes, scores, nms_threshold)

                    if post_nms_count > 0:
                        with DetectionOutputFunction._timers['detection_output.post_nms_topN']:
                            boxes = boxes[:post_nms_count, :]
                            scores = scores[:post_nms_count]

                    with DetectionOutputFunction._timers['detection_output.concat_results']:
                        n = boxes.size(0)
                        im_boxes.append(boxes)
                        im_scores.append(scores)
                        im_classes.append(torch.full((n,), j, dtype=torch.long, device=device))

            im_boxes = torch.cat(im_boxes, dim=0)
            im_scores = torch.cat(im_scores, dim=0)
            im_classes = torch.cat(im_classes, dim=0)

            boxes_num = im_boxes.shape[0]
            if boxes_num > max_detections_per_image:
                topk_indices = torch.topk(im_scores, max_detections_per_image)[1]
                im_boxes = im_boxes[topk_indices]
                im_scores = im_scores[topk_indices]
                im_classes = im_classes[topk_indices]
            elif force_max_output_size and boxes_num < max_detections_per_image:
                extra_boxes_num = max_detections_per_image - boxes_num
                im_boxes = torch.cat((im_boxes, im_boxes.new_zeros((extra_boxes_num, 4))), dim=0)
                im_scores = torch.cat((im_scores, im_scores.new_zeros((extra_boxes_num,))), dim=0)
                im_classes = torch.cat((im_classes, im_classes.new_zeros((extra_boxes_num,))), dim=0)

            out_boxes.append(im_boxes)
            out_scores.append(im_scores)
            out_classes.append(im_classes)

        if batch_idx is None:
            out_batch_ids = torch.cat(tuple(torch.full((len(b),), i, device=b.device, dtype=torch.long)
                                            for i, b in enumerate(out_boxes)), dim=0)
        else:
            out_batch_ids = torch.cat(tuple(torch.full((len(b),), i, device=b.device, dtype=torch.long)
                                            for i, b in zip(batch_idx, out_boxes)), dim=0)
        out_boxes = torch.cat(out_boxes, dim=0)
        out_scores = torch.cat(out_scores, dim=0)
        out_classes = torch.cat(out_classes, dim=0)

        # print_timing_stats(DetectionOutputFunction._timers)

        return out_boxes, out_classes, out_scores, out_batch_ids


def detection_output(all_rois, all_box_deltas, all_cls_scores, im_info, batch_idx,
                     num_classes, post_nms_count=2000, nms_threshold=0.7, score_threshold=0.01,
                     max_detections_per_image=100, class_agnostic_box_regression=False, force_max_output_size=False):
    return DetectionOutputFunction.apply(all_rois, all_box_deltas, all_cls_scores, im_info, batch_idx, num_classes,
                                         post_nms_count, nms_threshold, score_threshold, max_detections_per_image,
                                         class_agnostic_box_regression, force_max_output_size)


class DetectionOutput(nn.Module):
    def __init__(self, num_classes, post_nms_count=2000, nms_threshold=0.7, score_threshold=0.01,
                 max_detections_per_image=100, class_agnostic_box_regression=False, force_max_output_size=False):
        super().__init__()
        self.force_max_output_size = force_max_output_size
        self._num_classes = num_classes
        self._post_nms_count = post_nms_count
        self._nms_threshold = nms_threshold
        self._score_threshold = score_threshold
        self._max_detections_per_image = max_detections_per_image
        self._class_agnostic_box_regression = class_agnostic_box_regression
        self._timers = defaultdict(Timer)

    def forward(self, all_rois, all_box_deltas, all_cls_scores, im_info, batch_idx=None):
        return detection_output(all_rois, all_box_deltas, all_cls_scores, im_info, batch_idx, self._num_classes,
                                self._post_nms_count, self._nms_threshold, self._score_threshold,
                                self._max_detections_per_image, self._class_agnostic_box_regression,
                                self.force_max_output_size)
