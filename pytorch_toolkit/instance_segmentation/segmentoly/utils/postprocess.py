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

import cv2
import numpy as np
import pycocotools.mask as mask_util

from .blob import to_numpy
from .boxes import expand_boxes


def postprocess_batch(batch_ids, scores, classes, boxes, raw_cls_masks,
                      batch_size, im_h, im_w, im_scale_y=None, im_scale_x=None, im_scale=None,
                      full_image_masks=True, encode_masks=False,
                      confidence_threshold=0.0):
    boxes_all = [np.empty((0, 4), dtype=np.float32) for _ in range(batch_size)]
    scores_all = [np.empty((0, ), dtype=np.float32) for _ in range(batch_size)]
    classes_all = [np.empty((0, ), dtype=np.float32) for _ in range(batch_size)]
    raw_masks_all = [None for _ in range(batch_size)]
    masks_all = [[] for _ in range(batch_size)]

    if batch_ids is None:
        return scores_all, classes_all, boxes_all, masks_all

    batch_ids = to_numpy(batch_ids)

    num_objs_per_batch = []
    for i in range(batch_size):
        num_objs_per_batch.append(np.count_nonzero(batch_ids == i))

    begin = 0
    for i in range(0, len(num_objs_per_batch)):
        end = begin + num_objs_per_batch[i]
        # Scale boxes back to the original image
        boxes_all[i] = boxes[begin:end]
        scores_all[i] = scores[begin:end]
        classes_all[i] = classes[begin:end]
        raw_masks_all[i] = raw_cls_masks[begin:end]
        begin = end

    # Resize segmentation masks to fit corresponding bounding boxes.
    for i in range(batch_size):
        scores_all[i], classes_all[i], boxes_all[i], masks_all[i] = \
            postprocess(scores_all[i], classes_all[i], boxes_all[i], raw_masks_all[i],
                        im_h[i], im_w[i], im_scale[i], full_image_masks, encode_masks,
                        confidence_threshold)

    return scores_all, classes_all, boxes_all, masks_all


def postprocess(scores, classes, boxes, raw_cls_masks,
                im_h, im_w, im_scale, full_image_masks=True, encode_masks=False,
                confidence_threshold=0.0):
    no_detections = (np.empty((0, ), dtype=np.float32), np.empty((0, ), dtype=np.float32),\
                     np.empty((0, 4), dtype=np.float32), [])
    if scores is None:
        return no_detections

    scale = im_scale

    scores = to_numpy(scores)
    classes = to_numpy(classes)
    boxes = to_numpy(boxes)
    raw_cls_masks = to_numpy(raw_cls_masks)

    confidence_filter = scores > confidence_threshold
    scores = scores[confidence_filter]
    classes = classes[confidence_filter]
    boxes = boxes[confidence_filter]
    raw_cls_masks = list(segm for segm, is_valid in zip(raw_cls_masks, confidence_filter) if is_valid)

    if len(scores) == 0:
        return no_detections

    boxes = boxes / scale
    classes = classes.astype(np.uint32)
    masks = []
    for box, cls, raw_mask in zip(boxes, classes, raw_cls_masks):
        raw_cls_mask = raw_mask[cls, ...]
        mask = segm_postprocess(box, raw_cls_mask, im_h, im_w, full_image_masks, encode_masks)
        masks.append(mask)

    return scores, classes, boxes, masks


def segm_postprocess(box, raw_cls_mask, im_h, im_w, full_image_mask=True, encode=False):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_boxes(box[np.newaxis, :],
                                raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0))[0]
    extended_box = extended_box.astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)

    if full_image_mask:
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                     (x0 - extended_box[0]):(x1 - extended_box[0])]
    else:
        original_box = box.astype(int)
        x0, y0 = np.clip(original_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(original_box[2:] + 1, a_min=0, a_max=[im_w, im_h])
        im_mask = np.ascontiguousarray(mask[(y0 - original_box[1]):(y1 - original_box[1]),
                                            (x0 - original_box[0]):(x1 - original_box[0])])

    if encode:
        im_mask = mask_util.encode(np.array(im_mask[:, :, np.newaxis].astype(np.uint8), order='F'))[0]
        im_mask['counts'] = im_mask['counts'].decode('utf-8')

    return im_mask
