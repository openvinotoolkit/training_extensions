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

import numpy as np

from segmentoly.utils.blob import to_numpy
from segmentoly.utils.postprocess import segm_postprocess


def postprocess_batch(batch_ids, scores, classes, boxes, raw_cls_masks, raw_texts,
                      batch_size, im_h, im_w, im_scale_y=None, im_scale_x=None, im_scale=None,
                      full_image_masks=True, encode_masks=False,
                      confidence_threshold=0.0):
    boxes_all = [np.empty((0, 4), dtype=np.float32) for _ in range(batch_size)]
    scores_all = [np.empty((0,), dtype=np.float32) for _ in range(batch_size)]
    classes_all = [np.empty((0,), dtype=np.float32) for _ in range(batch_size)]
    raw_masks_all = [None for _ in range(batch_size)]
    masks_all = [[] for _ in range(batch_size)]
    raw_texts_all = [None for _ in range(batch_size)]

    if batch_ids is None:
        return scores_all, classes_all, boxes_all, masks_all

    scale_x = im_scale_x
    scale_y = im_scale_y
    if im_scale is not None:
        scale_x = im_scale
        scale_y = im_scale
    assert len(scale_x) == len(scale_y)

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
        raw_texts_all[i] = raw_texts[begin:end]
        begin = end

    # Resize segmentation masks to fit corresponding bounding boxes.
    for i in range(batch_size):
        scores_all[i], classes_all[i], boxes_all[i], masks_all[i], raw_texts_all[i] = \
            postprocess(scores_all[i], classes_all[i], boxes_all[i], raw_masks_all[i],
                        raw_texts_all[i],
                        im_h[i], im_w[i], scale_y[i], scale_x[i], None,
                        full_image_masks, encode_masks,
                        confidence_threshold)

    return scores_all, classes_all, boxes_all, masks_all, raw_texts_all


def postprocess(scores, classes, boxes, raw_cls_masks, raw_texts,
                im_h, im_w, im_scale_y=None, im_scale_x=None, im_scale=None,
                full_image_masks=True, encode_masks=False,
                confidence_threshold=0.0):
    no_detections = (np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), \
                     np.empty((0, 4), dtype=np.float32), [], [])
    if scores is None:
        return no_detections

    scale = im_scale
    if scale is None:
        assert (im_scale_x is not None) and (im_scale_y is not None)
        scale = [im_scale_x, im_scale_y, im_scale_x, im_scale_y]

    scores = to_numpy(scores)
    classes = to_numpy(classes)
    boxes = to_numpy(boxes)
    raw_cls_masks = to_numpy(raw_cls_masks)
    raw_texts = to_numpy(raw_texts)

    confidence_filter = scores > confidence_threshold
    scores = scores[confidence_filter]
    classes = classes[confidence_filter]
    boxes = boxes[confidence_filter]
    raw_texts = raw_texts[confidence_filter]
    raw_cls_masks = list(
        segm for segm, is_valid in zip(raw_cls_masks, confidence_filter) if is_valid)

    if len(scores) == 0:
        return no_detections

    boxes = boxes / scale
    classes = classes.astype(np.uint32)
    masks = []
    for box, cls, raw_mask in zip(boxes, classes, raw_cls_masks):
        raw_cls_mask = raw_mask[cls, ...]
        mask = segm_postprocess(box, raw_cls_mask, im_h, im_w, full_image_masks, encode_masks)
        masks.append(mask)

    return scores, classes, boxes, masks, raw_texts
