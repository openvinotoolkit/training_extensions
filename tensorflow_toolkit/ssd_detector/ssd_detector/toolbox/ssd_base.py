# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from ssd_detector.toolbox.layers import channel_to_last, get_spatial_dims
from ssd_detector.toolbox.priors import prior_box, prior_box_clusterd, prior_box_specs


# pylint: disable=too-many-instance-attributes
class SSDBase:
  def __init__(self, input_shape, num_classes=2, overlap_threshold=0.5, data_format='NHWC'):
    assert len(input_shape) == 4
    assert data_format in ['NHWC', 'NCHW']

    self.input_shape = input_shape
    self.num_classes = num_classes
    self.clip = False
    self.overlap_threshold = overlap_threshold
    self.mbox_loc = None
    self.mbox_conf = None
    self.mbox_priorbox = None
    self.logits = None
    self.predictions = None
    self.detections = None
    self.priors_array = None
    self.priors = None
    self.priors_info = []
    self.flattens_for_tfmo = []
    self.data_format = data_format

  # pylint: disable=too-many-arguments
  def _add_single_ssd_head(self, blob, num_classes, num_anchors, prefix, suffix=''):
    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None, padding='SAME', normalizer_params=None):
      if len(blob.shape) == 4:
        locs = slim.conv2d(blob, num_anchors * 4, (3, 3),
                           scope='{}_mbox_loc{}'.format(prefix, suffix), data_format=self.data_format)
        locs = channel_to_last(locs, data_format=self.data_format)
        locs = slim.flatten(locs)
        conf = slim.conv2d(blob, num_anchors * num_classes, (3, 3), biases_initializer=tf.constant_initializer(0.0),
                           scope='{}_mbox_conf{}'.format(prefix, suffix), data_format=self.data_format)
        conf = channel_to_last(conf, data_format=self.data_format)
        conf = slim.flatten(conf)
        self.flattens_for_tfmo.extend([locs, conf])
      elif len(blob.shape) == 2:
        locs = slim.fully_connected(blob, num_anchors * 4, activation_fn=None,
                                    scope='{}_mbox_loc{}'.format(prefix, suffix))
        conf = slim.fully_connected(blob, num_anchors * num_classes, activation_fn=None,
                                    scope='{}_mbox_conf{}'.format(prefix, suffix))
      else:
        raise Exception('Unsupported input blob shape for SSD.')
      return conf, locs

  # pylint: disable=too-many-locals
  def create_heads(self, connections, params_dicts):
    image_size = get_spatial_dims(self.input_shape, self.data_format)

    with tf.variable_scope('ssd_heads'):
      scores, bboxes, priors = [], [], []
      priors_array = []
      for head_id, (tensor, params) in enumerate(zip(connections, params_dicts)):
        with tf.variable_scope('head_{}'.format(head_id)):
          if 'clustered_sizes' in params:
            priors_fn = prior_box_clusterd
          elif 'box_specs' in params:
            priors_fn = prior_box_specs
          else:
            priors_fn = prior_box
          fn_params = {k: v for k, v in params.items() if not k == 'prefix' and not k == 'suffix'}
          fn_params['data_format'] = self.data_format

          numpy_priors, num_priors_per_pixel = priors_fn(tensor, image_size, **fn_params)
          assert np.prod(get_spatial_dims(tensor)) * num_priors_per_pixel == numpy_priors.shape[2] // 4
          self.priors_info.append([get_spatial_dims(tensor), num_priors_per_pixel])
          priors_array.append(numpy_priors)

          priors_tensor = tf.convert_to_tensor(numpy_priors, name='{}_priorbox'.format(params['prefix']))
          priors.append(priors_tensor)

          score, bbox = self._add_single_ssd_head(tensor, self.num_classes, num_priors_per_pixel, params['prefix'],
                                                  params.get('suffix', ''))
          scores.append(score)
          bboxes.append(bbox)

      with tf.name_scope('concat_reshape_softmax'):
        # Gather all predictions
        self.mbox_loc = tf.concat(bboxes, axis=-1, name='mbox_loc') if len(connections) > 1 else bboxes[0]
        self.mbox_conf = tf.concat(scores, axis=-1, name='mbox_conf') if len(connections) > 1 else scores[0]
        self.mbox_priorbox = tf.concat(priors, axis=-1, name='mbox_priorbox') if len(connections) > 1 else priors[0]

        total_priors = self.mbox_conf.get_shape()[-1] // self.num_classes
        self.mbox_loc = tf.reshape(self.mbox_loc, shape=(-1, total_priors, 4), name='mbox_loc_final')
        self.logits = tf.reshape(self.mbox_conf, shape=(-1, total_priors, self.num_classes), name='mbox_conf_logits')
        self.mbox_conf = tf.sigmoid(self.logits, name='mbox_conf_final')
        # self.mbox_conf = tf.nn.softmax(self.logits, name='mbox_conf_final')

    self.priors_array = np.reshape(np.concatenate(priors_array, axis=-1), (2, -1, 4))
    self.priors = tf.reshape(self.mbox_priorbox, shape=(1, 2, -1, 4), name='mbox_priorbox_final')
    assert self.priors_array.shape[1] == total_priors

    self.predictions = dict(locs=self.mbox_loc, confs=self.mbox_conf, logits=self.logits)
    return self.predictions

  def _iou(self, box):
    """Compute intersection over union for the box with all priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
    # compute intersection
    inter_upleft = np.maximum(self.priors_array[0, :, :2], box[:2])
    inter_botright = np.minimum(self.priors_array[0, :, 2:], box[2:])
    inter_wh = np.maximum(inter_botright - inter_upleft, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (self.priors_array[0, :, 2] - self.priors_array[0, :, 0])
    area_gt *= (self.priors_array[0, :, 3] - self.priors_array[0, :, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    return iou

  def _encode_box(self, box, return_iou=True):
    """Encode box for training, do it only for assigned priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
    iou = self._iou(box)
    encoded_box = np.zeros((self.priors_array.shape[1], 4 + return_iou), np.float32)
    assign_mask = iou > self.overlap_threshold
    if not assign_mask.any():
      assign_mask[iou.argmax()] = True
    if return_iou:
      encoded_box[:, -1][assign_mask] = iou[assign_mask]
    assigned_priors = self.priors_array[:, assign_mask, :]
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]
    assigned_priors_center = 0.5 * (assigned_priors[0, :, :2] + assigned_priors[0, :, 2:])
    assigned_priors_wh = (assigned_priors[0, :, 2:] - assigned_priors[0, :, :2])
    # we encode variance
    encoded_box[:, :2][assign_mask] = \
      (box_center - assigned_priors_center) / assigned_priors_wh / assigned_priors[1, :, :2]
    encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh) / assigned_priors[1, :, 2:]
    return encoded_box.ravel()

  def _assign_boxes(self, boxes):
    """Assign boxes to priors for training.

        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes), num_classes without background.

        # Return
            assignment: Tensor with assigned boxes, numpy tensor of shape (num_priors, 4 + num_classes).
        """

    target_shape = self.priors_array.shape[1], 4 + self.num_classes
    assignment = np.zeros(target_shape, dtype=np.float32)
    assignment[:, 4] = 1.0  # mark all as background
    # pylint: disable=len-as-condition
    if len(boxes) == 0:
      return assignment
    encoded_boxes = np.apply_along_axis(self._encode_box, 1, boxes[:, :4])
    encoded_boxes = encoded_boxes.reshape(-1, self.priors_array.shape[1], 5)
    best_iou = encoded_boxes[:, :, -1].max(axis=0)
    best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
    best_iou_mask = best_iou > 0
    best_iou_idx = best_iou_idx[best_iou_mask]
    assign_num = len(best_iou_idx)
    encoded_boxes = encoded_boxes[:, best_iou_mask, :]
    assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
    assignment[:, 4][best_iou_mask] = 0
    assignment[:, 5:][best_iou_mask] = boxes[best_iou_idx, 4:]
    return assignment

  def _compute_target(self, encoded_annotation):
    annotations = [pickle.loads(ea) for ea in encoded_annotation]
    assigns = []
    for annotation in annotations:
      one_hots, rects = [], []
      for label, boxes in annotation.items():
        one_hot = np.zeros((len(boxes), self.num_classes), dtype=np.float32)
        one_hot[:, label] = 1
        one_hots.extend(one_hot[:, 1:])
        rects.extend([[bb.xmin, bb.ymin, bb.xmax, bb.ymax] for bb in boxes])
      boxes_with_labels = np.hstack((np.asarray(rects), np.asarray(one_hots)))
      assigns.append(self._assign_boxes(boxes_with_labels))
    return np.array(assigns)

  def create_targets(self, annoation_tensor):
    assigns = tf.py_func(self._compute_target, [annoation_tensor], tf.float32, stateful=False, name='compute_target')
    assigns = tf.reshape(assigns, [-1, self.priors_array.shape[1], 4 + self.num_classes])
    return assigns

  def _decode_boxes(self, locs, priors, variance, scope='bboxes_decode'):
    with tf.variable_scope(scope):
      # priors (1, #priors, 4) = #priors * [xmin, ymin, xmax, ymax]
      prior_w = priors[:, 2] - priors[:, 0]
      prior_h = priors[:, 3] - priors[:, 1]
      prior_cx = 0.5 * (priors[:, 0] + priors[:, 2])
      prior_cy = 0.5 * (priors[:, 1] + priors[:, 3])

      # Compute center, height and width
      center_x = locs[:, :, 0] * prior_w * variance[:, 0] + prior_cx
      center_y = locs[:, :, 1] * prior_h * variance[:, 1] + prior_cy
      width = prior_w * tf.exp(locs[:, :, 2] * variance[:, 2])
      height = prior_h * tf.exp(locs[:, :, 3] * variance[:, 3])

      xmin = center_x - 0.5 * width
      ymin = center_y - 0.5 * height
      xmax = center_x + 0.5 * width
      ymax = center_y + 0.5 * height
      bbox = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
      if self.clip:
        bbox = tf.clip_by_value(bbox, 0., 1.)
      return bbox

  def detection_output(self, background_label=0, confidence_threshold=0.01, top_k=400, nms_threshold=0.45,
                       keep_top_k=200, scope='DetectionOutput', use_plain_caffe_format=True):
    assert background_label == 0  # only such mode is supported

    total_priors_count = self.priors.shape.as_list()[2]
    if top_k > total_priors_count:
      tf.logging.warning('detection_output::top_k value is greater than total priors count. '
                         'Set top_k to total_priors_count = {}'.format(total_priors_count))
      top_k = total_priors_count

    if keep_top_k > total_priors_count:
      tf.logging.warning('detection_output::keep_top_k value is greater than total priors count. '
                         'Set keep_top_k to total_priors_count = {}'.format(total_priors_count))
      keep_top_k = total_priors_count

    with tf.variable_scope(scope):
      bboxes = self._decode_boxes(self.mbox_loc, priors=self.priors[0, 0], variance=self.priors[0, 1])
      scores = self.mbox_conf

      batch_size = tf.shape(bboxes)[0]
      num_classes = self.num_classes

      def b_body(img_id, detection_batch):
        b_scores = scores[img_id]
        b_bboxes = bboxes[img_id]

        def c_body(class_id, detection_array):
          # Zeroing predictions below threshold
          with tf.variable_scope('bboxes_c_select', reuse=True):
            c_scores = b_scores[:, class_id]
            c_fmask = tf.cast(tf.greater(c_scores, confidence_threshold), scores.dtype)
            c_scores = c_scores * c_fmask
            c_bboxes = b_bboxes * tf.expand_dims(c_fmask, axis=-1)

          # Apply NMS
          with tf.variable_scope('bboxes_c_nms', reuse=True):
            c_indices = tf.image.non_max_suppression(c_bboxes, c_scores, top_k, nms_threshold)
            size = tf.size(c_indices)
            c_batch_ = tf.to_float(img_id) * tf.ones(shape=[top_k, 1], dtype=tf.float32)  # len(indices) x 1
            c_labels = tf.to_float(class_id) * tf.ones(shape=[top_k, 1], dtype=tf.float32)  # len(indices) x 1

            extra_size = top_k - size
            c_scores = tf.expand_dims(tf.gather(c_scores, c_indices), axis=-1)  # len(indices) x 1
            empty_c_scores = tf.zeros([extra_size, 1], dtype=tf.float32)
            c_scores = tf.concat([c_scores, empty_c_scores], axis=0)

            c_bboxes = tf.gather(c_bboxes, c_indices)  # len(indices) x 4
            empty_c_bboxes = tf.zeros([extra_size, 4], dtype=tf.float32)
            c_bboxes = tf.concat([c_bboxes, empty_c_bboxes], axis=0)
            c_predictions = tf.concat([c_batch_, c_labels, c_scores, c_bboxes], axis=1)  # top_k x 7
          return class_id + 1, detection_array.write(index=class_id - 1, value=c_predictions)

        # loop over num_classes
        class_id = 1  # c = 0 is a background, classes starts with index 1
        detection_img = tf.TensorArray(tf.float32, size=num_classes - 1)
        _, detection_img = tf.while_loop(lambda c, pa: tf.less(c, num_classes), c_body, [class_id, detection_img],
                                         back_prop=False, parallel_iterations=1)
        detection_img_flat = detection_img.concat()

        # Select topmost 'keep_top_k' predictions
        with tf.variable_scope('bboxes_keep_top_k', reuse=True):
          k = tf.minimum(keep_top_k, tf.shape(detection_img_flat)[0])
          _, indices = tf.nn.top_k(detection_img_flat[:, 2], k, sorted=True)
          detection_img_flat = tf.gather(detection_img_flat, indices)

        return img_id + 1, detection_batch.write(index=img_id, value=detection_img_flat)

      # loop over batch
      detection_batch = tf.TensorArray(tf.float32, size=batch_size)
      _, detection_batch = tf.while_loop(lambda img_id, ra: tf.less(img_id, batch_size),
                                         b_body, [0, detection_batch], back_prop=False)

      self.detections = detection_batch.concat() if use_plain_caffe_format else detection_batch.stack()
      self.detections = tf.reshape(self.detections, [-1, keep_top_k, self.detections.shape[-1]])
      return self.detections

  def get_config_for_tfmo(self, confidence_threshold=0.01, top_k=400, nms_threshold=0.45, keep_top_k=200):
    locs, confs, priors = self.mbox_loc.name[:-2], self.mbox_conf.name[:-2], self.mbox_priorbox.name[:-2]

    flatten_scopes = [f.name[:-2].replace('flatten/Reshape', '') for f in self.flattens_for_tfmo]

    json_lines = [
      '[',
      '    {',
      '        "custom_attributes": {',
      '            "code_type": "caffe.PriorBoxParameter.CENTER_SIZE",',
      # '            "num_classes": {},'.format(self.num_classes),   # Now TF MO estimates this automatically
      '            "confidence_threshold": {},'.format(confidence_threshold),
      '            "keep_top_k": {},'.format(keep_top_k),
      '            "nms_threshold": {},'.format(nms_threshold),
      '            "top_k": {},'.format(top_k),
      '            "pad_mode": "caffe.ResizeParameter.CONSTANT",',
      '            "resize_mode": "caffe.ResizeParameter.WARP"',
      '        },',
      '        "id": "SSDToolboxDetectionOutput",',
      '        "include_inputs_to_sub_graph": true,',
      '        "include_outputs_to_sub_graph": true,',
      '        "instances": {',
      '            "end_points": [',
      '                "{}",'.format(locs),
      '                "{}",'.format(confs),
      '                "{}"'.format(priors),
      '            ],',
      '            "start_points": [',
      '                "{}",'.format(locs),
      '                "{}",'.format(confs),
      '                "{}"'.format(priors),
      '            ]',
      '        },',
      '        "match_kind": "points"',
      '    }']
    json_nhwc_lines = [
      '    {',
      '        "custom_attributes": {},',
      '        "id": "ConvFlatten",',
      '        "inputs": [',
      '            [',
      '                {',
      '                    "node": "flatten/Reshape$",',
      '                    "port": 0',
      '                },',
      '                {',
      '                    "node": "flatten/Shape$",',
      '                    "port": 0',
      '                }',
      '            ],',
      '            [',
      '                {',
      '                    "node": "flatten/Shape$",',
      '                    "port": 0',
      '                },',
      '                {',
      '                    "node": "flatten/Reshape$",',
      '                    "port": 0',
      '                }',
      '            ]',
      '        ],',
      '        "instances": [',
      ',\n'.join(['            "{}"'.format(s) for s in flatten_scopes]),
      '        ],',
      '        "match_kind": "scope",',
      '        "outputs": [',
      '            {',
      '                "node": "flatten/Reshape$",',
      '                "port": 0',
      '            }',
      '        ]',
      '    }',
      ']',
    ]

    json_config = '\n'.join(json_lines)
    if self.data_format == 'NHWC':
      json_config += ',\n' + '\n'.join(json_nhwc_lines)

    class_name = self.__class__.__name__
    outputs = [locs, confs, priors] if self.detections is None else [self.detections.name[:-2]]

    return dict(json=json_config, cut_points=[locs, confs, priors], output_nodes=outputs, class_name=class_name)
