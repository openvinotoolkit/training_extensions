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

import time
from functools import partial

import cv2
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from action_detection.nn.data.dataset import get_detection_dataset
from action_detection.nn.models import ActionNet
from action_detection.nn.monitors.base_monitor import BaseMonitor
from action_detection.postprocessing.detection_output import action_detection_output, action_warp_gt
from action_detection.postprocessing.quality import calc_map_mr, calc_action_accuracy


class ActionMonitor(BaseMonitor):
    """Class to control the behaviour of action network.
    """

    def _create_model(self, network_input, global_step=None, is_training=None,
                      merge_train_ops=False, merging_transit=False):
        if len(network_input) == 1:
            images, labels, annotation = network_input[0], None, None
        else:
            images, labels, annotation = network_input

        fn_activation = partial(tf.nn.leaky_relu, alpha=0.1)

        return ActionNet(self._params.backbone, images, labels, annotation, fn_activation,
                         is_training, self._params.head_params, merge_train_ops, merging_transit,
                         self._params.lr_params, self._params.mbox_params, self._params.action_params,
                         global_step=global_step, name=self._params.name,
                         use_nesterov=self._params.use_nesterov, norm_kernels=self._params.norm_kernels)

    def _transfer_train_parameters(self, sess, model):
        trg_scope = '{}/{}'.format(self._params.name, self._params.backbone)

        model.load_available_variables(self._init_model_path, sess, self._src_scope,
                                       '{}/shared'.format(trg_scope))
        model.load_available_variables(self._init_model_path, sess, self._src_scope,
                                       '{}/detection'.format(trg_scope))
        model.load_available_variables(self._init_model_path, sess, self._src_scope,
                                       '{}/classification'.format(trg_scope))

    def _get_extra_param_names(self):
        return ['moving_center_{}'.format(i) for i in xrange(self._params.num_actions)]

    def _create_train_dataset(self, data_path):
        return get_detection_dataset(data_path, self._params.image_size, self._batch_size, True, 'train_data',
                                     self._train_prefetch_size, self._train_data_process_num_threads,
                                     tuple_process_fn=self._params.tuple_augmentation,
                                     image_process_fn=self._params.image_augmentation,
                                     max_num_objects_per_image=self._params.max_num_objects_per_image,
                                     labels_map=self._params.labels_map,
                                     ignore_classes=self._params.ignore_classes,
                                     use_class_balancing=self._params.use_class_balancing)

    def _create_test_dataset(self, data_path):
        return get_detection_dataset(data_path, self._params.image_size, self._batch_size, False, 'val_data',
                                     self._test_prefetch_size, self._test_data_process_num_threads,
                                     max_num_objects_per_image=self._params.max_num_objects_per_image,
                                     use_difficult=True, labels_map=self._params.labels_map)

    def _get_test_ops(self, model, network_inputs):
        return model.predictions

    def _test(self, sess, network_inputs, test_ops):
        def _normalize_confusion_matrix(input_cm):
            """Normalizes by row the input confusion matrix.

            :param input_cm: Input confusion matrix
            :return: Normalized confusion matrix
            """

            assert len(input_cm.shape) == 2
            assert input_cm.shape[0] == input_cm.shape[1]

            row_sums = np.maximum(1, np.sum(input_cm, axis=1, keepdims=True)).astype(np.float32)
            norm_cm = input_cm.astype(np.float32) / row_sums

            return norm_cm

        def _print_confusion_matrix(input_cm, name, classes):
            """Prints values of the confusion matrix.

            :param input_cm: Input confusion matrix
            :param name: Header of the output
            :param classes: List of class names
            """

            assert len(input_cm.shape) == 2
            assert input_cm.shape[0] == input_cm.shape[1]
            assert len(classes) == input_cm.shape[0]

            max_class_name_length = max([len(cl) for cl in classes])

            norm_cm = _normalize_confusion_matrix(input_cm)

            self._log('{} CM:'.format(name))
            for i, class_name in enumerate(classes):
                values = ''
                for j in range(len(classes)):
                    values += '{:7.2f} |'.format(norm_cm[i, j] * 100.)
                self._log('   {0: <{1}}|{2}'.format(class_name, max_class_name_length + 1, values))

        def _calculate_accuracy(input_cm):
            """Calculates the mean of diagonal elements of normalized confusion matrix.

            :param input_cm: Input confusion matrix
            :return: Accuracy value
            """

            assert len(input_cm.shape) == 2
            assert input_cm.shape[0] == input_cm.shape[1]

            base_acc = float(np.sum(input_cm.diagonal())) / float(np.maximum(1, np.sum(input_cm)))

            norm_cm = _normalize_confusion_matrix(input_cm)
            norm_acc = np.mean(norm_cm.diagonal())

            return base_acc, norm_acc

        all_detections = []
        all_gt = []

        for _ in trange(self._params.val_steps, desc='Dumping predictions'):
            gt_labels, gt_annot, pr_loc, pr_det_conf, pr_action_conf = sess.run(network_inputs[1:] + test_ops)

            batch_detections = action_detection_output(pr_loc, pr_det_conf, pr_action_conf, self._params.bg_class,
                                                       min_det_conf=self._params.det_conf,
                                                       min_action_conf=self._params.action_conf)
            batch_gt = action_warp_gt(gt_annot, gt_labels, self._params.bg_class)

            all_detections.extend(batch_detections)
            all_gt.extend(batch_gt)

        sess.close()

        action_cm = calc_action_accuracy(all_detections, all_gt, self._params.bg_class, self._params.num_actions)
        base_accuracy, norm_accuracy = _calculate_accuracy(action_cm)
        self._log('Task Action Accuracy. Base: {:.2f} Norm: {:.2f}'.format(base_accuracy * 100., norm_accuracy * 100.))
        _print_confusion_matrix(action_cm, 'Task', self._params.valid_actions)

        det_metrics_by_class = calc_map_mr(all_detections, all_gt, return_all=True)
        ap_value, mr_value = det_metrics_by_class[(self._params.bg_class + 1) % 2]
        self._log('Class-agnostic AP: {:.3f}%   mr@0.1: {:.3f}%'.format(1e2 * ap_value, 1e2 * mr_value))

    def _get_out_node_names(self):
        det_name_templates = ['{}/out_detection_loc', '{}/out_detection_conf']
        out_det_node_names = [t.format(self._params.name) for t in det_name_templates]

        action_name_template = '{}/action_heads/out_head_{}_anchor_{}'
        out_action_node_names = []
        for head_id, head_desc in enumerate(self._params.head_params):
            for anchor_id in xrange(head_desc.anchors.shape[0]):
                out_action_node_names.append(action_name_template.format(self._params.name, head_id + 1, anchor_id + 1))

        return out_det_node_names + out_action_node_names

    def _demo(self, sess, data_path, input_image, inference_ops, out_scale):
        def _draw_actions(image, detections, bg_label, min_det_score, min_action_score):
            """Draws detected persons with action captions.

            :param image: Input image
            :param detections: Detected boxes
            :param bg_label: Label of background class
            :param min_det_score: Min detection score to show box
            :param min_action_score: Min action score to show box
            """

            image = np.copy(image)
            image_height, image_width = image.shape[:2]

            det_label = (bg_label + 1) % 2
            detections = detections[det_label]
            if len(detections) == 0:
                return image

            for i in xrange(detections.loc.shape[0]):
                if detections.scores[i] < min_det_score:
                    continue

                ymin = int(detections.loc[i, 0] * image_height)
                xmin = int(detections.loc[i, 1] * image_width)
                ymax = int(detections.loc[i, 2] * image_height)
                xmax = int(detections.loc[i, 3] * image_width)

                action_score = detections.action_scores[i]
                action_color = self._params.action_colors_map[detections.action_labels[i]] \
                    if action_score > min_action_score else self._params.undefined_action_color
                action_name = self._params.action_names_map[detections.action_labels[i]] \
                    if action_score > min_action_score else self._params.undefined_action_name

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), action_color, 1)

                caption = '{}: {:.1f} {:.1f}'.format(action_name, 1e2 * action_score, 1e2 * detections.scores[i])
                cv2.putText(image, caption, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, action_color, 1)

                head_x = int(0.5 * (xmin + xmax))
                head_y = int(0.85 * ymin + 0.15 * ymax)
                cv2.circle(image, (head_x, head_y), 4, (0, 0, 255), -1)

            return image

        vidcap = cv2.VideoCapture(data_path)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=video_length, desc='Read video')
        success = True
        while success:
            success, frame = vidcap.read()
            pbar.update(1)

            if success:
                float_image = cv2.resize(
                    frame, (self._params.image_size.w, self._params.image_size.h)).astype(np.float32)
                pr_loc, pr_det_conf, pr_action_conf = sess.run(inference_ops, feed_dict={input_image: float_image})

                batch_detections = action_detection_output(pr_loc, pr_det_conf, pr_action_conf, self._params.bg_class,
                                                           min_det_conf=self._params.det_conf,
                                                           min_action_conf=self._params.action_conf)

                if out_scale != 1.0:
                    out_height = int(frame.shape[0] * out_scale)
                    out_width = int(frame.shape[1] * out_scale)
                    frame = cv2.resize(frame, (out_width, out_height))

                annotated_frame = _draw_actions(frame, batch_detections[0], self._params.bg_class,
                                                min_det_score=self._params.det_conf,
                                                min_action_score=self._params.action_conf)

                cv2.imshow('Demo', annotated_frame)

                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == ord('p'):
                    while True:
                        new_key = cv2.waitKey(1)
                        if new_key == ord('p') or new_key == 27:
                            break
                        else:
                            time.sleep(0.1)

        pbar.close()
        vidcap.release()
        cv2.destroyAllWindows()
