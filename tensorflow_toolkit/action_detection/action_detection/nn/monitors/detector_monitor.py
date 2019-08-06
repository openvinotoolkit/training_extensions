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

from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import trange

from action_detection.nn.data.dataset import get_detection_dataset
from action_detection.nn.models import SSD
from action_detection.nn.monitors.base_monitor import BaseMonitor
from action_detection.postprocessing.detection_output import ssd_detection_output, ssd_warp_gt
from action_detection.postprocessing.quality import calc_map_mr


class DetectorMonitor(BaseMonitor):
    """Class to control the behaviour of detection network.
    """

    def _create_model(self, network_input, global_step=None, is_training=None,
                      merge_train_ops=False, merging_transit=False):
        if len(network_input) == 1:
            images, labels, annotation = network_input[0], None, None
        else:
            images, labels, annotation = network_input

        fn_activation = partial(tf.nn.leaky_relu, alpha=0.1)

        return SSD(self._params.backbone, images, labels, annotation, fn_activation, is_training,
                   self._params.head_params, merge_train_ops, merging_transit,
                   self._params.lr_params, self._params.mbox_params,
                   global_step=global_step, name=self._params.name,
                   use_nesterov=self._params.use_nesterov, norm_kernels=self._params.norm_kernels)

    def _create_train_dataset(self, data_path):
        return get_detection_dataset(data_path, self._params.image_size, self._batch_size, True, 'train_data',
                                     self._train_prefetch_size, self._train_data_process_num_threads,
                                     tuple_process_fn=self._params.tuple_augmentation,
                                     image_process_fn=self._params.image_augmentation,
                                     max_num_objects_per_image=self._params.max_num_objects_per_image,
                                     labels_map=self._params.labels_map)

    def _create_test_dataset(self, data_path):
        return get_detection_dataset(data_path, self._params.image_size, self._batch_size, False, 'val_data',
                                     self._test_prefetch_size, self._test_data_process_num_threads,
                                     max_num_objects_per_image=self._params.max_num_objects_per_image,
                                     use_difficult=False,
                                     labels_map=self._params.labels_map)

    def _get_test_ops(self, model, network_inputs):
        return model.predictions

    def _test(self, sess, network_inputs, test_ops):
        self._log('\nValidation...')
        all_detections = []
        all_gt = []
        for _ in trange(self._params.val_steps, desc='Dumping predictions'):
            gt_labels, gt_annot, pr_loc, pr_conf = sess.run(network_inputs[1:] + test_ops)

            batch_detections = ssd_detection_output(pr_loc, pr_conf, self._params.bg_class, min_conf=0.1)
            batch_gt = ssd_warp_gt(gt_annot, gt_labels, self._params.bg_class)

            all_detections.extend(batch_detections)
            all_gt.extend(batch_gt)

        metrics_by_class = calc_map_mr(all_detections, all_gt, return_all=True)

        ap_values = []

        self._log('\nMetrics by class:')
        for class_id in xrange(self._params.num_classes):
            if class_id == self._params.bg_class:
                continue

            if class_id in metrics_by_class:
                ap_value, mr_value = metrics_by_class[class_id]
            else:
                ap_value, mr_value = 0.0, 1.0

            ap_values.append(ap_value)

            self._log('   {:>3}: AP: {:.3f}%   mr@0.1: {:.3f}%'.format(class_id, 1e2 * ap_value, 1e2 * mr_value))

        map_metric = np.mean(ap_values) if len(ap_values) > 0 else 0.0
        self._log('Total. mAP: {:.3f}%'.format(1e2 * map_metric))

    def _get_out_node_names(self):
        name_templates = ['{}/out_detection_loc', '{}/out_detection_logits', '{}/out_detection_priors']
        return [t.format(self._params.name) for t in name_templates]

    def _demo(self, sess, data_path, input_image, inference_ops, out_scale):
        raise NotImplementedError()
