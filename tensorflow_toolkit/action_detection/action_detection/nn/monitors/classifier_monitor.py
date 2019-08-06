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

from datetime import datetime
from functools import partial

import tensorflow as tf
from tqdm import trange

from action_detection.nn.data.dataset import get_classification_dataset
from action_detection.nn.data.preprocessing import ImageNetProcessFn
from action_detection.nn.models import ImageClassifier
from action_detection.nn.monitors.base_monitor import BaseMonitor


class ClassifierMonitor(BaseMonitor):
    """Class to control the behaviour of classification network.
    """

    def _create_model(self, network_input, global_step=None, is_training=None,
                      merge_train_ops=False, merging_transit=False):
        if len(network_input) == 1:
            images, labels = network_input[0], None
        else:
            images, labels = network_input

        fn_activation = partial(tf.nn.leaky_relu, alpha=0.1)

        return ImageClassifier(self._params.backbone, images, labels, fn_activation, is_training,
                               self._params.num_classes, merge_train_ops, merging_transit, self._params.lr_params,
                               global_step=global_step, keep_probe=self._params.keep_probe, name=self._params.name,
                               use_nesterov=self._params.use_nesterov, norm_kernels=self._params.norm_kernels)

    def _create_train_dataset(self, data_path):
        return get_classification_dataset(data_path, self._params.num_classes, self._params.image_size,
                                          self._batch_size, True, 'train_data',
                                          self._train_prefetch_size, self._train_data_process_num_threads,
                                          process_fn=self._params.image_augmentation)

    def _create_test_dataset(self, data_path):
        image_processing = ImageNetProcessFn(self._params.val_central_fraction, self._params.image_size)

        return get_classification_dataset(data_path, self._params.num_classes, self._params.image_size,
                                          self._batch_size, False, 'val_data',
                                          self._test_prefetch_size, self._test_data_process_num_threads,
                                          process_fn=image_processing)

    def _get_test_ops(self, model, network_inputs):
        return [model.num_valid_op(network_inputs[1])]

    def _test(self, sess, network_inputs, test_ops):
        total_num_valid = 0
        for _ in trange(self._params.val_steps, desc='Validation'):
            num_valid = sess.run(test_ops)
            total_num_valid += num_valid

        val_accuracy = float(total_num_valid) / float(self._params.val_steps * self._batch_size)
        self._log('{}: Validation accuracy: {:.5f}%'.format(datetime.now(), 1e2 * val_accuracy))

    def _get_out_node_names(self):
        name_templates = ['{}/classifier_layer/output']
        return [t.format(self._params.name) for t in name_templates]

    def _demo(self, sess, data_path, input_image, inference_ops, out_scale):
        raise NotImplementedError()
