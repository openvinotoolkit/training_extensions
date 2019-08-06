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

import tensorflow as tf


def create_variable(name, shape, initializer, dtype=tf.float32, device=None, collections=None, trainable=True):
    """Wrapper for the parameterized variable creation.

    :param name: Name of variable
    :param shape: Shape of variable
    :param initializer: Initializer function if needed
    :param dtype: Type of data
    :param device: Target device if needed
    :param collections: Target collection to add in
    :param trainable: Whether to add variable set of trainable variables
    :return: Specified variable
    """

    if device is None:
        if not callable(initializer):
            return tf.get_variable(
                name, initializer=initializer, dtype=dtype, collections=collections, trainable=trainable)
        else:
            return tf.get_variable(
                name, shape, initializer=initializer, dtype=dtype, collections=collections, trainable=trainable)
    else:
        with tf.device(device):
            if not callable(initializer):
                return tf.get_variable(
                    name, initializer=initializer, dtype=dtype, collections=collections, trainable=trainable)
            else:
                return tf.get_variable(
                    name, shape, initializer=initializer, dtype=dtype, collections=collections, trainable=trainable)
