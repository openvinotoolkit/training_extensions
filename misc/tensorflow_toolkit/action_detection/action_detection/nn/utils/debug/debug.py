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

from __future__ import print_function

import tensorflow as tf


def print_tensor(in_tensor, header='', out_tensor=None):
    """Debug function to incorporate print tensor op into the computation graph.

    :param in_tensor: Tensor to print values
    :param header: Text header to print
    :param out_tensor: Tensor to attach in if needed
    :return: Same tensor or output tensor
    """

    prefix = '\n{}: '.format(header)
    separator = '\n'

    in_shape = tf.shape(in_tensor)

    if out_tensor is None:
        with tf.control_dependencies([tf.print(prefix, in_shape, separator, in_tensor)]):
            out = tf.identity(in_tensor)
    else:
        with tf.control_dependencies([tf.print(prefix, in_shape, separator, in_tensor)]):
            out = tf.identity(out_tensor)
    return out
