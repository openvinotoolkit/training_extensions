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
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def get_spatial_dims(tensor_or_shape, data_format='NHWC'):
  if isinstance(tensor_or_shape, (list, tuple)):
    input_shape = tensor_or_shape
  else:
    input_shape = tensor_or_shape.get_shape().as_list()

  assert data_format in ('NCHW', 'NHWC')
  assert len(input_shape) == 4

  if data_format == 'NHWC':
    height = input_shape[1]
    width = input_shape[2]
  else:
    height = input_shape[2]
    width = input_shape[3]
  return height, width


@slim.add_arg_scope
def channel_to_last(inputs, data_format='NHWC', scope=None):
  assert data_format in ('NCHW', 'NHWC')
  with tf.name_scope(scope, 'channel_to_last', [inputs]):
    return inputs if data_format == 'NHWC' else tf.transpose(inputs, perm=(0, 2, 3, 1))
