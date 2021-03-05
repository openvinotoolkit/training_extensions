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
from nets import resnet_utils
from nets.resnet_v1 import NoOpScope
from nets.resnet_utils import resnet_arg_scope  # pylint: disable=unused-import

# pylint: disable=too-many-locals, too-many-arguments, unused-argument, invalid-name
@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):

  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:

    residual = slim.conv2d(inputs, depth_bottleneck, [3, 3], stride=stride,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params={'decay': 0.99, 'zero_debias_moving_mean': True},
                           activation_fn=tf.nn.relu,
                           scope='conv1')
    residual = slim.conv2d(residual, depth_bottleneck, 3, stride=1,
                           normalizer_fn=None,
                           activation_fn=None,
                           rate=rate,
                           biases_initializer=None,
                           scope='conv2')
    if stride == 1:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
        inputs,
        depth_bottleneck, [1, 1],
        stride=stride,
        normalizer_fn=None,
        activation_fn=tf.nn.relu6 if use_bounded_activations else None,
        biases_initializer=None,
        scope='shortcut')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      if tf.get_default_graph().get_name_scope().find('block3') < 0:
        output = slim.batch_norm(shortcut + residual, activation_fn=tf.nn.relu)
      else:
        output = shortcut + residual
        output = slim.utils.collect_named_outputs(outputs_collections,
                                                  sc.name,
                                                  output)
    return output

# pylint: disable=unused-argument
def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):

  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with (slim.arg_scope([slim.batch_norm], decay=0.99, zero_debias_moving_mean=True, is_training=is_training)
            if is_training is not None else NoOpScope()):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = slim.batch_norm(net, decay=0.99, zero_debias_moving_mean=True, scale=True)
          net = slim.conv2d(net, 64, 7, stride=2, padding='SAME',
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'decay': 0.99, 'zero_debias_moving_mean': True},
                            activation_fn=tf.nn.relu,
                            scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
          end_points_collection)

        num_types = 4
        num_color = 3
        types = slim.conv2d(net, num_types, [1, 1], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                            normalizer_fn=None, scope='logits_type')
        end_points[sc.name + '/logits_type'] = net

        if global_pool:
          types = tf.reduce_mean(types, [1, 2], name='pool5', keepdims=True)
          end_points['global_pool_types'] = types

        if spatial_squeeze:
          types = tf.squeeze(types, [1, 2], name='type')
          end_points[sc.name + '/type'] = types
        end_points['predictions_types'] = types

        color = slim.conv2d(net, num_color, [1, 1], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                            normalizer_fn=None, scope='logits')
        end_points[sc.name + '/logits'] = color

        if global_pool:
          color = tf.reduce_mean(color, [1, 2], name='pool6', keepdims=True)
          end_points['global_pool_color'] = color

        if spatial_squeeze:
          color = tf.squeeze(color, [1, 2], name='color')
          end_points[sc.name + '/color'] = color
        end_points['predictions_color'] = color

        return net, end_points

def resnet_v1_block(scope, base_depth, num_units, stride):

  return resnet_utils.Block(scope, bottleneck, [{
    'depth': base_depth * 4,
    'depth_bottleneck': base_depth,
    'stride': 1
  }] * (num_units - 1) + [{
    'depth': base_depth * 4,
    'depth_bottleneck': base_depth,
    'stride': stride
  }])

def resnet_v1_10(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='resnet_v1_10'):
  blocks = [
    resnet_v1_block('block1', base_depth=64, num_units=1, stride=1),
    resnet_v1_block('block2', base_depth=128, num_units=1, stride=2),
    resnet_v1_block('block3', base_depth=128, num_units=1, stride=2)
  ]

  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
