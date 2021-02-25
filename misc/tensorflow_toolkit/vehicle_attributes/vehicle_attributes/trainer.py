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

import os
import random
import numpy as np

import tensorflow as tf

import tensorflow.contrib.slim as slim

from vehicle_attributes.readers.vehicle_attributes_json import BarrierAttributesJson
from vehicle_attributes.networks.resnet_10_bn import resnet_v1_10, resnet_arg_scope
from vehicle_attributes.utils import get_checkpoint_variable_names

def set_initial_weights(init_ckpt=None):
  if init_ckpt:
    variables = slim.get_variables_to_restore()
    varnames = get_checkpoint_variable_names(init_ckpt)
    vars_to_restore = {v.name[:-2]: v for v in variables if v.name[:-2] in varnames}

    tf.train.init_from_checkpoint(init_ckpt, vars_to_restore)

def create_session(config, type):
  if type == 'train':
    random_seed = config.train.random_seed
  else:
    random_seed = 666

  np.random.seed(random_seed)
  tf.set_random_seed(random_seed)
  random.seed(random_seed)

  config_type = getattr(config, type).execution

  if hasattr(config_type, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config_type.CUDA_VISIBLE_DEVICES

  intra_op_parallelism_threads = config_type.intra_op_parallelism_threads if \
    hasattr(config_type, 'intra_op_parallelism_threads') else 0
  inter_op_parallelism_threads = config_type.inter_op_parallelism_threads if \
    hasattr(config_type, 'inter_op_parallelism_threads') else 0
  session_config = tf.ConfigProto(allow_soft_placement=True,
                                  intra_op_parallelism_threads=intra_op_parallelism_threads,
                                  inter_op_parallelism_threads=inter_op_parallelism_threads)
  if hasattr(config_type, 'per_process_gpu_memory_fraction'):
    session_config.gpu_options.per_process_gpu_memory_fraction = config_type.per_process_gpu_memory_fraction
  if hasattr(config_type, 'allow_growth'):
    session_config.gpu_options.allow_growth = config_type.allow_growth

  return session_config

# pylint: disable=too-many-locals
def resnet_v1_10_1(features,
                   labels,
                   mode,
                   params):

  learning_rate = params.get('learning_rate', 0.001)
  optimizer_func = params.get('optimizer', lambda learning_rate: tf.train.AdagradOptimizer(learning_rate=learning_rate))
  use_pretrained_weights = params['use_pretrained_weights']
  init_ckpt = params['pretrained_ckpt']
  inputs = features
  is_training = bool(mode == tf.estimator.ModeKeys.TRAIN)
  with slim.arg_scope(resnet_arg_scope()):
    backbone = resnet_v1_10(inputs,
                            is_training=is_training,
                            scope='resnet_v1_10')

  def type_to_one_hot(res_type):
    softmax = tf.nn.softmax(res_type)
    mask = tf.greater_equal(softmax, softmax[tf.argmax(softmax)])
    return tf.cast(mask, dtype=tf.int32)

  predictions = {
    "types": backbone[1]['predictions_types'],
    "color_lab": backbone[1]['predictions_color'],
    "types_class": tf.map_fn(type_to_one_hot, backbone[1]['predictions_types'], dtype=tf.int32)
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  assert mode in(tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL)

  loss_color = tf.losses.mean_squared_error(labels=labels[:, 4:7], predictions=predictions['color_lab'])/2.
  loss_type = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels[:, 0:4], 1),
                                                                            logits=predictions['types']))
  loss = tf.cond(tf.train.get_global_step() < 100000,
                 lambda: loss_color + 0.00001 *loss_type,
                 lambda: loss_color + 0.1 * loss_type)

  if mode == tf.estimator.ModeKeys.TRAIN:
    if use_pretrained_weights:
      set_initial_weights(init_ckpt)

    optimizer = optimizer_func(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = slim.learning.create_train_op(total_loss=loss,
                                               optimizer=optimizer,
                                               global_step=tf.train.get_global_step())
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss_color', loss_color)
    tf.summary.scalar('loss_type', loss_type)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
    "Mean absolute error of l color component": tf.metrics.mean_absolute_error(labels=labels[:, 4]*255, \
      predictions=predictions["color_lab"][:, 0]*255),
    "Mean absolute error of a color component": tf.metrics.mean_absolute_error(labels=labels[:, 5]*255, \
      predictions=predictions["color_lab"][:, 1]*255),
    "Mean absolute error of b color component": tf.metrics.mean_absolute_error(labels=labels[:, 6]*255, \
      predictions=predictions["color_lab"][:, 2]*255),
    "Color mean absolute error": tf.metrics.mean_absolute_error(labels=labels[:, 4:7]*255,
                                                                predictions=predictions["color_lab"]*255),
    "Type accuracy - average": tf.metrics.accuracy(labels=labels[:, 0:4], predictions=predictions['types_class']),
    "Type accuracy - car": tf.metrics.accuracy(labels=labels[:, 0], predictions=predictions['types_class'][:, 0]),
    "Type accuracy - bus": tf.metrics.accuracy(labels=labels[:, 1], predictions=predictions['types_class'][:, 1]),
    "type accuracy - truck": tf.metrics.accuracy(labels=labels[:, 2], predictions=predictions['types_class'][:, 2]),
    "Type accuracy - van": tf.metrics.accuracy(labels=labels[:, 3], predictions=predictions['types_class'][:, 3])
  }

  return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class InputTrainData:
  # pylint: disable=too-many-arguments
  def __init__(self, batch_size, input_shape, json_path, cache_type='NONE',
               num_parallel_calls=4, prefetch_size=4):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.json_path = json_path
    self.cache_type = cache_type
    self.num_parallel_calls = num_parallel_calls
    self.prefetch_size = prefetch_size

    dataset_size = BarrierAttributesJson.init_cache(json_path, cache_type)
    self.dataset_size = dataset_size

  # pylint: disable=unnecessary-lambda
  def input_fn(self):
    train_dataset = BarrierAttributesJson.create_dataset(self.dataset_size)

    transform_fn = lambda value: BarrierAttributesJson.transform_fn(value)

    map_fn = lambda value: tf.py_func(transform_fn, [value], (tf.float32, tf.float32))

    dataset = train_dataset.shuffle(buffer_size=self.dataset_size, reshuffle_each_iteration=True)
    dataset = dataset.repeat().map(map_fn, num_parallel_calls=self.num_parallel_calls)
    dataset = dataset.batch(self.batch_size).prefetch(self.prefetch_size)

    images, annotation = dataset.make_one_shot_iterator().get_next()

    images.set_shape([None] + list(self.input_shape))
    annotation.set_shape([None, 7])

    return images, annotation

class InputEvalData:
  # pylint: disable=too-many-arguments
  def __init__(self, batch_size, input_shape, json_path, cache_type='NONE',
               num_parallel_calls=4, prefetch_size=4):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.json_path = json_path
    self.cache_type = cache_type
    self.num_parallel_calls = num_parallel_calls
    self.prefetch_size = prefetch_size

    dataset_size = BarrierAttributesJson.init_cache(json_path, cache_type)
    self.dataset_size = dataset_size

  # pylint: disable=unnecessary-lambda
  def input_fn(self):
    infer_dataset = BarrierAttributesJson.create_dataset(self.dataset_size)

    transform_fn = lambda value: BarrierAttributesJson.transform_fn(value)

    map_fn = lambda value: tf.py_func(transform_fn, [value], (tf.float32, tf.float32))

    dataset = infer_dataset.map(map_fn, num_parallel_calls=self.num_parallel_calls)
    dataset = dataset.batch(self.batch_size).prefetch(self.prefetch_size)

    images, annotation = dataset.make_one_shot_iterator().get_next()

    images.set_shape([None] + list(self.input_shape))
    annotation.set_shape([None, 7])

    return images, annotation
