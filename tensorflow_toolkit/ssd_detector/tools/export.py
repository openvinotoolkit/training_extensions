#!/usr/bin/env python3
#
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
import argparse
import os

import tensorflow as tf
from tfutils.helpers import dump_frozen_graph, load_module, execute_mo
from ssd_detector.networks.mobilenet_ssd import MobileNetSSD


def parse_args():
  parser = argparse.ArgumentParser(description='Export model in IE format')
  parser.add_argument('--model_name', default='vlp')
  parser.add_argument('--data_type', default='FP32', choices=['FP32', 'FP16'], help='Data type of IR')
  parser.add_argument('--output_dir', default=None, help='Output Directory')
  parser.add_argument('--checkpoint', default=None, help='Default: latest')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


def freezing_graph(config, checkpoint, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  detector_params = config.detector_params.copy()
  with tf.Session() as sess:
    input_tensor = tf.placeholder(dtype=tf.float32, shape=(None,) + tuple(config.input_shape))

    for unnecessary_param in ['initial_weights_path',
                              'learning_rate',
                              'optimizer',
                              'weights_decay_factor',
                              'collect_priors_summary']:
      if unnecessary_param in detector_params:
        del detector_params[unnecessary_param]

    ssd = MobileNetSSD(input_tensor=input_tensor, is_training=False, **detector_params)
    ssd.detection_output()
    # For eval.py
    tf.get_variable('eval_iteration', initializer=0, dtype=tf.int32, trainable=False)
    tf.get_variable('global_step', initializer=tf.constant_initializer(0, dtype=tf.int64), shape=(), dtype=tf.int64,
                    trainable=False)

    train_param, _ = ssd.create_transform_parameters(width=config.input_shape[0], height=config.input_shape[1])

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    mean_values = [train_param.mean_value for _ in range(3)]
    print(mean_values)
    print(train_param.scale)
    print(1./train_param.scale)

    ssd_config = ssd.get_config_for_tfmo()
    graph_file = os.path.join(output_dir, 'graph.pb')
    frozen = dump_frozen_graph(sess, graph_file, ssd_config['output_nodes'])

    # Generate custom_operations_config for mo
    ssd_config_path = frozen.replace('.pb.frozen', '.tfmo.json')
    with open(ssd_config_path, mode='w') as file:
      file.write(ssd_config['json'])

    return frozen, ssd_config_path, train_param, ssd_config


def main(_):
  args = parse_args()
  config = load_module(args.path_to_config)

  checkpoint = args.checkpoint if args.checkpoint else tf.train.latest_checkpoint(config.MODEL_DIR)
  print(checkpoint)
  if not checkpoint or not os.path.isfile(checkpoint+'.index'):
    raise FileNotFoundError(str(checkpoint))

  step = checkpoint.split('-')[-1]
  output_dir = args.output_dir if args.output_dir else os.path.join(config.MODEL_DIR, 'export_{}'.format(step))

  # Freezing graph
  frozen_dir = os.path.join(output_dir, 'frozen_graph')
  frozen_graph, ssd_config_path, train_param, ssd_config = freezing_graph(config, checkpoint, frozen_dir)

  # Export to IR
  export_dir = os.path.join(output_dir, 'IR', args.data_type)
  input_shape = [1] + list(config.input_shape) # Add batch size 1 in shape

  scale = 1./train_param.scale
  mean_value = [train_param.mean_value for _ in range(3)]
  mo_params = {
    'model_name': args.model_name,
    'output': ','.join(ssd_config['cut_points']),
    'input_shape': input_shape,
    'scale': scale,
    'mean_value': mean_value,
    'tensorflow_use_custom_operations_config': ssd_config_path,
    'data_type':args.data_type,
  }
  execute_mo(mo_params, frozen_graph, export_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
