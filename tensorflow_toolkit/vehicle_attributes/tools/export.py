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

import os
import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim

from vehicle_attributes.networks.resnet_10_bn import resnet_v1_10, resnet_arg_scope
from tfutils.helpers import load_module, execute_mo

def parse_args():
  parser = argparse.ArgumentParser(description='Export vehicle attributes model in IE format')
  parser.add_argument('--mo', default='mo.py', help="Path to model optimizer 'mo.py' script")
  parser.add_argument('--mo_config', default='cars_100/mo.yaml', help="Path config for model optimizer")
  parser.add_argument('--data_type', default='FP32', choices=['FP32', 'FP16'], help='Data type of IR')
  parser.add_argument('--output_dir', default=None, help='Output Directory')
  parser.add_argument('--checkpoint', default=None, help='Default: latest')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()

def freezing_graph(config, checkpoint, output_dir):
  with tf.Session() as sess:
    shape = [None] + list(config.input_shape)
    inputs = tf.placeholder(dtype=tf.float32, shape=shape, name="input")
    with slim.arg_scope(resnet_arg_scope()):
      _ = resnet_v1_10(inputs, is_training=False)
    saver = tf.train.Saver()

    saver.restore(sess, os.path.abspath(checkpoint))

    frozen = tf.graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          ['resnet_v1_10/type', 'resnet_v1_10/color'])

    tf.train.write_graph(frozen, output_dir, 'graph.pb_txt', as_text=True)
    frozen_path = tf.train.write_graph(frozen, output_dir, 'graph.pb.frozen', as_text=False)
    return frozen_path

def main(_):
  args = parse_args()
  config = load_module(args.path_to_config)

  checkpoint = args.checkpoint if args.checkpoint else tf.train.latest_checkpoint(config.model_dir)
  if not checkpoint or not os.path.isfile(checkpoint+'.index'):
    raise FileNotFoundError(str(checkpoint))

  step = checkpoint.split('.')[-1].split('-')[-1]
  output_dir = args.output_dir if args.output_dir else os.path.join(config.model_dir, 'export_{}'.format(step))

  # Freezing graph
  frozen_dir = os.path.join(output_dir, 'frozen_graph')
  frozen_graph = freezing_graph(config, checkpoint, frozen_dir)

  # Export to IR
  export_dir = os.path.join(output_dir, 'IR', args.data_type)

  mo_params = {
    'model_name': 'vehicle_attributes',
    'scale': 255,
    'input_shape': [1] + list(config.input_shape),
    'data_type': args.data_type,
  }
  execute_mo(mo_params, frozen_graph, export_dir)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
