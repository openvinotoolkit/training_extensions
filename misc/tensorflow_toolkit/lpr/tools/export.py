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
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_io

from lpr.trainer import inference
from tfutils.helpers import load_module, execute_mo


def parse_args():
  parser = argparse.ArgumentParser(description='Export model in IE format')
  parser.add_argument('--data_type', default='FP32', choices=['FP32', 'FP16'], help='Data type of IR')
  parser.add_argument('--output_dir', default=None, help='Output Directory')
  parser.add_argument('--checkpoint', default=None, help='Default: latest')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


def freezing_graph(config, checkpoint, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  shape = (None,) + tuple(config.input_shape) # NHWC, dynamic batch
  graph = tf.Graph()
  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      input_tensor = tf.placeholder(dtype=tf.float32, shape=shape, name='input')
      prob = inference(config.rnn_cells_num, input_tensor, config.num_classes)
      prob = tf.transpose(prob, (1, 0, 2))
      data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])
      result = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)
      predictions = tf.to_int32(result[0][0])
      tf.sparse_to_dense(predictions.indices, [tf.shape(input_tensor, out_type=tf.int64)[0], config.max_lp_length],
                         predictions.values, default_value=-1, name='d_predictions')
      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  sess = tf.Session(graph=graph)
  sess.run(init)
  saver.restore(sess, checkpoint)
  frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["d_predictions"])
  tf.train.write_graph(sess.graph, output_dir, 'graph.pbtxt', as_text=True)
  path_to_frozen_model = graph_io.write_graph(frozen, output_dir, 'graph.pb.frozen', as_text=False)
  return path_to_frozen_model

def main(_):
  args = parse_args()
  config = load_module(args.path_to_config)

  checkpoint = args.checkpoint if args.checkpoint else tf.train.latest_checkpoint(config.model_dir)
  print(checkpoint)
  if not checkpoint or not os.path.isfile(checkpoint+'.index'):
    raise FileNotFoundError(str(checkpoint))

  step = checkpoint.split('.')[-2].split('-')[-1]
  output_dir = args.output_dir if args.output_dir else os.path.join(config.model_dir, 'export_{}'.format(step))

  # Freezing graph
  frozen_dir = os.path.join(output_dir, 'frozen_graph')
  frozen_graph = freezing_graph(config, checkpoint, frozen_dir)

  # Export to IR
  export_dir = os.path.join(output_dir, 'IR', args.data_type)

  mo_params = {
    'framework': 'tf',
    'model_name': 'lpr',
    'input': 'input',
    'output': 'd_predictions',
    'reverse_input_channels': True,
    'scale': 255,
    'input_shape': [1] + list(config.input_shape),
    'data_type': args.data_type,
  }

  execute_mo(mo_params, frozen_graph, export_dir)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
