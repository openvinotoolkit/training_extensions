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
import random
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lpr.trainer import inference
from lpr.utils import accuracy, dataset_size
from tfutils.helpers import load_module


def parse_args():
  parser = argparse.ArgumentParser(description='Perform evaluation of a trained model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


def read_data(height, width, channels_num, list_file_name, batch_size=10):
  reader = tf.TextLineReader()
  _, value = reader.read(list_file_name)
  filename, label = tf.decode_csv(value, [[''], ['']], ' ')

  image_file = tf.read_file(filename)
  rgb_image = tf.image.decode_png(image_file, channels=channels_num)
  rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
  resized_image = tf.image.resize_images(rgb_image_float, [height, width])
  resized_image.set_shape([height, width, channels_num])

  image_batch, label_batch, file_batch = tf.train.batch([resized_image, label, image_file], batch_size=batch_size)
  return image_batch, label_batch, file_batch


def data_input(height, width, channels_num, filename, batch_size=1):
  files_string_producer = tf.train.string_input_producer([filename])
  image, label, filename = read_data(height, width, channels_num, files_string_producer, batch_size)
  return image, label, filename

# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def validate(config):
  if hasattr(config.eval, 'random_seed'):
    np.random.seed(config.eval.random_seed)
    tf.set_random_seed(config.eval.random_seed)
    random.seed(config.eval.random_seed)

  if hasattr(config.eval.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  height, width, channels_num = config.input_shape
  rnn_cells_num = config.rnn_cells_num

  graph = tf.Graph()
  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      inp_data, label_val, file_names = data_input(height, width, channels_num,
                                                   config.eval.file_list_path, batch_size=config.eval.batch_size)

      prob = inference(rnn_cells_num, inp_data, config.num_classes)
      prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC

      data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size

      result = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)

      predictions = tf.to_int32(result[0][0])
      d_predictions = tf.sparse_to_dense(predictions.indices,
                                         [tf.shape(inp_data, out_type=tf.int64)[0], config.max_lp_length],
                                         predictions.values, default_value=-1, name='d_predictions')

      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  # session
  conf = tf.ConfigProto()
  if hasattr(config.eval.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.eval.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  sess = tf.Session(graph=graph, config=conf)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  sess.run(init)


  checkpoints_dir = config.model_dir
  latest_checkpoint = None
  wait_iters = 0

  if not os.path.exists(os.path.join(checkpoints_dir, 'eval')):
    os.mkdir(os.path.join(checkpoints_dir, 'eval'))
  writer = tf.summary.FileWriter(os.path.join(checkpoints_dir, 'eval'), sess.graph)


  while True:
    if config.eval.checkpoint != '':
      new_checkpoint = config.eval.checkpoint
    else:
      new_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    if latest_checkpoint != new_checkpoint:
      latest_checkpoint = new_checkpoint
      saver.restore(sess, latest_checkpoint)
      current_step = tf.train.load_variable(latest_checkpoint, 'global_step')

      test_size = dataset_size(config.eval.file_list_path)
      time_start = time.time()

      mean_accuracy, mean_accuracy_minus_1 = 0.0, 0.0

      steps = int(test_size / config.eval.batch_size) if int(test_size / config.eval.batch_size) else 1
      num = 0
      for _ in range(steps):
        val, slabel, _ = sess.run([d_predictions, label_val, file_names])
        acc, acc1, num_ = accuracy(slabel, val, config.vocab, config.r_vocab, config.lpr_patterns)
        mean_accuracy += acc
        mean_accuracy_minus_1 += acc1
        num += num_

      writer.add_summary(
        tf.Summary(value=[tf.Summary.Value(tag='evaluation/acc', simple_value=float(mean_accuracy / num)),
                          tf.Summary.Value(tag='evaluation/acc-1', simple_value=float(mean_accuracy_minus_1 / num))]),
        current_step)
      print('Test acc: {}'.format(mean_accuracy / num))
      print('Test acc-1: {}'.format(mean_accuracy_minus_1 / num))
      print('Time per step: {} for test size {}'.format(time.time() - time_start / steps, test_size))
    else:
      if wait_iters % 12 == 0:
        sys.stdout.write('\r')
        for _ in range(11 + wait_iters // 12):
          sys.stdout.write(' ')
        sys.stdout.write('\r')
        for _ in range(1 + wait_iters // 12):
          sys.stdout.write('|')
      else:
        sys.stdout.write('.')
      sys.stdout.flush()
      time.sleep(5)
      wait_iters += 1
    if config.eval.checkpoint != '':
      break


  coord.request_stop()
  coord.join(threads)
  sess.close()

def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  validate(cfg)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
