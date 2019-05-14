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

import argparse
import os
import random
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lpr.utils import dataset_size
from lpr.trainer import inference, decode_beams
from tfutils.helpers import load_module


def parse_args():
  parser = argparse.ArgumentParser(description='Infer of a trained model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


def read_data(height, width, channels_num, list_file_name, batch_size=1):
  reader = tf.TextLineReader()
  _, value = reader.read(list_file_name)
  filename = value
  image_filename = tf.read_file(filename)
  rgb_image = tf.image.decode_png(image_filename, channels=channels_num)
  rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
  resized_image = tf.image.resize_images(rgb_image_float, [height, width])
  resized_image.set_shape([height, width, channels_num])

  image_batch, file_batch = tf.train.batch([resized_image, filename], batch_size=batch_size,
                                           allow_smaller_final_batch=True)
  return image_batch, file_batch


def data_input(height, width, channels_num, filename, batch_size=1):
  files_string_producer = tf.train.string_input_producer([filename])
  image, filename = read_data(height, width, channels_num, files_string_producer, batch_size)
  return image, filename

# pylint: disable=too-many-statements, too-many-locals
def infer(config):
  if hasattr(config.infer, 'random_seed'):
    np.random.seed(config.infer.random_seed)
    tf.set_random_seed(config.infer.random_seed)
    random.seed(config.infer.random_seed)

  if hasattr(config.infer.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  height, width, channels_num = config.input_shape
  rnn_cells_num = config.rnn_cells_num

  graph = tf.Graph()

  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      inp_data, filenames = data_input(height, width, channels_num, config.infer.file_list_path,
                                       batch_size=config.infer.batch_size)

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

  latest_checkpoint = config.infer.checkpoint
  if config.infer.checkpoint == '':
    latest_checkpoint = tf.train.latest_checkpoint(config.model_dir)

  saver.restore(sess, latest_checkpoint)

  infer_size = dataset_size(config.infer.file_list_path)
  steps = int(infer_size / config.infer.batch_size) if int(infer_size / config.infer.batch_size) else 1

  for _ in range(steps):

    vals, batch_filenames = sess.run([d_predictions, filenames])
    print(batch_filenames)
    pred = decode_beams(vals, config.r_vocab)

    for i, filename in enumerate(batch_filenames):
      filename = filename.decode('utf-8')


      img = cv2.imread(filename)
      size = cv2.getTextSize(pred[i], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
      text_width = size[0][0]
      text_height = size[0][1]

      img_he, img_wi, _ = img.shape
      img = cv2.copyMakeBorder(img, 0, text_height + 10, 0,
                               0 if text_width < img_wi else text_width - img_wi, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
      cv2.putText(img, pred[i], (0, img_he + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

      cv2.imshow('License Plate', img)
      key = cv2.waitKey(0)
      if key == 27:
        break

  coord.request_stop()
  coord.join(threads)
  sess.close()


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  infer(cfg)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
