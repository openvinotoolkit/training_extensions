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

import random
import os
import argparse
import numpy as np
import tensorflow as tf
from lpr.trainer import CTCUtils, inference, InputData
from tfutils.helpers import load_module


def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of a model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  parser.add_argument('--init_checkpoint', default=None, help='Path to checkpoint')
  return parser.parse_args()

# pylint: disable=too-many-locals, too-many-statements
def train(config, init_checkpoint):
  if hasattr(config.train, 'random_seed'):
    np.random.seed(config.train.random_seed)
    tf.set_random_seed(config.train.random_seed)
    random.seed(config.train.random_seed)

  if hasattr(config.train.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  CTCUtils.vocab = config.vocab
  CTCUtils.r_vocab = config.r_vocab

  input_train_data = InputData(batch_size=config.train.batch_size,
                               input_shape=config.input_shape,
                               file_list_path=config.train.file_list_path,
                               apply_basic_aug=config.train.apply_basic_aug,
                               apply_stn_aug=config.train.apply_stn_aug,
                               apply_blur_aug=config.train.apply_blur_aug)


  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    input_data, input_labels = input_train_data.input_fn()

    prob = inference(config.rnn_cells_num, input_data, config.num_classes)
    prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC

    data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size
    ctc = tf.py_func(CTCUtils.compute_ctc_from_labels, [input_labels], [tf.int64, tf.int64, tf.int64])
    ctc_labels = tf.to_int32(tf.SparseTensor(ctc[0], ctc[1], ctc[2]))

    predictions = tf.to_int32(
      tf.nn.ctc_beam_search_decoder(prob, data_length, merge_repeated=False, beam_width=10)[0][0])
    tf.sparse_tensor_to_dense(predictions, default_value=-1, name='d_predictions')
    tf.reduce_mean(tf.edit_distance(predictions, ctc_labels, normalize=False), name='error_rate')

    loss = tf.reduce_mean(
      tf.nn.ctc_loss(inputs=prob, labels=ctc_labels, sequence_length=data_length, ctc_merge_repeated=True), name='loss')

    learning_rate = tf.train.piecewise_constant(global_step, [150000, 200000],
                                                [config.train.learning_rate, 0.1 * config.train.learning_rate,
                                                 0.01 * config.train.learning_rate])
    opt_loss = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, config.train.opt_type,
                                               config.train.grad_noise_scale, name='train_step')

    tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1000, write_version=tf.train.SaverDef.V2, save_relative_paths=True)

  conf = tf.ConfigProto()
  if hasattr(config.train.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.train.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  session = tf.Session(graph=graph, config=conf)
  coordinator = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

  session.run('init')

  if init_checkpoint:
    tf.logging.info('Initialize from: ' + init_checkpoint)
    saver.restore(session, init_checkpoint)
  else:
    lastest_checkpoint = tf.train.latest_checkpoint(config.model_dir)
    if lastest_checkpoint:
      tf.logging.info('Restore from: ' + lastest_checkpoint)
      saver.restore(session, lastest_checkpoint)

  writer = None
  if config.train.need_to_save_log:
    writer = tf.summary.FileWriter(config.model_dir, session.graph)

  graph.finalize()


  for i in range(config.train.steps):
    curr_step, curr_learning_rate, curr_loss, curr_opt_loss = session.run([global_step, learning_rate, loss, opt_loss])

    if i % config.train.display_iter == 0:
      if config.train.need_to_save_log:

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='train/loss',
                                                              simple_value=float(curr_loss)),
                                             tf.Summary.Value(tag='train/learning_rate',
                                                              simple_value=float(curr_learning_rate)),
                                             tf.Summary.Value(tag='train/optimization_loss',
                                                              simple_value=float(curr_opt_loss))
                                             ]),
                           curr_step)
        writer.flush()

      tf.logging.info('Iteration: ' + str(curr_step) + ', Train loss: ' + str(curr_loss))

    if ((curr_step % config.train.save_checkpoints_steps == 0 or curr_step == config.train.steps)
        and config.train.need_to_save_weights):
      saver.save(session, config.model_dir + '/model.ckpt-{:d}.ckpt'.format(curr_step))

  coordinator.request_stop()
  coordinator.join(threads)
  session.close()


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  train(cfg, args.init_checkpoint)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
