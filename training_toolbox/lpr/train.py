import random
import os
import time
import argparse
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from lpr.toolbox.utils import dataset_size
from lpr.trainer import CTCUtils, inference, InputData, decode, LPRVocab
from utils.helpers import load_module


def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of a model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


def train(config):
  if hasattr(config.train, 'random_seed'):
    np.random.seed(config.train.random_seed)
    tf.set_random_seed(config.train.random_seed)
    random.seed(config.train.random_seed)

  if hasattr(config.train.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  vocab, r_vocab, num_classes = LPRVocab.create_vocab(config.train.train_list_file_path,
                                                      config.train.val_list_file_path,
                                                      config.use_h_concat,
                                                      config.use_oi_concat)

  CTCUtils.vocab = vocab
  CTCUtils.r_vocab = r_vocab

  input_train_data = InputData(batch_size=config.train.batch_size,
                               input_shape=config.input_shape,
                               file_list_path=config.train.train_list_file_path,
                               apply_basic_aug=config.train.apply_basic_aug,
                               apply_stn_aug=config.train.apply_stn_aug,
                               apply_blur_aug=config.train.apply_blur_aug)

  input_val_data = InputData(batch_size=config.train.val_batch_size,
                             input_shape=config.input_shape,
                             file_list_path=config.train.val_list_file_path,
                             apply_basic_aug=False,
                             apply_stn_aug=False,
                             apply_blur_aug=False)

  graph = tf.Graph()
  with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_inp, train_labels = input_train_data.input_fn()
    with slim.arg_scope([slim.fully_connected, slim.conv2d], reuse=True):
      validation_inp, validation_labels = input_val_data.input_fn()
    train_mode = tf.placeholder(tf.bool, (), name='train_mode')

    input_data = tf.cond(train_mode, lambda: train_inp, lambda: validation_inp)
    input_labels = tf.cond(train_mode, lambda: train_labels, lambda: validation_labels, name='inp_labels')

    prob = inference(config.train.rnn_cells_num, input_data, num_classes)
    prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC

    data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size
    ctc = tf.py_func(CTCUtils.compute_ctc_from_labels, [input_labels], [tf.int64, tf.int64, tf.int64])
    ctc_labels = tf.to_int32(tf.SparseTensor(ctc[0], ctc[1], ctc[2]))

    predictions = tf.to_int32(
      tf.nn.ctc_beam_search_decoder(prob, data_length, merge_repeated=False, beam_width=10)[0][0])
    d_predictions = tf.sparse_tensor_to_dense(predictions, default_value=-1, name='d_predictions')

    error_rate = tf.reduce_mean(tf.edit_distance(predictions, ctc_labels, normalize=False), name='error_rate')

    loss = tf.reduce_mean(
      tf.nn.ctc_loss(inputs=prob, labels=ctc_labels, sequence_length=data_length, ctc_merge_repeated=True), name='loss')

    learning_rate = tf.train.piecewise_constant(global_step, [150000, 200000],
                                                [config.train.learning_rate, 0.1 * config.train.learning_rate,
                                                 0.01 * config.train.learning_rate])
    # grad_noise_scale = tf.train.piecewise_constant(global_step, [100000],
    #                                             [args.grad_noise_scale, 0.1 * args.grad_noise_scale])
    train_step = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, config.train.opt_type,
                                                 config.train.grad_noise_scale, name='train_step')

    tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1000, write_version=tf.train.SaverDef.V2)

  conf = tf.ConfigProto()
  if hasattr(config.train.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.train.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  session = tf.Session(graph=graph, config=conf)
  coordinator = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

  session.run('init')

  work_path = config.model_dir
  if os.path.exists(os.path.join(work_path, 'snapshot{:06d}.ckpt'.format(config.train.start_iter))):
    tf.logging.info('Restore from: ' + work_path)
    saver.restore(session, os.path.join(work_path, 'snapshot{:06d}.ckpt'.format(config.train.start_iter)))

  writer = None
  if config.train.need_to_save_log:
    writer = tf.summary.FileWriter(work_path, session.graph)

  graph.finalize()

  val_size = dataset_size(config.train.val_list_file_path)
  val_steps = config.train.val_steps
  if config.train.val_steps < 1:
    val_steps = int(val_size / config.train.val_batch_size)  # HACK: batch norm pseudo-test

  for i in range(config.train.steps):
    curr_step, train_loss, _ = session.run([global_step, loss, train_step], feed_dict={train_mode: True})

    if i % config.train.display_iter == 0:
      if config.train.need_to_save_log:
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='train/loss',
                                                              simple_value=float(train_loss))]),
                           curr_step)
        writer.flush()

      tf.logging.info('Iteration: ' + str(curr_step) + ', Train loss: ' + str(train_loss))

    if i % config.train.val_iter == 0:
      t = time.time()

      mean_error, mean_accuracy = 0.0, 0.0

      current_step = 0
      val_true_labels = []
      for j in range(val_steps):
        current_step, val_predicted_values, val_true_labels, val_error = session.run(
          [global_step, d_predictions, input_labels, error_rate],
          feed_dict={train_mode: False})
        mean_accuracy += CTCUtils.accuracy(val_true_labels, val_predicted_values) / val_steps
        mean_error += val_error / val_steps

      if config.train.need_to_save_log:
        writer.add_summary(
          tf.Summary(value=[tf.Summary.Value(tag='validation/err', simple_value=float(mean_error)),
                            tf.Summary.Value(tag='validation/acc', simple_value=float(mean_accuracy))]), current_step)
        writer.flush()

      tf.logging.info('Iteration: ' + str(current_step) + ', GT: ' + val_true_labels[0].decode("utf-8") + ' -- ' +
                      decode(val_predicted_values, r_vocab)[0] + ', Error: ' +
                      str(mean_error) + ', Accuracy: ' + str(mean_accuracy) + ', Time per step: ' +
                      str((time.time() - t) / val_steps) + ' secs.')

    if ((curr_step % config.train.snap_iter == 0 or curr_step == config.train.steps)
        and config.train.need_to_save_weights):
      saver.save(session, work_path + '/snapshot{:06d}.ckpt'.format(curr_step))

  coordinator.request_stop()
  coordinator.join(threads)
  session.close()


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  train(cfg)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
