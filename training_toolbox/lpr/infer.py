import argparse
import os
import random
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lpr.toolbox.utils import dataset_size
from lpr.trainer import inference, decode_beams
from utils.helpers import load_module



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

  image_batch, file_batch = tf.train.batch([resized_image, filename], batch_size=batch_size)
  return image_batch, file_batch


def data_input(height, width, channels_num, filename, batch_size=1):
  files_string_producer = tf.train.string_input_producer([filename])
  image, filename = read_data(height, width, channels_num, files_string_producer, batch_size)
  return image, filename

# pylint: disable=too-many-locals, too-many-statements
def infer(config):
  if hasattr(config.infer, 'random_seed'):
    np.random.seed(config.infer.random_seed)
    tf.set_random_seed(config.infer.random_seed)
    random.seed(config.infer.random_seed)

  if hasattr(config.infer.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  height, width, channels_num = config.input_shape
  max_lp_length = config.max_lp_length
  rnn_cells_num = config.rnn_cells_num

  graph = tf.Graph()

  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      inp_data, filenames = data_input(height, width, channels_num, config.infer.file_list_path, batch_size=1)

      prob = inference(rnn_cells_num, inp_data, config.num_classes)
      prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC

      data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size

      result = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)

      predictions = [tf.to_int32(p) for p in result[0]]
      d_predictions = tf.stack([tf.sparse_to_dense(p.indices, [1, max_lp_length], p.values, default_value=-1)
                                for p in predictions])

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

  steps = dataset_size(config.infer.file_list_path)

  for _ in range(steps):

    val, filename = sess.run([d_predictions, filenames])
    filename = filename[0].decode('utf-8')
    pred = decode_beams(val, config.r_vocab)[0]

    img = cv2.imread(filename)
    size = cv2.getTextSize(pred[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_width = size[0][0]
    text_height = size[0][1]

    img_he, img_wi, _ = img.shape
    img = cv2.copyMakeBorder(img, 0, text_height + 10, 0,
                             0 if text_width < img_wi else text_width - img_wi, cv2.BORDER_CONSTANT,
                             value=(255, 255, 255))
    cv2.putText(img, pred[0], (0, img_he + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

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
