from __future__ import print_function
import argparse
import os
import subprocess


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_io

from lpr.trainer import inference
from utils.helpers import dump_frozen_graph, load_module


def parse_args():
  parser = argparse.ArgumentParser(description='Export model in IE format')
  parser.add_argument('path_to_config', help='Path to a config.py')
  parser.add_argument('mo', help="Path to model optimizer 'mo.py' script")
  return parser.parse_args()


def execute_tfmo(mo_py_path, frozen, input_shape, batch_size, precision):

  assert frozen.endswith('.pb.frozen')
  assert batch_size > 0
  assert precision in ('FP32', 'FP16')
  folder = os.path.dirname(frozen)
  name = os.path.splitext(frozen)[0].replace('.pb', '')

  input_shape = [batch_size] + list(input_shape[1:])


  params = (
    'python3',
    '-u',
    mo_py_path or 'mo.py',
    '--framework={}'.format('tf'),
    '--reverse_input_channels',
    '--scale={}'.format(255),
    '--input_shape=[{}]'.format(','.join([str(shape) for shape in input_shape])),
    '--input={}'.format('0:Conv/Conv2D'),
    '--output={}'.format('stack'),
    '--input_model={}'.format(frozen),
    '--output_dir={}'.format(folder),
    '--model_name={}'.format(name),
    '--data_type={}'.format(precision)
  )

  if mo_py_path:
    subprocess.call([p for p in params if p])
  else:
    print('\nPath to `mo.py` is not specified. Please provide correct path to Model Optimizer `mo.py` script')

# pylint: disable=too-many-locals
def export(config, tfmo, batch_size=1, precision='FP32'):
  shape = (None,) + tuple(config.input_shape) # NHWC, dynamic batch
  graph = tf.Graph()
  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      input_tensor = tf.placeholder(dtype=tf.float32, shape=shape)
      prob = inference(config.rnn_cells_num, input_tensor, config.num_classes)
      prob = tf.transpose(prob, (1, 0, 2))
      data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])
      result = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)
      predictions = [tf.to_int32(p) for p in result[0]]
      _ = tf.stack([tf.sparse_to_dense(p.indices, [1, config.max_lp_length], p.values, default_value=-1)
                    for p in predictions])
      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  sess = tf.Session(graph=graph)
  sess.run(init)
  checkpoints_dir = config.model_dir
  latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
  saver.restore(sess, latest_checkpoint)
  frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["stack"])
  tf.train.write_graph(sess.graph, os.path.join(config.model_dir, 'ie_model/'), 'graph.pbtxt', as_text=True)
  path_to_frozen_model = graph_io.write_graph(frozen, os.path.join(config.model_dir, 'ie_model/'),
                                              'graph.pb.frozen', as_text=False)
  execute_tfmo(tfmo, path_to_frozen_model, shape, batch_size, precision)


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  export(cfg, args.mo, 1, 'FP32') #set batch_size and precision


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
