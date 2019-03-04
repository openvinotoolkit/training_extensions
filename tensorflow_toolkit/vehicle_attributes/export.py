import os
import argparse
import subprocess

import tensorflow as tf
import tensorflow.contrib.slim as slim

from vehicle_attributes.networks.resnet_10_bn import resnet_v1_10, resnet_arg_scope
from utils.helpers import load_module

def parse_args():
  parser = argparse.ArgumentParser(description='Export vehicle attributes model in IE format')
  parser.add_argument('mo', help="Path to model optimizer 'mo.py' script")
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()

def execute_tfmo(mo_py_path, frozen, input_shape):
  """
  This function executes Model Optimizer for TensorFlow.
  As a result it has converted to Inference Engine IR model in the 'ie_model' folder.
  :param mo_py_path: path to Model Optimizer mo.py
  :param frozen: path to frozen pb-file to convert to IE
  """
  assert frozen.endswith('.pb.frozen')
  folder = os.path.dirname(frozen)
  name = os.path.basename(frozen).replace('.pb.frozen', '')

  params = (
    'python3', '-u', mo_py_path or 'mo.py',
    '--framework={}'.format('tf'),
    '--input_model={}'.format(frozen),
    '--scale={}'.format(255),
    '--input_shape=[{}]'.format(','.join([str(shape) for shape in input_shape])),
    '--output_dir={}'.format(folder),
    '--model_name={}'.format(name),
  )

  if mo_py_path:
    subprocess.call([p for p in params if p])
  else:
    print('\nPath to `mo.py` is not specified. Please provide correct path to Model Optimizer `mo.py` script')

def export(mo_py_path, config):

  with tf.Session() as sess:
    shape = [None] + list(config.input_shape)
    inputs = tf.placeholder(dtype=tf.float32, shape=shape, name="input")
    with slim.arg_scope(resnet_arg_scope()):
      _ = resnet_v1_10(inputs, is_training=False)
    saver = tf.train.Saver()

    checkpoints_dir = config.model_dir
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    saver.restore(sess, os.path.abspath(latest_checkpoint))

    frozen = tf.graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          ['resnet_v1_10/type', 'resnet_v1_10/color'])

    tf.train.write_graph(frozen, os.path.join(config.model_dir, 'ie_model/'), 'graph.pb_txt', as_text=True)
    frozen_path = tf.train.write_graph(frozen, os.path.join(config.model_dir, 'ie_model/'),
                                       'graph.pb.frozen', as_text=False)

    batch_size = 1
    input_shape = [batch_size] + list(shape[1:])
    execute_tfmo(mo_py_path, frozen_path, input_shape)

def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  export(args.mo, cfg)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
