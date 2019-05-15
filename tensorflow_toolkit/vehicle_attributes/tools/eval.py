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

import time
import sys
import argparse
import pprint

import tensorflow as tf

from tfutils.helpers import load_module
from vehicle_attributes.trainer import create_session, resnet_v1_10_1, InputEvalData

def parse_args():
  parser = argparse.ArgumentParser(description='Perform evaluation of a trained vehicle attributes model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()

def eval_loop(estimator, eval_data, config):
  latest_checkpoint = None
  wait_iters = 0
  save_images_step = 0
  new_checkpoint = None
  while True:
    while new_checkpoint is None:
      time.sleep(1)
      new_checkpoint = tf.train.latest_checkpoint(config.model_dir)
    new_checkpoint = tf.train.latest_checkpoint(config.model_dir)
    if latest_checkpoint != new_checkpoint:
      latest_checkpoint = new_checkpoint

      print('\nEvaluating {0}'.format(latest_checkpoint))
      print('=============================================================')

      start = time.time()

      eval_results = estimator.evaluate(input_fn=eval_data.input_fn, name='val')
      def without_keys(rdict, keys):
        return {x: rdict[x] for x in rdict if x not in keys}
      results = without_keys(eval_results, {"global_step", "loss"})
      printer = pprint.PrettyPrinter(indent=4)
      printer.pprint(results)

      finish = time.time()

      print('=============================================================')
      print('[{0}]: {1} evaluation time = {2}\n'.format(latest_checkpoint, 'val', finish - start))

      save_images_step += 1
      wait_iters = 0
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
      time.sleep(0.5)
      wait_iters += 1

def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)

  session_config = create_session(cfg, 'eval')

  run_config = tf.estimator.RunConfig(session_config=session_config)

  va_estimator = tf.estimator.Estimator(
    model_fn=resnet_v1_10_1,
    params=cfg.resnet_params,
    model_dir=cfg.model_dir,
    config=run_config)

  eval_data = InputEvalData(batch_size=cfg.eval.batch_size,
                            input_shape=cfg.input_shape,
                            json_path=cfg.eval.annotation_path)

  eval_loop(va_estimator, eval_data, cfg)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
