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
import cv2
import tensorflow as tf

from vehicle_attributes.trainer import create_session, resnet_v1_10_1, InputTrainData
from tfutils.helpers import load_module

def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of vehicle attributes model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()

def train(config):
  cv2.setNumThreads(1)

  session_config = create_session(config, 'train')

  run_config = tf.estimator.RunConfig(session_config=session_config,
                                      keep_checkpoint_every_n_hours=config.train.keep_checkpoint_every_n_hours,
                                      save_summary_steps=config.train.save_summary_steps,
                                      save_checkpoints_steps=config.train.save_checkpoints_steps,
                                      tf_random_seed=config.train.random_seed)

  va_predictor = tf.estimator.Estimator(
    model_fn=resnet_v1_10_1,
    params=config.resnet_params,
    model_dir=config.model_dir,
    config=run_config)

  input_data = InputTrainData(batch_size=config.train.batch_size,
                              input_shape=config.input_shape,
                              json_path=config.train.annotation_path)

  va_predictor.train(
    input_fn=input_data.input_fn,
    steps=config.train.steps,
    hooks=[])

def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  train(cfg)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
