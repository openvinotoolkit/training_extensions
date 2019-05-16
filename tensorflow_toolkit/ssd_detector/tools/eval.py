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
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from cachetools import cached, Cache

from ssd_detector.trainer import create_session, detection_model, InputValData
from ssd_detector.toolbox.coco_metrics_eval import calc_coco_metrics
from ssd_detector.toolbox.summary import group_ssd_heads, write_histogram_2d
from tfutils.helpers import draw_bboxes, load_module


def parse_args():
  parser = argparse.ArgumentParser(description='Perform evaluation of a detection model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


@cached(Cache(100))
def load_coco(path):
  from pycocotools.coco import COCO
  return COCO(path)


# pylint: disable=too-many-locals,too-many-arguments
def eval_dataset(annotations, config, eval_name, checkpoint_path, session_config, sample_images=None,
                 dump_priors_info=True):
  log_dir = os.path.join(config.MODEL_DIR, 'eval_' + eval_name)
  run_config = tf.estimator.RunConfig(session_config=session_config)

  # Override default FileWriter. Don't store the graph definition.
  # pylint: disable=protected-access
  tf.summary.FileWriterCache._cache[log_dir] = tf.summary.FileWriter(log_dir, graph=None)

  input_data = InputValData(config.eval.batch_size, config.input_shape, config.eval.annotation_path[eval_name],
                            classes=config.classes,
                            num_parallel_calls=config.eval.execution.transformer_parallel_calls,
                            prefetch_size=config.eval.execution.transformer_prefetch_size)

  config.detector_params['log_dir'] = log_dir
  config.detector_params['steps_per_epoch'] = math.ceil(input_data.dataset_size / input_data.batch_size)

  detector_params = config.detector_params.copy()
  if not dump_priors_info:
    detector_params['collect_priors_summary'] = False
  predictor = tf.estimator.Estimator(
    model_fn=detection_model,
    params=detector_params,
    model_dir=config.MODEL_DIR,
    config=run_config
  )

  eval_results = predictor.evaluate(input_fn=input_data.input_fn, name=eval_name, checkpoint_path=checkpoint_path)
  writer = tf.summary.FileWriterCache.get(log_dir)
  predictions = eval_results['predictions']

  if checkpoint_path is not None:
    step = tf.train.load_variable(checkpoint_path, 'global_step')
  else:
    step = tf.train.load_variable(config.MODEL_DIR, 'global_step')

  if dump_priors_info:
    summaries = []
    for key, assigned_priors in eval_results.items():
      if key.startswith('prior_histogram/'):
        name = key.replace('prior_histogram/', '', 1)
        summaries.append(tf.Summary.Value(tag=name, simple_value=np.sum(assigned_priors)))
    if summaries:
      writer.add_summary(tf.Summary(value=summaries), step)

    group = group_ssd_heads(eval_results)

    write_histogram_2d(group, step, log_dir, use_lognorm=True)
    write_histogram_2d(group, step, log_dir, use_lognorm=False)

  metrics = calc_coco_metrics(annotations, predictions, config.classes)
  summaries = [tf.Summary.Value(tag='accuracy/' + name, simple_value=val) for name, val in metrics.items()]
  writer.add_summary(tf.Summary(value=summaries), step)

  if sample_images is not None:
    preprocessed_images = [item[0] for item in sample_images]
    annotations = [pickle.loads(item[1]) for item in sample_images]
    images = [item[2] for item in sample_images]

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=np.array(preprocessed_images),
      y=None,
      num_epochs=1,
      batch_size=config.eval.batch_size,
      shuffle=False
    )

    predictions = predictor.predict(input_fn=predict_input_fn)
    predictions = [pred for pred in predictions]  # Get values from the generator

    images = draw_bboxes(images, annotations, predictions, config.classes, conf_threshold=0.1)

    def write_images(images, step, writer):
      summaries = []
      for idx, img in enumerate(images):
        encoded_image = cv2.imencode('.jpg', img)[1].tostring()
        img_sum = tf.Summary.Image(encoded_image_string=encoded_image, height=img.shape[0], width=img.shape[1])
        summaries.append(tf.Summary.Value(tag='img/{0}'.format(idx), image=img_sum))

      summary = tf.Summary(value=summaries)
      writer.add_summary(summary, step)

    write_images(images, step, writer)

  writer.flush()


@cached(Cache(100))
def get_sample_images(annotation_path, params):
  return InputValData.sample_data(annotation_path, *pickle.loads(params))


def eval_once(config, checkpoint, save_sample_prediction, dump_priors_info=True):
  session_config = create_session(config, 'eval')
  print('\nEvaluating {0}'.format(checkpoint))
  print('=============================================================')

  for dataset_name in config.eval.datasets:
    start = time.time()

    if save_sample_prediction:
      sample_images = get_sample_images(config.eval.annotation_path[dataset_name],
                                        pickle.dumps((config.eval.vis_num, config.input_shape, config.classes)))
    else:
      sample_images = None

    annotation = load_coco(config.eval.annotation_path[dataset_name])

    proc = mp.Process(target=eval_dataset,
                      args=(annotation, config, dataset_name, checkpoint, session_config, sample_images,
                            dump_priors_info))
    proc.start()
    proc.join()

    finish = time.time()
    print('=============================================================')
    print('[{0}]: {1} evaluation time = {2}\n'.format(checkpoint, dataset_name, finish - start))


def eval_loop(config):
  _ = create_session(config, 'eval')
  latest_checkpoint = None
  wait_iters = 0
  save_images_step = 0
  dump_priors_info = True
  while True:
    new_checkpoint = tf.train.latest_checkpoint(config.MODEL_DIR)
    if latest_checkpoint != new_checkpoint:
      latest_checkpoint = new_checkpoint

      save_sample_prediction = save_images_step % config.eval.save_images_step == 0 if \
        config.eval.save_images_step != 0 else False
      eval_once(config, latest_checkpoint, save_sample_prediction, dump_priors_info)
      dump_priors_info = False

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
      time.sleep(5)
      wait_iters += 1


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  eval_loop(cfg)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.WARN)
  tf.app.run(main)
