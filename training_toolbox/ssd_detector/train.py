import argparse
import os

import cv2
import tensorflow as tf

from utils import load_module
from ssd_detector.trainer import create_session, detection_model, InputTrainData


os.environ['MKL_NUM_THREADS'] = '1'


def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of a detection model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()


def train(config):
  cv2.setNumThreads(1)

  session_config = create_session(config, 'train')

  input_data = InputTrainData(batch_size=config.train.batch_size, input_shape=config.input_shape,
                              json_path=config.train.annotation_path,
                              fill_with_current_image_mean=config.train.fill_with_current_image_mean,
                              cache_type=config.train.cache_type,
                              classes=config.classes, num_parallel_calls=config.train.execution.transformer_parallel_calls,
                              prefetch_size=config.train.execution.transformer_prefetch_size)

  steps_per_epoch = config.train.save_checkpoints_steps if config.train.save_checkpoints_steps \
    else input_data.dataset_size // input_data.batch_size

  run_config = tf.estimator.RunConfig(session_config=session_config,
                                      keep_checkpoint_every_n_hours=config.train.keep_checkpoint_every_n_hours,
                                      save_summary_steps=config.train.save_summary_steps,
                                      save_checkpoints_steps=steps_per_epoch,
                                      tf_random_seed=config.train.random_seed)

  config.detector_params['steps_per_epoch'] = steps_per_epoch
  config.detector_params['log_dir'] = config.model_dir

  predictor = tf.estimator.Estimator(
    model_fn=detection_model,
    params=config.detector_params,
    model_dir=config.model_dir,
    config=run_config
  )

  predictor.train(input_fn=input_data.input_fn, steps=config.train.steps, max_steps=config.train.max_steps)


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  train(cfg)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
