import os

import tensorflow as tf

from utils.helpers import download_archive_and_extract


def download_test_data(root_dir):
  src_path = 'https://download.01.org/openvinotoolkit/2018_R3/training_toolbox_tensorflow/data/test/test_v1.0.0.zip'
  target_dir = os.path.join(root_dir, 'data/test')

  tf.logging.info('Downloading "{0}" to "{1}"'.format(src_path, target_dir))
  download_archive_and_extract(src_path, target_dir)

