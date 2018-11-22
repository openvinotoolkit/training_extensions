import os

import tensorflow as tf

from utils.helpers import download_archive_and_extract


def download_test_data(root_dir):
  src_path = 'https://drive.google.com/uc?export=download&id=13oX2m2XtrKlw3pLpxFd_XCjvObmWFmDO'
  target_dir = os.path.join(root_dir, 'data/test')

  tf.logging.info('Downloading "{0}" to "{1}"'.format(src_path, target_dir))
  download_archive_and_extract(src_path, target_dir)

