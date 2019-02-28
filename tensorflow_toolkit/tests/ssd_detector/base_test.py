import os
import tempfile
import unittest
from datetime import datetime

from tests.ssd_detector import CONFIG


class BaseTest(unittest.TestCase):
  @staticmethod
  def create_model_dir(prefix):
    if not os.path.exists(CONFIG.MODEL_DIR):
      os.makedirs(CONFIG.MODEL_DIR)
    prefix = 'test_{}_{}_'.format(prefix, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    CONFIG.MODEL_DIR = tempfile.mkdtemp(prefix=prefix, dir=CONFIG.MODEL_DIR)
