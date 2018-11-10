import os
import tempfile
import unittest
from datetime import datetime

from tests.ssd_detector import CONFIG


class BaseTest(unittest.TestCase):
  @staticmethod
  def create_model_dir(prefix):
    if not os.path.exists(CONFIG.model_dir):
      os.makedirs(CONFIG.model_dir)
    prefix = 'test_{}_{}_'.format(prefix, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    CONFIG.model_dir = tempfile.mkdtemp(prefix=prefix, dir=CONFIG.model_dir)
