import importlib
import os

from .utils import download_test_data

os.environ['MKL_NUM_THREADS'] = '1'
OPEN_VINO_DIR = os.environ.get('OPEN_VINO_DIR', '')

CONFIG = importlib.import_module('.config', __package__)


def setUpModule():
  download_test_data(CONFIG.root_dir)
