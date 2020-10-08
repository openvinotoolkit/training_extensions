import os

DEFAULT_MMDETECTION_DIR = f'{os.path.dirname(__file__)}/../../../../external/mmdetection'
MMDETECTION_DIR = os.getenv('MMDETECTION_DIR', DEFAULT_MMDETECTION_DIR)
MMDETECTION_TOOLS = os.path.join(MMDETECTION_DIR, 'tools')
