import os

MODEL_TEMPLATE_FILENAME = 'template.yaml'
MODULES_CONFIG_FILENAME = 'modules.yaml'

DEFAULT_MMDETECTION_DIR = os.path.abspath(f'{os.path.dirname(__file__)}/../../../external/mmdetection')
MMDETECTION_DIR = os.getenv('MMDETECTION_DIR', DEFAULT_MMDETECTION_DIR)
MMDETECTION_TOOLS = os.path.join(MMDETECTION_DIR, 'tools')

DEFAULT_MMACTION_DIR = os.path.abspath(f'{os.path.dirname(__file__)}/../../../external/mmaction2')
MMACTION_DIR = os.getenv('MMACTION_DIR', DEFAULT_MMACTION_DIR)
MMACTION_TOOLS = os.path.join(MMACTION_DIR, 'tools')

DEFAULT_REID_DIR = os.path.abspath(f'{os.path.dirname(__file__)}/../../../external/deep-object-reid')
REID_DIR = os.getenv('REID_DIR', DEFAULT_REID_DIR)
REID_TOOLS = os.path.join(REID_DIR, 'tools')
