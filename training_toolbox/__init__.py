from os import path, environ
import sys


# import slim
EXTERNAL_RESEARCH_DIR = path.realpath(path.dirname(__file__) + '../../external/models/research/')
sys.path.append(environ.get('TF_RESEARCH_DIR', EXTERNAL_RESEARCH_DIR))
sys.path.append(environ.get('SLIM_RESEARCH_DIR', path.join(EXTERNAL_RESEARCH_DIR, 'slim')))
