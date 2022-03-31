from __future__ import absolute_import, print_function

from .version import __version__
from .task import OTETextToSpeechTask
from .openvino_task import OpenVINOTTSTask

# All other objects (classes, functions, etc.) in this package be considered internal.
__all__ = ['__version__', 'OTETextToSpeechTask', 'OpenVINOTTSTask']
