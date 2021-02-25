from .base import BaseTrainer
from .instance_segmentation import InstanceSegmentationTrainer
from .mmaction import MMActionTrainer
from .mmdetection import MMDetectionTrainer
from .reid import ReidTrainer

__all__ = [
    'BaseTrainer',
    'MMActionTrainer',
    'MMDetectionTrainer',
    'InstanceSegmentationTrainer',
    'ReidTrainer',
]
