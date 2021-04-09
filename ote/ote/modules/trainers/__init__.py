from .base import BaseTrainer
from .instance_segmentation import InstanceSegmentationTrainer
from .mmaction import MMActionTrainer
from .mmdetection import MMDetectionTrainer
from .mmpose import MMPoseTrainer
from .reid import ReidTrainer

__all__ = [
    'BaseTrainer',
    'InstanceSegmentationTrainer',
    'MMActionTrainer',
    'MMDetectionTrainer',
    'MMPoseTrainer',
    'InstanceSegmentationTrainer',
    'ReidTrainer',
]
