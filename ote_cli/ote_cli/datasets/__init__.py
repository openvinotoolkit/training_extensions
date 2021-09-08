# TODO(ikrylov): re-do

try:
    from .object_detection.dataset import ObjectDetectionDataset
except ImportError as e:
    print(e)
    ObjectDetectionDataset = None

try:
    from torchreid.integration.sc.utils import ClassificationDatasetAdapter
    ClassificationDataset = ClassificationDatasetAdapter
except ImportError as e:
    print(e)
    ClassificationDataset = None


def get_dataset_class(name):
    registry = {
        'detection': ObjectDetectionDataset,
        'classification': ClassificationDataset
    }

    return registry[str(name).lower()]

__all__ = [
    'ObjectDetectionDataset',
    'ClassificationDataset',
    'get_dataset_class',
]
