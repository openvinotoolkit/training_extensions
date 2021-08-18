from .object_detection.dataset import ObjectDetectionDataset


def get_dataset_class(name):
    registry = {
        'detection': ObjectDetectionDataset
    }

    return registry[str(name).lower()]

__all__ = [
    'ObjectDetectionDataset',
    'get_dataset_class',
]
