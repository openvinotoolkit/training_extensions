from .object_detection.dataset import ObjectDetectionDataset


def get_dataset_class(name):
    registry = {
        'object detection': ObjectDetectionDataset
    }

    return registry[name.lower()]

__all__ = [
    'ObjectDetectionDataset',
    'get_dataset_class',
]
