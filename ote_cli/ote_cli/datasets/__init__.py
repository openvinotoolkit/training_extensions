# TODO(ikrylov): re-do

try:
    from .object_detection.dataset import ObjectDetectionDataset
except ImportError as e:
    print(e)


def get_dataset_class(name):
    registry = {
        'detection': ObjectDetectionDataset
    }

    return registry[str(name).lower()]

__all__ = [
    'ObjectDetectionDataset',
    'get_dataset_class',
]
