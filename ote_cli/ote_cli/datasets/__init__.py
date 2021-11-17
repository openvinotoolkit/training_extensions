from typing import Optional, Type

from ote_sdk.ote_sdk.entities.datasets import DatasetEntity

try:
    ObjectDetectionDataset: Optional[Type[DatasetEntity]]
    from .object_detection.dataset import ObjectDetectionDataset
except ImportError:
    ObjectDetectionDataset = None

try:
    ClassificationDataset: Optional[Type[DatasetEntity]]
    from torchreid.integration.sc.utils import ClassificationDatasetAdapter

    ClassificationDataset = ClassificationDatasetAdapter
except ImportError:
    ClassificationDataset = None

try:
    SemanticSegmentationDataset: Optional[Type[DatasetEntity]]
    from .semantic_segmentation.dataset import SemanticSegmentationDataset
except ImportError:
    SemanticSegmentationDataset = None

assert any((ObjectDetectionDataset, ClassificationDataset, SemanticSegmentationDataset))


def get_dataset_class(name):
    registry = {
        "detection": ObjectDetectionDataset,
        "classification": ClassificationDataset,
        "segmentation": SemanticSegmentationDataset,
    }

    return registry[str(name).lower()]


__all__ = [
    "ObjectDetectionDataset",
    "ClassificationDataset",
    "SemanticSegmentationDataset",
    "get_dataset_class",
]
