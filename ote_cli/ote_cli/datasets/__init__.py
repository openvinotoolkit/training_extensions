try:
    from .object_detection.dataset import ObjectDetectionDataset
except ImportError:
    ObjectDetectionDataset = None

try:
    from torchreid.integration.sc.utils import ClassificationDatasetAdapter

    ClassificationDataset = ClassificationDatasetAdapter
except ImportError:
    ClassificationDataset = None

try:
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
