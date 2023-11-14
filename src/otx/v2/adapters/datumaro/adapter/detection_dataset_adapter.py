"""Detection Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List

from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.components.annotation import AnnotationType as DatumAnnotationType

from otx.v2.api.entities.dataset_item import DatasetItemEntityWithID
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType

from .datumaro_dataset_adapter import DatumaroDatasetAdapter


class DetectionDatasetAdapter(DatumaroDatasetAdapter):
    """Detection adapter inherited from DatumaroDatasetAdapter.

    It converts annotataion format of DatumDataset for object detection, and instance segmentation tasks
    """

    def get_otx_dataset(self) -> Dict[Subset, DatumDataset]:
        """Convert DatumaroDataset's annotation for Detection."""
        # Prepare label information
        used_labels: List[int] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    converted_annotations = []
                    for annotation in datumaro_item.annotations:
                        if (
                            self.task_type in (TaskType.INSTANCE_SEGMENTATION, TaskType.ROTATED_DETECTION)
                            and annotation.type == DatumAnnotationType.polygon
                        ) and self._is_normal_polygon(annotation):
                            converted_annotations.append(annotation)
                        if self.task_type is TaskType.DETECTION and annotation.type == DatumAnnotationType.bbox:
                            if self._is_normal_bbox(
                                annotation.points[0],
                                annotation.points[1],
                                annotation.points[2],
                                annotation.points[3],
                            ):
                                converted_annotations.append(annotation)

                        if annotation.label not in used_labels:
                            used_labels.append(annotation.label)
                        datumaro_item.annotations = converted_annotations

        self.remove_unused_label_entities(used_labels)
        return self.dataset
