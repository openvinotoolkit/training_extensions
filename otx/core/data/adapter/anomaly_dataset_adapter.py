"""Anomaly Classification / Detection / Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=invalid-name, too-many-locals, no-member
from typing import Any, Dict, List, Union

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset as DatumaroDataset
from datumaro.plugins.transforms import MasksToPolygons

from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import LabelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter


class AnomalyDatasetAdapter(BaseDatasetAdapter):
    """BaseDataset Adpater for Anomaly tasks inherited from BaseDatasetAdapter."""

    def _select_data_type(self, data_candidates: Union[List[str], str]) -> str:
        """Select specific type among candidates.

        Args:
            data_candidates (Union[List[str], str]): Type candidates made by Datumaro.Environment().detect_dataset()

        Returns:
            str: Selected data type
        """
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            assert "mvtec_classification" in data_candidates
            data_type = "mvtec_classification"
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            assert "mvtec_segmentation" in data_candidates
            data_type = "mvtec_segmentation"
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            assert "mvtec_detection" in data_candidates
            data_type = "mvtec_detection"
        else:
            raise ValueError(f"Unrecognized anomaly task type: {self.task_type}")
        return data_type

    def _prepare_label_information(
        self,
        datumaro_dataset: Dict[Subset, DatumaroDataset],
    ) -> Dict[str, Any]:
        """Prepare LabelEntity List."""
        normal_label = LabelEntity(id=ID(0), name="Normal", domain=self.domain)
        abnormal_label = LabelEntity(
            id=ID(1),
            name="Anomalous",
            domain=self.domain,
            is_anomalous=True,
        )

        label_information = super()._prepare_label_information(datumaro_dataset)
        label_information["label_entities"] = [
            normal_label if class_name.name == "good" else abnormal_label
            for class_name in label_information["category_items"]
        ]

        return label_information

    def _get_shapes_from_datumaro_item(self, datumaro_item):
        height, width = datumaro_item.media.size
        shapes: List[Annotation] = []
        for ann in datumaro_item.annotations:
            if ann.type == AnnotationType.label:
                shapes.append(self._get_label_entity(ann))
            elif ann.type == AnnotationType.mask:
                datumaro_polygons = MasksToPolygons.convert_mask(ann)
                for d_polygon in datumaro_polygons:
                    shapes.append(self._get_polygon_entity(d_polygon, width, height))
            elif ann.type == AnnotationType.bbox:
                if self._is_normal_bbox(ann.points[0], ann.points[1], ann.points[2], ann.points[3]):
                    shapes.append(self._get_normalized_bbox_entity(ann, width, height))
        return shapes

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Anomaly tasks."""
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        # Convert dataset items
        dataset_items: List[DatasetItemEntity] = []
        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:

                    shapes = self._get_shapes_from_datumaro_item(datumaro_item)

                    image = Image(file_path=datumaro_item.media.path)
                    dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                    dataset_items.append(dataset_item)

        self.remove_duplicate_label_entities()
        return DatasetEntity(items=dataset_items)

    def remove_duplicate_label_entities(self):
        """Remove duplicate labels from label entities.

        Because label entities will be used to make Label Schema,
        If there are duplicate labels in the Label Schema, it will hurt the model performance.
        """
        self.label_entities = list(set(self.label_entities))
