"""Visual Prompting Dataset Adapter."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional

from datumaro.components.annotation import AnnotationType as DatumAnnotationType
from datumaro.plugins.transforms import MasksToPolygons

from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.task_type import TaskType

from .segmentation_dataset_adapter import SegmentationDatasetAdapter


class VisualPromptingDatasetAdapter(SegmentationDatasetAdapter):
    """Visual prompting adapter inherited from SegmentationDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for visual prompting tasks.
    To handle masks, this adapter is inherited from SegmentationDatasetAdapter.
    """

    def __init__(
        self,
        task_type: TaskType,
        train_data_roots: Optional[str] = None,
        train_ann_files: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        val_ann_files: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        test_ann_files: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
        unlabeled_file_list: Optional[str] = None,
        cache_config: Optional[dict] = None,
        encryption_key: Optional[str] = None,
        use_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            task_type,
            train_data_roots,
            train_ann_files,
            val_data_roots,
            val_ann_files,
            test_data_roots,
            test_ann_files,
            unlabeled_data_roots,
            unlabeled_file_list,
            cache_config,
            encryption_key,
            use_mask,
            **kwargs,
        )

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Visual Prompting."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items: List[DatasetItemEntity] = []
        used_labels: List[int] = []
        self.updated_label_id: Dict[int, int] = {}

        if hasattr(self, "data_type_candidates"):
            if self.data_type == "voc":
                self.set_voc_labels()

            if self.data_type == "common_semantic_segmentation":
                self.set_common_labels()

        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = self.datum_media_2_otx_media(datumaro_item.media)
                    shapes = []
                    for ann in datumaro_item.annotations:
                        if ann.type == DatumAnnotationType.polygon:
                            # save polygons as-is, they will be converted to masks.
                            if self._is_normal_polygon(ann):
                                shapes.append(self._get_polygon_entity(ann, image.width, image.height))

                        if ann.type == DatumAnnotationType.mask:
                            if self.use_mask:
                                # use masks loaded in datumaro as-is
                                if self.data_type == "common_semantic_segmentation":
                                    if new_label := self.updated_label_id.get(ann.label, None):
                                        ann.label = new_label
                                    else:
                                        continue
                                shapes.append(self._get_mask_entity(ann))

                            else:
                                # convert masks to polygons, they will be converted to masks again
                                datumaro_polygons = MasksToPolygons.convert_mask(ann)
                                for d_polygon in datumaro_polygons:
                                    if new_label := self.updated_label_id.get(d_polygon.label, None):
                                        d_polygon.label = new_label
                                    else:
                                        continue

                                    shapes.append(self._get_polygon_entity(d_polygon, image.width, image.height))
                                    if d_polygon.label not in used_labels:
                                        used_labels.append(d_polygon.label)

                        if ann.label not in used_labels and ann.type != DatumAnnotationType.mask:
                            used_labels.append(ann.label)

                    if len(shapes) > 0:
                        dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                        dataset_items.append(dataset_item)
        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)
