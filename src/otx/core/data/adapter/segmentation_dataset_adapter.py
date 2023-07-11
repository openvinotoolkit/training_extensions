"""Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tqdm
from datumaro.components.annotation import AnnotationType as DatumAnnotationType
from datumaro.components.annotation import Mask
from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.plugins.data_formats.common_semantic_segmentation import (
    CommonSemanticSegmentationBase,
    make_categories,
)
from datumaro.plugins.transforms import MasksToPolygons
from datumaro.util.meta_file_util import parse_meta_file
from skimage.segmentation import felzenszwalb

from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.subset import Subset
from otx.core.data.adapter.base_dataset_adapter import BaseDatasetAdapter

# pylint: disable=invalid-name, too-many-locals, no-member, too-many-nested-blocks, too-many-branches,
# pylint: too-many-arguments


class SegmentationDatasetAdapter(BaseDatasetAdapter):
    """Segmentation adapter inherited from BaseDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for semantic segmentation task
    """

    def get_otx_dataset(self) -> DatasetEntity:
        """Convert DatumaroDataset to DatasetEntity for Segmentation."""
        # Prepare label information
        label_information = self._prepare_label_information(self.dataset)
        self.label_entities = label_information["label_entities"]

        dataset_items: List[DatasetItemEntity] = []
        used_labels: List[int] = []
        self.updated_label_id: Dict[int, int] = {}

        if hasattr(self, "data_type_candidates"):
            if "voc" in self.data_type_candidates[0]:
                self.set_voc_labels()
            elif self.data_type_candidates[0] == "common_semantic_segmentation":
                self.set_common_labels()

        else:
            # For datasets used for self-sl.
            # They are not included in any data type and `data_type_candidates` is not set,
            # so they must be handled independently. But, setting `self.updated_label_id` is compatible
            # with "common_semantic_segmentation", so we can use it.
            self.set_common_labels()

        for subset, subset_data in self.dataset.items():
            for _, datumaro_items in subset_data.subsets().items():
                for datumaro_item in datumaro_items:
                    image = self.datum_media_2_otx_media(datumaro_item.media)
                    assert isinstance(image, Image)
                    shapes: List[Annotation] = []
                    for ann in datumaro_item.annotations:
                        if ann.type == DatumAnnotationType.mask:
                            # TODO: consider case -> didn't include the background information
                            datumaro_polygons = MasksToPolygons.convert_mask(ann)
                            for d_polygon in datumaro_polygons:
                                new_label = self.updated_label_id.get(d_polygon.label, None)
                                if new_label is not None:
                                    d_polygon.label = new_label
                                else:
                                    continue

                                shapes.append(self._get_polygon_entity(d_polygon, image.width, image.height))
                                if d_polygon.label not in used_labels:
                                    used_labels.append(d_polygon.label)

                    if len(shapes) > 0 or subset == Subset.UNLABELED:
                        dataset_item = DatasetItemEntity(image, self._get_ann_scene_entity(shapes), subset=subset)
                        dataset_items.append(dataset_item)

        self.remove_unused_label_entities(used_labels)
        return DatasetEntity(items=dataset_items)

    def set_voc_labels(self):
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background & ignored label in VOC from datumaro
        self._remove_labels(["background", "ignored"])

    def set_common_labels(self):
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background if in label_entities
        is_removed = self._remove_labels(["background"])

        # Shift label id since datumaro always extracts bg polygon with label 0
        if is_removed is False:
            self.updated_label_id = {k + 1: v for k, v in self.updated_label_id.items()}

    def _remove_labels(self, label_names: List):
        """Remove background label in label entity set."""
        is_removed = False
        new_label_entities = []
        for i, entity in enumerate(self.label_entities):
            if entity.name not in label_names:
                new_label_entities.append(entity)
            else:
                is_removed = True

        self.label_entities = new_label_entities

        for i, entity in enumerate(self.label_entities):
            self.updated_label_id[int(entity.id)] = i
            entity.id = ID(i)

        return is_removed


class SelfSLSegmentationDatasetAdapter(SegmentationDatasetAdapter):
    """Self-SL for segmentation adapter inherited from SegmentationDatasetAdapter."""

    # pylint: disable=protected-access
    def _import_datasets(
        self,
        train_data_roots: Optional[str] = None,
        train_ann_files: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        val_ann_files: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        test_ann_files: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
        unlabeled_file_list: Optional[str] = None,
        encryption_key: Optional[str] = None,
        pseudo_mask_dir: Path = None,
    ) -> Dict[Subset, DatumDataset]:
        """Import custom Self-SL dataset for using DetCon.

        Self-SL for semantic segmentation using DetCon uses pseudo masks as labels,
        but Datumaro cannot load this custom data structure because it is not in Datumaro format.
        So, it is required to manually load and set annotations.

        Args:
            train_data_roots (Optional[str]): Path for training data.
            train_ann_files (Optional[str]): Path for training annotation file
            val_data_roots (Optional[str]): Path for validation data
            val_ann_files (Optional[str]): Path for validation annotation file
            test_data_roots (Optional[str]): Path for test data.
            test_ann_files (Optional[str]): Path for test annotation file
            unlabeled_data_roots (Optional[str]): Path for unlabeled data.
            unlabeled_file_list (Optional[str]): Path of unlabeled file list
            encryption_key (Optional[str]): Encryption key to load an encrypted dataset
                                        (only required for DatumaroBinary format)
            pseudo_mask_dir (Path): Directory to save pseudo masks. Defaults to None.

        Returns:
            DatumaroDataset: Datumaro Dataset
        """
        if pseudo_mask_dir is None:
            raise ValueError("pseudo_mask_dir must be set.")
        if train_data_roots is None:
            raise ValueError("train_data_root must be set.")

        logger = get_logger()
        logger.warning(f"Please check if {train_data_roots} is data roots only for images, not annotations.")
        dataset = {}
        dataset[Subset.TRAINING] = DatumDataset.import_from(train_data_roots, format="image_dir")
        self.is_train_phase = True

        # Load pseudo masks
        total_labels = []
        os.makedirs(pseudo_mask_dir, exist_ok=True)
        print("[*] Generating pseudo masks for DetCon algorithm. It can take some time...")
        for item in tqdm.tqdm(dataset[Subset.TRAINING]):
            img_path = item.media.path
            pseudo_mask_path = pseudo_mask_dir / os.path.basename(img_path)
            if pseudo_mask_path.suffix == ".jpg":
                pseudo_mask_path = pseudo_mask_path.with_name(f"{pseudo_mask_path.stem}.png")

            if not os.path.isfile(pseudo_mask_path):
                # Create pseudo mask
                pseudo_mask = self.create_pseudo_masks(item.media.data, str(pseudo_mask_path))  # type: ignore
            else:
                # Load created pseudo mask
                pseudo_mask = cv2.imread(str(pseudo_mask_path), cv2.IMREAD_GRAYSCALE)

            # Set annotations into each item
            annotations = []
            labels = np.unique(pseudo_mask)
            for label_id in labels:
                if label_id not in total_labels:
                    # Stack label_id to save dataset_meta.json
                    total_labels.append(label_id)
                annotations.append(
                    Mask(image=CommonSemanticSegmentationBase._lazy_extract_mask(pseudo_mask, label_id), label=label_id)
                )
            item.annotations = annotations

        if not os.path.isfile(os.path.join(pseudo_mask_dir, "dataset_meta.json")):
            # Save dataset_meta.json for newly created pseudo masks
            # FIXME: Because background class is ignored when generating polygons, meta is set with len(labels)-1.
            # It must be considered to set the whole labels later.
            # (-> {i: f"target{i+1}" for i in range(max(total_labels)+1)})
            meta = {"label_map": {i + 1: f"target{i+1}" for i in range(max(total_labels))}}
            with open(os.path.join(pseudo_mask_dir, "dataset_meta.json"), "w", encoding="UTF-8") as f:
                json.dump(meta, f, indent=4)

        # Make categories for pseudo masks
        label_map = parse_meta_file(os.path.join(pseudo_mask_dir, "dataset_meta.json"))
        dataset[Subset.TRAINING].define_categories(make_categories(label_map))

        return dataset

    def create_pseudo_masks(self, img: np.ndarray, pseudo_mask_path: str, mode: str = "FH") -> None:
        """Create pseudo masks for self-sl for semantic segmentation using DetCon.

        Args:
            img (np.ndarray) : A sample to create a pseudo mask.
            pseudo_mask_path (Path): The path to save a pseudo mask.
            mode (str): The mode to create a pseudo mask. Defaults to "FH".

        Returns:
            np.array: a created pseudo mask for item.
        """
        if mode == "FH":
            pseudo_mask = felzenszwalb(img, scale=1000, min_size=1000)
        else:
            raise ValueError((f'{mode} is not supported to create pseudo masks for DetCon. Choose one of ["FH"].'))

        cv2.imwrite(pseudo_mask_path, pseudo_mask.astype(np.uint8))

        return pseudo_mask
