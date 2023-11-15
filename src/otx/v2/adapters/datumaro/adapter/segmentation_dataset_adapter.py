"""Segmentation Dataset Adapter."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import tqdm
from datumaro.components.annotation import AnnotationType as DatumAnnotationType
from datumaro.components.annotation import Mask, Annotation
from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.plugins.data_formats.common_semantic_segmentation import (
    CommonSemanticSegmentationBase,
    make_categories,
)
from datumaro.util.meta_file_util import parse_meta_file
from otx.v2.api.entities.label_schema import LabelSchemaEntity
from skimage.segmentation import felzenszwalb

from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils.logger import get_logger

from .datumaro_dataset_adapter import DatumaroDatasetAdapter


class SegmentationDatasetAdapter(DatumaroDatasetAdapter):
    """Segmentation adapter inherited from DatumaroDatasetAdapter.

    It converts DatumaroDataset --> DatasetEntity for semantic segmentation task
    """

    def __init__(
        self,
        task_type: TaskType,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        cache_config: dict | None = None,
        encryption_key: str | None = None,
        use_mask: bool = True,
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
        """Convert DatumaroDataset to DatasetEntity for Segmentation."""
        # TODO (Eugene): this part needs refactoring and could be reused for both visual prompting and segmentation
        # CVS-124394

        used_labels: set[int] = set()
        # TODO (Eugene): self.updated_label_id - unnecessary class variable and unclear naming
        # CVS-124394
        self.updated_label_id: dict[int, int] = {}

        if hasattr(self, "data_type_candidates"):
            if self.data_type_candidates[0] == "voc":
                self.set_voc_labels()

            if self.data_type_candidates[0] == "common_semantic_segmentation":
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
                    anno_list: Annotation = []
                    for ann in datumaro_item.annotations:
                        if ann.type == DatumAnnotationType.mask:
                            label_id = self.updated_label_id.get(ann.label, None)
                            if label_id is not None:
                                ann.label = label_id
                                anno_list.append(ann)
                                used_labels.add(ann.label)

                    if len(anno_list) > 0 or subset == Subset.UNLABELED:
                        datumaro_item.annotations = anno_list

        self.remove_unused_label_entities(list(used_labels))
        return self.dataset

    def set_voc_labels(self) -> None:
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background & ignored label in VOC from datumaro
        self._remove_labels(["background", "ignored"])

    def set_common_labels(self) -> None:
        """Set labels for common_semantic_segmentation dataset."""
        # Remove background if in label_entities
        is_removed = self._remove_labels(["background"])

        # Shift label id since datumaro always extracts bg polygon with label 0
        if is_removed is False:
            self.updated_label_id = {k + 1: v for k, v in self.updated_label_id.items()}

    def _remove_labels(self, label_names: list) -> bool:
        """Remove background label in label entity set."""
        is_removed = False
        new_label_entities = []
        for _i, entity in enumerate(self.label_entities):
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

    def _import_datasets(
        self,
        train_data_roots: str | None = None,
        train_ann_files: str | None = None,
        val_data_roots: str | None = None,
        val_ann_files: str | None = None,
        test_data_roots: str | None = None,
        test_ann_files: str | None = None,
        unlabeled_data_roots: str | None = None,
        unlabeled_file_list: str | None = None,
        encryption_key: str | None = None,
        pseudo_mask_dir: Path | None = None,
    ) -> dict[Subset, DatumDataset]:
        """Import custom Self-SL dataset for using DetCon.

        Self-SL for semantic segmentation using DetCon uses pseudo masks as labels,
        but Datumaro cannot load this custom data structure because it is not in Datumaro format.
        So, it is required to manually load and set annotations.

        Args:
            train_data_roots (str | None): Path for training data.
            train_ann_files (str | None): Path for training annotation file
            val_data_roots (str | None): Path for validation data
            val_ann_files (str | None): Path for validation annotation file
            test_data_roots (str | None): Path for test data.
            test_ann_files (str | None): Path for test annotation file
            unlabeled_data_roots (str | None): Path for unlabeled data.
            unlabeled_file_list (str | None): Path of unlabeled file list
            encryption_key (str | None): Encryption key to load an encrypted dataset
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
                pseudo_mask = self.create_pseudo_masks(item.media.data, str(pseudo_mask_path))
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
                    Mask(
                        image=CommonSemanticSegmentationBase._lazy_extract_mask(pseudo_mask, label_id),
                        label=label_id,
                    ),
                )
            item.annotations = annotations

        if not os.path.isfile(os.path.join(pseudo_mask_dir, "dataset_meta.json")):
            # Save dataset_meta.json for newly created pseudo masks
            # It must be considered to set the whole labels later.
            # (-> {i: f"target{i+1}" for i in range(max(total_labels)+1)})
            meta = {"label_map": {i + 1: f"target{i+1}" for i in range(max(total_labels))}}
            with open(os.path.join(pseudo_mask_dir, "dataset_meta.json"), "w", encoding="UTF-8") as f:
                json.dump(meta, f, indent=4)

        # Make categories for pseudo masks
        label_map = parse_meta_file(os.path.join(pseudo_mask_dir, "dataset_meta.json"))
        dataset[Subset.TRAINING].define_categories(make_categories(label_map))

        return dataset

    def create_pseudo_masks(self, img: np.ndarray, pseudo_mask_path: str, mode: str = "FH") -> np.ndarray:
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
            raise ValueError(f'{mode} is not supported to create pseudo masks for DetCon. Choose one of ["FH"].')

        cv2.imwrite(pseudo_mask_path, pseudo_mask.astype(np.uint8))

        return pseudo_mask
