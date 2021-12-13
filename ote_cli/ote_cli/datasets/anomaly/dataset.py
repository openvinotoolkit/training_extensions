"""DataLoaders for Anomaly Tasks."""
# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset


class AnomalyClassificationDataset(DatasetEntity):
    """Dataloader for Anomaly Classification Task.

    Example:
        >>> train_subset = {
                "ann_file": "data/anomaly/train.json",
                "data_root": "data/anomaly/shapes",
            }
        >>> val_subset = {"ann_file": "data/anomaly/val.json", "data_root": "data/anomaly/shapes"}
        >>> training_dataset = AnomalyClassificationDataset(
                train_subset=train_subset, val_subset=val_subset
            )
        >>> test_subset = {"ann_file": "data/anomaly/test.json", "data_root": "data/anomaly/shapes"}
        >>> testing_dataset = AnomalyClassificationDataset(test_subset=test_subset)

    Args:
        train_subset (Optional[Dict[str, str]], optional): Path to annotation
            and dataset used for training. Defaults to None.
        val_subset (Optional[Dict[str, str]], optional): Path to annotation
            and dataset used for validation. Defaults to None.
        test_subset (Optional[Dict[str, str]], optional): Path to annotation
            and dataset used for testing. Defaults to None.
    """

    def __init__(
        self,
        train_subset: Optional[Dict[str, str]] = None,
        val_subset: Optional[Dict[str, str]] = None,
        test_subset: Optional[Dict[str, str]] = None,
    ):

        items: List[DatasetItemEntity] = []
        self.normal_label = LabelEntity(
            name="Normal", domain=Domain.ANOMALY_CLASSIFICATION
        )
        self.abnormal_label = LabelEntity(
            name="Anomalous", domain=Domain.ANOMALY_CLASSIFICATION
        )

        if train_subset is not None:
            train_ann_file = Path(train_subset["ann_file"])
            train_data_root = Path(train_subset["data_root"])
            items.extend(
                self.get_dataset_items(
                    ann_file_path=train_ann_file,
                    data_root_dir=train_data_root,
                    subset=Subset.TRAINING,
                )
            )

        if val_subset is not None:
            val_ann_file = Path(val_subset["ann_file"])
            val_data_root = Path(val_subset["data_root"])
            items.extend(
                self.get_dataset_items(
                    ann_file_path=val_ann_file,
                    data_root_dir=val_data_root,
                    subset=Subset.VALIDATION,
                )
            )

        if test_subset is not None:
            test_ann_file = Path(test_subset["ann_file"])
            test_data_root = Path(test_subset["data_root"])
            items.extend(
                self.get_dataset_items(
                    ann_file_path=test_ann_file,
                    data_root_dir=test_data_root,
                    subset=Subset.TESTING,
                )
            )

        super().__init__(items=items)

    def get_dataset_items(
        self, ann_file_path: Path, data_root_dir: Path, subset: Subset
    ) -> List[DatasetItemEntity]:
        """Loads dataset based on the image path in annotation file.

        Args:
            ann_file_path (Path): Path to json containing the annotations.
                For example of annotation look at `data/anomaly/[train, test,val].json.
            data_root_dir (Path): Path to folder containing images.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        # read annotation file
        samples = pd.read_json(ann_file_path)

        dataset_items = []
        for _, sample in samples.iterrows():
            # Create image
            # convert path to str as PosixPath is not supported by Image
            image = Image(file_path=str(data_root_dir / sample.image_path))
            # Create annotation
            shape = Rectangle(x1=0, y1=0, x2=1, y2=1)
            label: LabelEntity = (
                self.normal_label if sample.label == "good" else self.abnormal_label
            )
            labels = [ScoredLabel(label)]
            annotations = [Annotation(shape=shape, labels=labels)]
            annotation_scene = AnnotationSceneEntity(
                annotations=annotations, kind=AnnotationSceneKind.ANNOTATION
            )

            # Create dataset item
            dataset_item = DatasetItemEntity(
                media=image, annotation_scene=annotation_scene, subset=subset
            )
            # Add to dataset items
            dataset_items.append(dataset_item)

        return dataset_items
