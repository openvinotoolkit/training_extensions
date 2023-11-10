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
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from datumaro.components.annotation import Bbox as DatumBbox
from datumaro.components.annotation import Label as DatumLabel
from datumaro.components.annotation import Polygon as DatumPolygon
from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.components.dataset_base import DatasetItem as DatumDatasetItem
from datumaro.components.media import Image as DatumImage

from otx.v2.api.entities.id import ID
from otx.v2.api.entities.label import Domain, LabelEntity
from otx.v2.api.entities.subset import Subset


class BaseAnomalyDataset(ABC):
    """Base Dataloader for Anomaly Tasks."""

    def __init__(
        self,
        train_subset: dict[str, str] | None = None,
        val_subset: dict[str, str] | None = None,
        test_subset: dict[str, str] | None = None,
    ) -> None:
        """Base Anomaly Dataset.

        Args:
            train_subset (Optional[Dict[str, str]], optional): Path to annotation
                and dataset used for training. Defaults to None.
            val_subset (Optional[Dict[str, str]], optional): Path to annotation
                and dataset used for validation. Defaults to None.
            test_subset (Optional[Dict[str, str]], optional): Path to annotation
                and dataset used for testing. Defaults to None.
        """
        self.dataset: dict[Subset, DatumDataset] = {}
        self.normal_label = LabelEntity(id=ID(0), name="Normal", domain=Domain.ANOMALY_CLASSIFICATION)
        self.abnormal_label = LabelEntity(
            id=ID(1),
            name="Anomalous",
            domain=Domain.ANOMALY_CLASSIFICATION,
            is_anomalous=True,
        )

        if train_subset is not None:
            train_ann_file = Path(train_subset["ann_file"])
            train_data_root = Path(train_subset["data_root"])
            self.dataset[Subset.TRAINING] = DatumDataset.from_iterable(
                self.get_dataset_items(
                    ann_file_path=train_ann_file,
                    data_root_dir=train_data_root,
                ),
            )

        if val_subset is not None:
            val_ann_file = Path(val_subset["ann_file"])
            val_data_root = Path(val_subset["data_root"])
            self.dataset[Subset.VALIDATION] = DatumDataset.from_iterable(
                self.get_dataset_items(
                    ann_file_path=val_ann_file,
                    data_root_dir=val_data_root,
                ),
            )

        if test_subset is not None:
            test_ann_file = Path(test_subset["ann_file"])
            test_data_root = Path(test_subset["data_root"])
            self.dataset[Subset.TESTING] = DatumDataset.from_iterable(
                self.get_dataset_items(
                    ann_file_path=test_ann_file,
                    data_root_dir=test_data_root,
                ),
            )

    @abstractmethod
    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path) -> list[DatumDatasetItem]:
        """To be implemented ib subclasses."""
        raise NotImplementedError

    def get(self, subset: Subset) -> DatumDataset:
        """To get subset DatumDataset."""
        return self.dataset.get(subset, DatumDataset.from_iterable([]))


class AnomalyClassificationDataset(BaseAnomalyDataset):
    """Dataloader for Anomaly Classification Task.

    Example:
    >>> train_subset = {
            "ann_file": "tests/assets/anomaly/classification/train.json",
            "data_root": "tests/assets/anomaly/hazelnut",
        }
    >>> val_subset = {
            "ann_file": "tests/assets/anomaly/classification/val.json",
            "data_root": "tests/assets/anomaly/hazelnut"
        }
    >>> training_dataset = AnomalyClassificationDataset(
            train_subset=train_subset, val_subset=val_subset
        )
    >>> test_subset = {
            "ann_file": "tests/assets/anomaly/classification/test.json",
            "data_root": "tests/assets/anomaly/hazelnut"
        }
    >>> testing_dataset = AnomalyClassificationDataset(test_subset=test_subset)
    """

    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path) -> list[DatumDatasetItem]:
        """Loads dataset based on the image path in annotation file.

        Args:
            ann_file_path (Path): Path to json containing the annotations.
                For example of annotation look at `tests/assets/anomaly/[train, test,val].json.
            data_root_dir (Path): Path to folder containing images.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatasetItemEntity]: List containing subset dataset.
        """
        # read annotation file
        samples = pd.read_json(ann_file_path)

        dataset_items = []
        for idx, sample in samples.iterrows():
            # Create image
            # convert path to str as PosixPath is not supported by Image
            media = DatumImage.from_file(path=str(data_root_dir / sample.image_path))
            # Create annotation
            label: LabelEntity = self.normal_label if sample.label == "good" else self.abnormal_label
            annotations = [DatumLabel(label=label.id, attributes={"is_anomalous": label.is_anomalous})]

            # Add to dataset items
            dataset_items.append(DatumDatasetItem(id=idx, media=media, annotations=annotations))

        return dataset_items


class AnomalySegmentationDataset(BaseAnomalyDataset):
    """Dataloader for Anomaly Segmentation Task.

    Example:
        >>> train_subset = {
                "ann_file": "tests/assets/anomaly/segmentation/train.json",
                "data_root": "tests/assets/anomaly/hazelnut",
            }
        >>> val_subset = {
                "ann_file": "tests/assets/anomaly/segmentation/val.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> training_dataset = AnomalySegmentationDataset(
                train_subset=train_subset, val_subset=val_subset
            )
        >>> test_subset = {
                "ann_file": "tests/assets/anomaly/segmentation/test.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> testing_dataset = AnomalySegmentationDataset(test_subset=test_subset)

    """

    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path) -> list[DatumDatasetItem]:
        """Loads dataset based on the image path in annotation file.

        Args:
            ann_file_path (Path): Path to json containing the annotations.
                For example of annotation look at `tests/assets/anomaly/[train, test,val].json.
            data_root_dir (Path): Path to folder containing images.
            subset (Subset): Subset of the dataset.

        Returns:
            List[DatumDatasetItem]: List containing subset dataset.
        """
        # read annotation file
        samples = pd.read_json(ann_file_path)

        dataset_items = []
        for idx, sample in samples.iterrows():
            # Create image
            # convert path to str as PosixPath is not supported by Image
            media = DatumImage.from_file(path=str(data_root_dir / sample.image_path))
            # Create annotation
            label: LabelEntity = self.normal_label if sample.label == "good" else self.abnormal_label
            annotations = [DatumLabel(label=label.id, attributes={"is_anomalous": label.is_anomalous})]
            if isinstance(sample.masks, list) and len(sample.masks) > 0:
                for contour in sample.masks:
                    points = [float(val) for pair in contour for val in pair]
                    polygon = DatumPolygon(
                        points=points,
                        label=label.id,
                        attributes={"is_anomalous": label.is_anomalous},
                    )
                    if polygon.get_area() > 0:
                        # Contour is a closed polygon with area > 0
                        annotations.append(polygon)
                    else:
                        # Contour is a closed polygon with area == 0
                        warnings.warn(
                            "The geometry of the segmentation map you are converting "
                            "is not fully supported. Polygons with a area of zero "
                            "will be removed.",
                            UserWarning,
                            stacklevel=2,
                        )

            # Add to dataset items
            dataset_items.append(DatumDatasetItem(id=idx, media=media, annotations=annotations))

        return dataset_items


class AnomalyDetectionDataset(BaseAnomalyDataset):
    """Dataloader for Anomaly Segmentation Task.

    Example:
        >>> train_subset = {
                "ann_file": "tests/assets/anomaly/detection/train.json",
                "data_root": "tests/assets/anomaly/hazelnut",
            }
        >>> val_subset = {
                "ann_file": "tests/assets/anomaly/detection/val.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> training_dataset = AnomalyDetectionDataset(
                train_subset=train_subset, val_subset=val_subset
            )
        >>> test_subset = {
                "ann_file": "tests/assets/anomaly/detection/test.json",
                "data_root": "tests/assets/anomaly/hazelnut"
            }
        >>> testing_dataset = AnomalyDetectionDataset(test_subset=test_subset)

    """

    def get_dataset_items(self, ann_file_path: Path, data_root_dir: Path) -> list[DatumDatasetItem]:
        """Loads dataset based on the image path in annotation file.

        Args:
            ann_file_path (Path): Path to json containing the annotations.
                For example of annotation look at `tests/assets/anomaly/[train, test,val].json.
            data_root_dir (Path): Path to folder containing images.
            subset (Subset): Subset of the dataset.

        Returns:
            list[DatumDatasetItem]: List containing subset dataset.
        """
        # read annotation file
        samples = pd.read_json(ann_file_path)

        dataset_items = []
        for idx, sample in samples.iterrows():
            # Create image
            # convert path to str as PosixPath is not supported by Image
            media = DatumImage.from_file(path=str(data_root_dir / sample.image_path))
            # Create annotation
            label: LabelEntity = self.normal_label if sample.label == "good" else self.abnormal_label
            annotations = [DatumLabel(label=label.id, attributes={"is_anomalous": label.is_anomalous})]
            if isinstance(sample.bboxes, list) and len(sample.bboxes) > 0:
                for bbox in sample.bboxes:
                    box = DatumBbox(
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        label=label.id,
                        attributes={"is_anomalous": label.is_anomalous},
                    )
                    if box.get_area() > 0:
                        # Contour is a closed polygon with area > 0
                        annotations.append(box)
                    else:
                        # Contour is a closed polygon with area == 0
                        warnings.warn(
                            "The geometry of the segmentation map you are converting "
                            "is not fully supported. Polygons with a area of zero "
                            "will be removed.",
                            UserWarning,
                            stacklevel=2,
                        )

            # Add to dataset items
            dataset_items.append(DatumDatasetItem(id=idx, media=media, annotations=annotations))

        return dataset_items
