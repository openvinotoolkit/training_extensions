"""Anomaly Dataset Utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from anomalib.data.base.datamodule import collate_fn
from anomalib.data.utils.transform import get_transforms
from datumaro.components.annotation import Bbox, Mask
from datumaro.components.annotation import Bbox as DatumBbox
from datumaro.components.annotation import Label as DatumLabel
from datumaro.components.annotation import Polygon as DatumPolygon
from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.components.dataset_base import DatasetItem as DatumDatasetItem
from datumaro.components.media import Image as DatumImage
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from otx.v2.api.entities.id import ID
from otx.v2.api.entities.label import Domain, LabelEntity
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType
from otx.v2.api.utils.logger import get_logger

if TYPE_CHECKING:
    from omegaconf import DictConfig, ListConfig



logger = get_logger()


class OTXAnomalyDataset(Dataset):
    """Anomaly Dataset Adaptor.

    This class converts OTX Dataset into Anomalib dataset that
    is a sub-class of Vision Dataset.

    Args:
        config (Union[DictConfig, ListConfig]): Anomalib config
        dataset (dict[Subset, DatumDataset]): Datumaro Dataset

    Example:
        >>> from tests.helpers.dataset import OTXAnomalyDatasetGenerator
        >>> from otx.utils.data import AnomalyDataset

        >>> dataset_generator = OTXAnomalyDatasetGenerator()
        >>> dataset = dataset_generator.generate()
        >>> anomaly_dataset = AnomalyDataset(config=config, dataset=dataset)
        >>> anomaly_dataset[0]["image"].shape
        torch.Size([3, 256, 256])
    """

    def __init__(self, config: DictConfig | ListConfig, dataset: DatumDataset, task_type: TaskType) -> None:
        """Initializes a new instance of the Data class.

        Args:
            config (Union[DictConfig, ListConfig]): The configuration for the data.
            dataset (dict[Subset, DatumDataset]): The dataset to use for the data.
            task_type (TaskType): The type of task to perform on the data.
        """
        self.config = config
        self.dataset = dataset
        self.task_type = task_type

        self.transform = get_transforms(
            config=config.dataset.transform_config.train,
            image_size=tuple(config.dataset.image_size),
            to_tensor=True,
        )

        self.item_ids: list[str] = [item.id for item in self.dataset]

    def __len__(self) -> int:
        """Get size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, int | Tensor]:
        """Get dataset item.

        Args:
            index (int): Index of the dataset sample.

        Raises:
            ValueError: When the task type is not supported.

        Returns:
            Dict[str, Union[int, Tensor]]: Dataset item.
        """
        dataset_item = self.dataset.get(id=self.item_ids[index])
        item: dict[str, int | Tensor] = {}
        item = {"index": index}
        image = dataset_item.media.data
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            # Detection currently relies on image labels only, meaning it'll use image
            #   threshold to find the predicted bounding boxes.
            item["image"] = self.transform(image=image)["image"]
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            item["image"] = self.transform(image=image)["image"]
            item["boxes"] = torch.empty((0, 4))
            height, width = self.config.dataset.image_size
            boxes = []
            for annotation in dataset_item.annotations:
                if isinstance(annotation.type, Bbox):
                    boxes.append(
                        Tensor(
                            [
                                annotation.x * width,
                                annotation.y * height,
                                (annotation.x + annotation.w) * width,
                                (annotation.y + annotation.h) * height,
                            ],
                        ),
                    )
                if boxes:
                    item["boxes"] = torch.stack(boxes)
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            for annotation in dataset_item.annotations:
                mask = annotation.image if isinstance(annotation.type, Mask) else np.zeros(image.shape[:2]).astype(int)
            pre_processed = self.transform(image=image, mask=mask)
            item["image"] = pre_processed["image"]
            item["mask"] = pre_processed["mask"]
        else:
            msg = f"Unsupported task type: {self.task_type}"
            raise ValueError(msg)

        if len(dataset_item.annotations) > 0 and dataset_item.annotations[0].attributes.get("is_anomalous"):
            item["label"] = 1
        else:
            item["label"] = 0
        return item


class OTXAnomalyDataModule(LightningDataModule):
    """Anomaly DataModule.

    This class converts OTX Dataset into Anomalib dataset and stores
    train/val/test dataloaders.

    Args:
        config (Union[DictConfig, ListConfig]): Anomalib config
        dataset (DatasetEntity): OTX SDK Dataset

    Example:
        >>> from tests.helpers.dataset import OTXAnomalyDatasetGenerator
        >>> from otx.utils.data import AnomalyDataModule

        >>> dataset_generator = OTXAnomalyDatasetGenerator()
        >>> dataset = dataset_generator.generate()
        >>> data_module = OTXAnomalyDataModule(config=config, dataset=dataset)
        >>> i, data = next(enumerate(data_module.train_dataloader()))
        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])
    """

    def __init__(
            self,
            config: DictConfig | ListConfig,
            dataset: dict[Subset, DatumDataset],
            task_type: TaskType) -> None:
        """Initializes a DataModule instance.

        Args:
            config (Union[DictConfig, ListConfig]): The configuration for the DataModule.
            dataset (DatasetEntity): The dataset to use for training, validation, testing, and prediction.
            task_type (TaskType): The type of task to perform (e.g. classification, regression, etc.).
        """
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.task_type = task_type

        self.train_otx_dataset: DatumDataset
        self.val_otx_dataset: DatumDataset
        self.test_otx_dataset: DatumDataset
        self.predict_otx_dataset: DatumDataset

    def setup(self, stage: str | None = None) -> None:
        """Setup Anomaly Data Module.

        Args:
            stage (Optional[str], optional): train/val/test stages.
                Defaults to None.
        """
        if stage != "predict":
            self.summary()

        if stage == "fit" or stage is None:
            self.train_otx_dataset = self.dataset.get(Subset.TRAINING)
            self.val_otx_dataset = self.dataset.get(Subset.VALIDATION)

        if stage == "validate":
            self.val_otx_dataset = self.dataset.get(Subset.VALIDATION)

        if stage == "test" or stage is None:
            self.test_otx_dataset = self.dataset.get(Subset.TESTING)

        if stage == "predict":
            self.predict_otx_dataset = self.dataset.get(Subset.TESTING)

    def summary(self) -> None:
        """Print size of the dataset, number of anomalous images and number of normal images."""
        for subset in [Subset.TRAINING, Subset.VALIDATION, Subset.TESTING]:
            dataset: DatumDataset = self.dataset.get(subset)
            num_items = len(dataset)
            num_normal = len([item for item in dataset if not item.attributes.get("is_anomalous")])
            num_anomalous = len([item for item in dataset if item.attributes.get("is_anomalous")])
            logger.info(
                "'%s' subset size: Total '%d' images. Normal: '%d', images. Anomalous: '%d' images",
                subset,
                num_items,
                num_normal,
                num_anomalous,
            )

    def train_dataloader(
        self,
    ) -> DataLoader | (list[DataLoader] | dict[str, DataLoader]):
        """Train Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]: Train dataloader.
        """
        dataset = OTXAnomalyDataset(self.config, self.train_otx_dataset, self.task_type)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.train_batch_size,
            num_workers=self.config.dataset.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Validation Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader]]: Validation Dataloader.
        """
        # [TODO] restore split_local_global_dataset and contains_anomalous_images logic within DatumDataset
        dataset = OTXAnomalyDataset(self.config, self.val_otx_dataset, self.task_type)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.eval_batch_size,
            num_workers=self.config.dataset.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """Test Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader]]: Test Dataloader.
        """
        dataset = OTXAnomalyDataset(self.config, self.test_otx_dataset, self.task_type)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.test_batch_size,
            num_workers=self.config.dataset.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> DataLoader | list[DataLoader]:
        """Predict Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader]]: Predict Dataloader.
        """
        dataset = OTXAnomalyDataset(self.config, self.predict_otx_dataset, self.task_type)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.eval_batch_size,
            num_workers=self.config.dataset.num_workers,
            collate_fn=collate_fn,
        )


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
