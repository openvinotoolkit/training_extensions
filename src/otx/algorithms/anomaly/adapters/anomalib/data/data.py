"""Anomaly Dataset Utils."""

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

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from anomalib.data.base.datamodule import collate_fn
from anomalib.data.utils.transform import get_transforms
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.shapes.polygon import Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.dataset_utils import (
    contains_anomalous_images,
    split_local_global_dataset,
)
from otx.api.utils.segmentation_utils import mask_from_dataset_item
from otx.utils.logger import get_logger

logger = get_logger()


class OTXAnomalyDataset(Dataset):
    """Anomaly Dataset Adaptor.

    This class converts OTX Dataset into Anomalib dataset that
    is a sub-class of Vision Dataset.

    Args:
        config (Union[DictConfig, ListConfig]): Anomalib config
        dataset (DatasetEntity): [description]: OTX SDK Dataset

    Example:
        >>> from tests.helpers.dataset import OTXAnomalyDatasetGenerator
        >>> from otx.utils.data import AnomalyDataset

        >>> dataset_generator = OTXAnomalyDatasetGenerator()
        >>> dataset = dataset_generator.generate()
        >>> anomaly_dataset = AnomalyDataset(config=config, dataset=dataset)
        >>> anomaly_dataset[0]["image"].shape
        torch.Size([3, 256, 256])
    """

    def __init__(self, config: Union[DictConfig, ListConfig], dataset: DatasetEntity, task_type: TaskType):
        self.config = config
        self.dataset = dataset
        self.task_type = task_type

        # TODO: distinguish between train and val config here
        self.transform = get_transforms(
            config=config.dataset.transform_config.train, image_size=tuple(config.dataset.image_size), to_tensor=True
        )

    def __len__(self) -> int:
        """Get size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Union[int, Tensor]]:
        """Get dataset item.

        Args:
            index (int): Index of the dataset sample.

        Raises:
            ValueError: When the task type is not supported.

        Returns:
            Dict[str, Union[int, Tensor]]: Dataset item.
        """
        dataset_item = self.dataset[index]
        item: Dict[str, Union[int, Tensor]] = {}
        item = {"index": index}
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            # Detection currently relies on image labels only, meaning it'll use image
            #   threshold to find the predicted bounding boxes.
            item["image"] = self.transform(image=dataset_item.numpy)["image"]
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            item["image"] = self.transform(image=dataset_item.numpy)["image"]
            item["boxes"] = torch.empty((0, 4))
            height, width = self.config.dataset.image_size
            boxes = []
            for annotation in dataset_item.get_annotations():
                if isinstance(annotation.shape, Rectangle) and not Rectangle.is_full_box(annotation.shape):
                    boxes.append(
                        Tensor(
                            [
                                annotation.shape.x1 * width,
                                annotation.shape.y1 * height,
                                annotation.shape.x2 * width,
                                annotation.shape.y2 * height,
                            ]
                        )
                    )
                if boxes:
                    item["boxes"] = torch.stack(boxes)
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            if any((isinstance(annotation.shape, Polygon) for annotation in dataset_item.get_annotations())):
                mask = mask_from_dataset_item(dataset_item, dataset_item.get_shapes_labels()).squeeze()
            else:
                mask = np.zeros(dataset_item.numpy.shape[:2]).astype(np.int)
            pre_processed = self.transform(image=dataset_item.numpy, mask=mask)
            item["image"] = pre_processed["image"]
            item["mask"] = pre_processed["mask"]
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        if len(dataset_item.get_shapes_labels()) > 0 and dataset_item.get_shapes_labels()[0].is_anomalous:
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

    def __init__(self, config: Union[DictConfig, ListConfig], dataset: DatasetEntity, task_type: TaskType) -> None:
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.task_type = task_type

        self.train_otx_dataset: DatasetEntity
        self.val_otx_dataset: DatasetEntity
        self.test_otx_dataset: DatasetEntity
        self.predict_otx_dataset: DatasetEntity

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup Anomaly Data Module.

        Args:
            stage (Optional[str], optional): train/val/test stages.
                Defaults to None.
        """
        if not stage == "predict":
            self.summary()

        if stage == "fit" or stage is None:
            self.train_otx_dataset = self.dataset.get_subset(Subset.TRAINING)
            self.val_otx_dataset = self.dataset.get_subset(Subset.VALIDATION)

        if stage == "validate":
            self.val_otx_dataset = self.dataset.get_subset(Subset.VALIDATION)

        if stage == "test" or stage is None:
            self.test_otx_dataset = self.dataset.get_subset(Subset.TESTING)

        if stage == "predict":
            self.predict_otx_dataset = self.dataset

    def summary(self):
        """Print size of the dataset, number of anomalous images and number of normal images."""
        for subset in [Subset.TRAINING, Subset.VALIDATION, Subset.TESTING]:
            dataset = self.dataset.get_subset(subset)
            num_items = len(dataset)
            num_normal = len([item for item in dataset if not item.get_shapes_labels()[0].is_anomalous])
            num_anomalous = len([item for item in dataset if item.get_shapes_labels()[0].is_anomalous])
            logger.info(
                "'%s' subset size: Total '%d' images. Normal: '%d', images. Anomalous: '%d' images",
                subset,
                num_items,
                num_normal,
                num_anomalous,
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
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

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Validation Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader]]: Validation Dataloader.
        """
        global_dataset, local_dataset = split_local_global_dataset(self.val_otx_dataset)
        logger.info(f"Global annotations: {len(global_dataset)}")
        logger.info(f"Local annotations: {len(local_dataset)}")
        if contains_anomalous_images(local_dataset):
            logger.info("Dataset contains polygon annotations. Passing masks to anomalib.")
            dataset = OTXAnomalyDataset(self.config, local_dataset, self.task_type)
        else:
            logger.info("Dataset does not contain polygon annotations. Not passing masks to anomalib.")
            dataset = OTXAnomalyDataset(self.config, global_dataset, TaskType.ANOMALY_CLASSIFICATION)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.eval_batch_size,
            num_workers=self.config.dataset.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
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

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
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
