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
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from anomalib.data.base.datamodule import collate_fn
from anomalib.data.utils.transform import get_transforms
from datumaro.components.annotation import Bbox, Mask
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from otx.v2.adapters.torch.lightning.anomalib.modules.logger import get_logger
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DatumDataset
    from omegaconf import DictConfig, ListConfig

logger = get_logger(__name__)


class OTXAnomalyDataset(Dataset):
    """Anomaly Dataset Adaptor.

    This class converts OTX Dataset into Anomalib dataset that
    is a sub-class of Vision Dataset.

    Args:
        config (Union[DictConfig, ListConfig]): Anomalib config
        dataset (Dict[Subset, DatumDataset]): [description]: OTX SDK Dataset

    Example:
        >>> from tests.helpers.dataset import OTXAnomalyDatasetGenerator
        >>> from otx.utils.data import AnomalyDataset

        >>> dataset_generator = OTXAnomalyDatasetGenerator()
        >>> dataset = dataset_generator.generate()
        >>> anomaly_dataset = AnomalyDataset(config=config, dataset=dataset)
        >>> anomaly_dataset[0]["image"].shape
        torch.Size([3, 256, 256])
    """

    def __init__(
            self,
            config: DictConfig | ListConfig,
            dataset: DatumDataset,
            task_type: TaskType,
        ) -> None:
        """Initializes a new instance of the Data class.

        Args:
            config (Union[DictConfig, ListConfig]): The configuration for the data.
            dataset (Dict[Subset, DatumDataset]): The dataset to use for the data.
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

        self.item_ids: list = [item.id for item in self.dataset]

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
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            # Detection currently relies on image labels only, meaning it'll use image
            #   threshold to find the predicted bounding boxes.
            item["image"] = self.transform(image=dataset_item.media.data)["image"]
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            item["image"] = self.transform(image=dataset_item.media.data)["image"]
            item["boxes"] = torch.empty((0, 4))
            height, width = self.config.dataset.image_size
            boxes = []
            for annotation in dataset_item.annotations():
                if isinstance(annotation.type, Bbox):
                    boxes.append(
                        Tensor(
                            [
                                annotation.shape.x1 * width,
                                annotation.shape.y1 * height,
                                annotation.shape.x2 * width,
                                annotation.shape.y2 * height,
                            ],
                        ),
                    )
                if boxes:
                    item["boxes"] = torch.stack(boxes)
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            for annotation in dataset_item.annotations():
                if isinstance(annotation.type, Mask):
                    mask = annotation.image
                else:
                    mask = np.zeros(dataset_item.numpy.shape[:2]).astype(int)
            pre_processed = self.transform(image=dataset_item.numpy, mask=mask)
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
            task_type: TaskType,
        ) -> None:
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
            self.predict_otx_dataset = self.dataset

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
