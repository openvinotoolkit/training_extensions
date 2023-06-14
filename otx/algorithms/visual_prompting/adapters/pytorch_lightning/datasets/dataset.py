"""Visual Prompting Dataset & DataModule."""

# Copyright (C) 2023 Intel Corporation
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

import cv2
import numpy as np
from omegaconf import DictConfig, ListConfig
from PIL import Image, ImageDraw
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.shapes.polygon import Polygon, Point
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.dataset_utils import (
    contains_anomalous_images,
    split_local_global_dataset,
)
from otx.api.utils.segmentation_utils import mask_from_dataset_item
from otx.api.utils.shape_factory import ShapeFactory

from .pipelines import ResizeLongestSide, collate_fn, MultipleInputsCompose, Pad
import torchvision.transforms as transforms

logger = get_logger()


class OTXVIsualPromptingDataset(Dataset):
    """Visual Prompting Dataset Adaptor.
    
    Args:
        config
        dataset
        transform
        stage
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        dataset: DatasetEntity,
        transform: MultipleInputsCompose,
    ) -> None:

        self.config = config
        self.dataset = dataset
        self.transform = transform

        self.labels = dataset.get_labels()
        self.label_idx = {label.id: i for i, label in enumerate(self.labels)}

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

        width, height = dataset_item.width, dataset_item.height
        bboxes: List[List[int]] = []
        points: List = []
        gt_masks: List[np.ndarray] = []
        # TODO (sungchul): load mask from dataset
        for annotation in dataset_item.get_annotations(labels=self.labels, include_empty=False, preserve_id=True):
            if isinstance(annotation.shape, Polygon) and not self.config.use_mask:
                # convert polygon to mask
                polygon = ShapeFactory.shape_as_polygon(annotation.shape)
                contour = np.asarray([[int(point.x * width), int(point.y * height)] for point in polygon.points])
                gt_mask = np.zeros(shape=(height, width), dtype=np.uint8)
                gt_mask = cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)
                gt_masks.append(gt_mask)

                # 
                y_indices, x_indices = np.where(gt_mask == 1)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)

                # add perturbation to bounding box coordinates
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(width, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(height, y_max + np.random.randint(0, 20))
                bboxes.append([x_min, y_min, x_max, y_max])

            if self.config.use_mask:
                # if using masks from dataset, we can use bboxes from dataset, too
                if isinstance(annotation.shape, Rectangle):
                    # use bbox from dataset
                    bbox = ShapeFactory.shape_as_rectangle(annotation.shape)
                    if min(bbox.width * width, bbox.height * height) < -1:
                        continue

                    # inject randomness
                    bbox = [
                        max(0, int(bbox.x1 * width) - np.random.randint(0, 20)),
                        max(0, int(bbox.y1 * height) - np.random.randint(0, 20)),
                        min(width, int(bbox.x2 * width) + np.random.randint(0, 20)),
                        min(height, int(bbox.y2 * height)+ np.random.randint(0, 20))
                    ]
                    bboxes.append(bbox)

                if isinstance(annotation.shape, Point):
                    logger.warn("Using points is not implemented yet.")

        if len(bboxes) == 0:
            # there is no bbox from dataset -> generate bboxes based on gt_masks
            for gt_mask in gt_masks:
                y_indices, x_indices = np.where(gt_mask == 1)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)

                # add perturbation to bounding box coordinates
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(width, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(height, y_max + np.random.randint(0, 20))
                bboxes.append([x_min, y_min, x_max, y_max])

                # TODO (sungchul): generate random points from gt_mask

        gt_masks = np.stack(gt_masks, axis=0)
        bboxes = np.array(bboxes)
        item.update(dict(
            original_size=(height, width),
            images=dataset_item.numpy,
            path=dataset_item.media.path,
            gt_masks=gt_masks,
            bboxes=bboxes,
            points=points, # TODO (sungchul): update point information
        ))
        item = self.transform(item)
        return item


class OTXVisualPromptingDataModule(LightningDataModule):
    """Visual Prompting DataModule."""

    def __init__(self, config: Union[DictConfig, ListConfig], dataset: DatasetEntity) -> None:
        super().__init__()
        self.config = config
        self.dataset = dataset

        self.train_otx_dataset: DatasetEntity
        self.val_otx_dataset: DatasetEntity
        self.test_otx_dataset: DatasetEntity
        self.predict_otx_dataset: DatasetEntity

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup Visual Prompting Data Module.

        Args:
            stage (Optional[str], optional): train/val/test stages.
                Defaults to None.
        """
        if not stage == "predict":
            self.summary()

        image_size = self.config.image_size
        if isinstance(image_size, int):
            image_size = [image_size]

        if stage == "fit" or stage is None:
            train_otx_dataset = self.dataset.get_subset(Subset.TRAINING)
            val_otx_dataset = self.dataset.get_subset(Subset.VALIDATION)

            # TODO (sungchul): distinguish between train and val config here
            train_transform = val_transform = MultipleInputsCompose([
                ResizeLongestSide(target_length=max(image_size)),
                Pad(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            ])

            self.train_dataset = OTXVIsualPromptingDataset(self.config, train_otx_dataset, train_transform)
            self.val_dataset = OTXVIsualPromptingDataset(self.config, val_otx_dataset, val_transform)

        if stage == "test":
            test_otx_dataset = self.dataset.get_subset(Subset.TESTING)
            test_transform = MultipleInputsCompose([
                ResizeLongestSide(target_length=max(image_size)),
                Pad(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            ])
            self.test_dataset = OTXVIsualPromptingDataset(self.config, test_otx_dataset, test_transform)

        if stage == "predict":
            predict_otx_dataset = self.dataset
            predict_transform = MultipleInputsCompose([
                ResizeLongestSide(target_length=max(image_size)),
                Pad(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            ])
            self.predict_dataset = OTXVIsualPromptingDataset(self.config, predict_otx_dataset, predict_transform)

    def summary(self):
        """Print size of the dataset, number of images."""
        for subset in [Subset.TRAINING, Subset.VALIDATION, Subset.TESTING]:
            dataset = self.dataset.get_subset(subset)
            num_items = len(dataset)
            logger.info(
                "'%s' subset size: Total '%d' images.",
                subset,
                num_items,
            )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Train Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]: Train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Validation Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader]]: Validation Dataloader.
        """
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Test Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader]]: Test Dataloader.
        """
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Predict Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader]]: Predict Dataloader.
        """
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )
