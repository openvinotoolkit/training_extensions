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
import torchvision.transforms as transforms
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import (
    MultipleInputsCompose,
    Pad,
    ResizeLongestSide,
    collate_fn,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.shape_factory import ShapeFactory

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
        offset_bbox: int = 0
    ) -> None:

        self.config = config
        self.dataset = dataset
        self.transform = transform
        self.offset_bbox = offset_bbox

        self.labels = dataset.get_labels()

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
        for annotation in dataset_item.get_annotations(labels=self.labels, include_empty=False, preserve_id=True):
            if isinstance(annotation.shape, Polygon) and not self.config.use_mask:
                # convert polygon to mask
                gt_mask = self.convert_polygon_to_mask(annotation.shape, width, height)
                gt_masks.append(gt_mask)

                bbox = self.generate_bbox_from_mask(gt_mask, width, height)
                bboxes.append(bbox)

                # TODO (sungchul): generate random points from gt_mask

            if self.config.use_mask:
                # TODO (sungchul): load mask from dataset
                logger.warning("Loading masks from dataset is not yet implemented.")

                # if using masks from dataset, we can use bboxes from dataset, too
                if isinstance(annotation.shape, Rectangle):
                    # load bbox from dataset
                    bbox = ShapeFactory.shape_as_rectangle(annotation.shape)
                    if min(bbox.width * width, bbox.height * height) < -1:
                        continue

                    bbox = self.generate_bbox(int(bbox.x1 * width), int(bbox.y1 * height), int(bbox.x2 * width), int(bbox.y2 * height), width, height)
                    bboxes.append(bbox)

                if isinstance(annotation.shape, Point):
                    logger.warning("Using points is not implemented yet.")

        if len(bboxes) == 0:
            # there is no bbox from dataset -> generate bboxes based on gt_masks
            # ex) self.config.use_mask = True, but there is no bbox in dataset
            for gt_mask in gt_masks:
                bbox = self.generate_bbox_from_mask(gt_mask, width, height)
                bboxes.append(bbox)

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

    def convert_polygon_to_mask(self, shape: Polygon, width: int, height: int) -> np.ndarray:
        """Convert polygon to mask.
        
        Args:
            shape (Polygon): 
            width (int): 
            height (int): 

        Returns:
            np.ndarray: Generated mask from given polygon.
        """
        polygon = ShapeFactory.shape_as_polygon(shape)
        contour = [[int(point.x * width), int(point.y * height)] for point in polygon.points]
        gt_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        gt_mask = cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)
        return gt_mask

    def generate_bbox(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> List[int]:
        """"""
        def get_randomness(length: int) -> int:
            if self.offset_bbox == 0:
                return 0
            return np.random.normal(0, min(length * 0.1, self.offset_bbox))

        bbox = [
            max(0, x1 + get_randomness(width)),
            max(0, y1 + get_randomness(height)),
            min(width, x2 + get_randomness(width)),
            min(height, y2 + get_randomness(height))
        ]
        return bbox

    def generate_bbox_from_mask(self, gt_mask: np.ndarray, width: int, height: int) -> List[int]:
        """Generate bounding box from given mask.

        Args:
            gt_mask (np.ndarry): 
            width (int):
            height (int):

        Returns:
            List[int]: Generated bounding box from given mask.
        """
        y_indices, x_indices = np.where(gt_mask == 1)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        return self.generate_bbox(x_min, y_min, x_max, y_max, width, height)


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

            self.train_dataset = OTXVIsualPromptingDataset(self.config, train_otx_dataset, train_transform, offset_bbox=self.config.offset_bbox)
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
