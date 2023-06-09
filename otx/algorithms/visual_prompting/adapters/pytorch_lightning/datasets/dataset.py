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
from otx.api.entities.shapes.polygon import Polygon
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
    """Visual Prompting Dataset Adaptor."""

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        dataset: DatasetEntity,
        transform: MultipleInputsCompose
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

    @staticmethod
    def polygons_to_mask(polygons: List[List[np.ndarray]], height: int, width: int) -> np.ndarray:
        # Initialize the mask as an array of zeros
        mask = np.zeros((len(polygons), height, width), dtype=np.uint8)

        # Loop over the polygons
        for i, polygon in enumerate(polygons):
            # Create an empty mask for this polygon
            poly_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Convert the polygon to a numpy array and reshape to ROWSx1x2
            poly_array = np.array(polygon).reshape((-1, 1, 2)).astype(int)
            
            # Draw the polygon onto the mask
            cv2.fillPoly(poly_mask, [poly_array], 1)
            
            # Add the mask for this polygon to the overall mask
            mask[i] = poly_mask

        return mask

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
        # height, width = self.config.dataset.image_size

        # load annotations for item
        bboxes = []
        labels = []
        polygons = []

        width, height = dataset_item.width, dataset_item.height
        # From otx/algorithms/detection/adapters/mmdet/datasets/dataset.py
        for annotation in dataset_item.get_annotations(labels=self.labels, include_empty=False, preserve_id=True):
            box = ShapeFactory.shape_as_rectangle(annotation.shape)
            if min(box.width * width, box.height * height) < -1:
                continue

            class_indices = [
                self.label_idx[label.id] for label in annotation.get_labels(include_empty=False)
            ]

            n = len(class_indices)
            bboxes.extend([[box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height] for _ in range(n)])
            polygon = ShapeFactory.shape_as_polygon(annotation.shape)
            polygon = np.array([p for point in polygon.points for p in [point.x * width, point.y * height]])
            polygons.extend([[polygon] for _ in range(n)])
            labels.extend(class_indices)

        masks = self.polygons_to_mask(polygons, dataset_item.height, dataset_item.width)
        bboxes = np.array(bboxes)

        item.update(dict(
            original_size=(height, width),
            image=dataset_item.numpy,
            mask=masks,
            bbox=bboxes,
            label=labels,
            point=None, # TODO (sungchul): update point information
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

        image_size = self.config.dataset.image_size
        if isinstance(image_size, int):
            image_size = [image_size]

        if stage == "fit" or stage is None:
            self.train_otx_dataset = self.dataset.get_subset(Subset.TRAINING)
            self.val_otx_dataset = self.dataset.get_subset(Subset.VALIDATION)

            # TODO (sungchul): distinguish between train and val config here
            self.train_transform = self.val_transform = MultipleInputsCompose([
                ResizeLongestSide(target_length=max(image_size)),
                Pad(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            ])

        if stage == "test":
            self.test_otx_dataset = self.dataset.get_subset(Subset.TESTING)
            self.test_transform = MultipleInputsCompose([
                ResizeLongestSide(target_length=max(image_size)),
                Pad(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            ])

        if stage == "predict":
            self.predict_otx_dataset = self.dataset
            self.predict_transform = MultipleInputsCompose([
                ResizeLongestSide(target_length=max(image_size)),
                Pad(),
                transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            ])

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
        dataset = OTXVIsualPromptingDataset(self.config, self.train_otx_dataset, self.train_transform)
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
        dataset = OTXVIsualPromptingDataset(self.config, self.val_otx_dataset, self.val_transform)
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
        dataset = OTXVIsualPromptingDataset(self.config, self.test_otx_dataset, self.test_transform)
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
        dataset = OTXVIsualPromptingDataset(self.config, self.predict_otx_dataset, self.predict_transform)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.eval_batch_size,
            num_workers=self.config.dataset.num_workers,
            collate_fn=collate_fn,
        )
