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

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.image import Image
from otx.v2.api.entities.label import LabelEntity
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.shapes.polygon import Polygon
from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.utils.shape_factory import ShapeFactory
from otx.v2.api.utils.logger import get_logger

from .pipelines import (
    MultipleInputsCompose,
    Pad,
    ResizeLongestSide,
    collate_fn,
)

if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import DictConfig, ListConfig

logger = get_logger()


def get_transform(
    image_size: int = 1024,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> MultipleInputsCompose:
    """Get transform pipeline.

    Args:
        image_size (int): Size of image. Defaults to 1024.
        mean (list[float]): Mean for normalization. Defaults to [123.675, 116.28, 103.53].
        std (list[float]): Standard deviation for normalization. Defaults to [58.395, 57.12, 57.375].

    Returns:
        MultipleInputsCompose: Transform pipeline.
    """
    if mean is None:
        mean = [123.675, 116.28, 103.53]
    if std is None:
        std = [58.395, 57.12, 57.375]
    return MultipleInputsCompose(
        [
            ResizeLongestSide(target_length=image_size),
            Pad(),
            transforms.Normalize(mean=mean, std=std),
        ],
    )


def convert_polygon_to_mask(shape: Polygon, width: int, height: int) -> np.ndarray:
    """Convert polygon to mask.

    Args:
        shape (Polygon): Polygon to convert.
        width (int): Width of image.
        height (int): Height of image.

    Returns:
        np.ndarray: Generated mask from given polygon.
    """
    polygon = ShapeFactory.shape_as_polygon(shape)
    contour = [[int(point.x * width), int(point.y * height)] for point in polygon.points]
    gt_mask = np.zeros(shape=(height, width), dtype=np.uint8)
    return cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)


def generate_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    offset_bbox: int = 0,
) -> list[int]:
    """Generate bounding box.

    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates. # type: ignore
        width (int): Width of image.
        height (int): Height of image.
        offset_bbox (int): Offset to apply to the bounding box, defaults to 0.

    Returns:
        list[int]: Generated bounding box.
    """

    def get_randomness(length: int) -> int:
        if offset_bbox == 0:
            return 0
        rng = np.random.default_rng()
        return int(rng.normal(0, min(int(length * 0.1), offset_bbox)))

    return [
        max(0, x1 + get_randomness(width)),
        max(0, y1 + get_randomness(height)),
        min(width, x2 + get_randomness(width)),
        min(height, y2 + get_randomness(height)),
    ]


def generate_bbox_from_mask(gt_mask: np.ndarray, width: int, height: int) -> list[int]:
    """Generate bounding box from given mask.

    Args:
        gt_mask (np.ndarry): Mask to generate bounding box.
        width (int): Width of image.
        height (int): Height of image.

    Returns:
        list[int]: Generated bounding box from given mask.
    """
    x_indices: np.ndarray
    y_indices: np.ndarray
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    y_indices, x_indices = np.where(gt_mask == 1)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return generate_bbox(x_min, y_min, x_max, y_max, width, height)


class OTXVisualPromptingDataset(Dataset):
    """Visual Prompting Dataset Adaptor."""

    def __init__(
        self,
        dataset: DatasetEntity,
        image_size: int,
        mean: list[float],
        std: list[float],
        offset_bbox: int = 0,
        pipeline: list | None = None,
    ) -> None:
        """Initializes a Dataset object.

        Args:
            dataset (DatasetEntity): The dataset to use.
            image_size (int): The size of the images in the dataset.
            mean (list[float]): The mean values for normalization.
            std (list[float]): The standard deviation values for normalization.
            offset_bbox (int, optional): The offset for bounding boxes. Defaults to 0.
        """
        self.dataset = dataset
        if pipeline is not None:
            self.transform = MultipleInputsCompose(pipeline)
        else:
            self.transform = get_transform(image_size, mean, std)
        self.offset_bbox = offset_bbox
        self.labels = dataset.get_labels()

    def __len__(self) -> int:
        """Get size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return len(self.dataset)

    @staticmethod
    def get_prompts(dataset_item: DatasetItemEntity, dataset_labels: list[LabelEntity]) -> dict:
        """Get propmts from dataset_item.

        Args:
            dataset_item (DatasetItemEntity): Dataset item entity.
            dataset_labels (list[LabelEntity]): Label information.

        Returns:
            dict: Processed prompts with ground truths.
        """
        width, height = dataset_item.width, dataset_item.height
        bboxes: list[list[int]] = []
        points: list = []  # TBD
        gt_masks: list[np.ndarray] = []
        labels: list[ScoredLabel] = []
        for annotation in dataset_item.get_annotations(labels=dataset_labels, include_empty=False, preserve_id=True):
            if isinstance(annotation.shape, Image):
                # use mask as-is
                gt_mask = annotation.shape.numpy.astype(np.uint8)
            elif isinstance(annotation.shape, Polygon):
                # convert polygon to mask
                gt_mask = convert_polygon_to_mask(annotation.shape, width, height)
            else:
                continue

            if gt_mask.sum() == 0:
                # pass no gt
                continue
            gt_masks.append(gt_mask)

            # generate bbox based on gt_mask
            bbox = generate_bbox_from_mask(gt_mask, width, height)
            bboxes.append(bbox)

            # add labels
            labels.extend(annotation.get_labels(include_empty=False))

        bboxes = np.array(bboxes)
        return {
            "original_size": (height, width),
            "gt_masks": gt_masks,
            "bboxes": bboxes,
            "points": points,
            "labels": labels,
        }

    def __getitem__(self, index: int) -> dict:
        """Get dataset item.

        Args:
            index (int): Index of the dataset sample.

        Returns:
            dict: Dataset item.
        """
        dataset_item = self.dataset[index]
        item: dict = {"index": index, "images": dataset_item.numpy}

        prompts = self.get_prompts(dataset_item, self.labels)
        if len(prompts["gt_masks"]) == 0:
            return {
                "images": [],
                "bboxes": [],
                "points": [],
                "gt_masks": [],
                "original_size": [],
                "path": [],
                "labels": [],
            }

        prompts["bboxes"] = np.array(prompts["bboxes"])
        item.update({**prompts, "path": dataset_item.media.path})
        return self.transform(item)


class OTXVisualPromptingDataModule(LightningDataModule):
    """Visual Prompting DataModule.

    Args:
        config (DictConfig | ListConfig): Configuration.
        dataset (DatasetEntity): Dataset entity.
    """

    def __init__(self, config: DictConfig | ListConfig, dataset: DatasetEntity) -> None:
        """Initializes a Dataset object.

        Args:
            config (DictConfig | ListConfig): The configuration for the dataset.
            dataset (DatasetEntity): The dataset to use.

        Attributes:
                train_otx_dataset (DatasetEntity): The training dataset.
                val_otx_dataset (DatasetEntity): The validation dataset.
                test_otx_dataset (DatasetEntity): The testing dataset.
                predict_otx_dataset (DatasetEntity): The prediction dataset.
        """
        super().__init__()
        self.config = config
        self.dataset = dataset

        self.train_otx_dataset: DatasetEntity
        self.val_otx_dataset: DatasetEntity
        self.test_otx_dataset: DatasetEntity
        self.predict_otx_dataset: DatasetEntity

    def setup(self, stage: str | None = None) -> None:
        """Setup Visual Prompting Data Module.

        Args:
            stage (str | None): train/val/test stages, defaults to None.
        """
        if stage != "predict":
            self.summary()

        image_size = self.config.image_size
        mean = self.config.normalize.mean
        std = self.config.normalize.std
        if stage == "fit" or stage is None:
            train_otx_dataset = self.dataset.get_subset(Subset.TRAINING)
            val_otx_dataset = self.dataset.get_subset(Subset.VALIDATION)

            self.train_dataset = OTXVisualPromptingDataset(
                train_otx_dataset,
                image_size,
                mean,
                std,
                offset_bbox=self.config.offset_bbox,
            )
            self.val_dataset = OTXVisualPromptingDataset(val_otx_dataset, image_size, mean, std)

        if stage == "test":
            test_otx_dataset = self.dataset.get_subset(Subset.TESTING)
            self.test_dataset = OTXVisualPromptingDataset(test_otx_dataset, image_size, mean, std)

        if stage == "predict":
            predict_otx_dataset = self.dataset
            self.predict_dataset = OTXVisualPromptingDataset(predict_otx_dataset, image_size, mean, std)

    def summary(self) -> None:
        """Print size of the dataset, number of images."""
        for subset in [Subset.TRAINING, Subset.VALIDATION, Subset.TESTING]:
            dataset = self.dataset.get_subset(subset)
            num_items = len(dataset)
            logger.info(
                "'%s' subset size: Total '%d' images.",
                subset,
                num_items,
            )

    def train_dataloader(self) -> DataLoader | (list[DataLoader] | dict[str, DataLoader]):
        """Train Dataloader.

        Returns:
            Union[DataLoader, list[DataLoader], Dict[str, DataLoader]]: Train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Validation Dataloader.

        Returns:
            Union[DataLoader, list[DataLoader]]: Validation Dataloader.
        """
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """Test Dataloader.

        Returns:
            Union[DataLoader, list[DataLoader]]: Test Dataloader.
        """
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> DataLoader | list[DataLoader]:
        """Predict Dataloader.

        Returns:
            Union[DataLoader, list[DataLoader]]: Predict Dataloader.
        """
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )
