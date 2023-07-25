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

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import (
    MultipleInputsCompose,
    Pad,
    ResizeLongestSide,
    collate_fn,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.polygon import Polygon
from otx.api.entities.subset import Subset
from otx.api.utils.shape_factory import ShapeFactory

logger = get_logger()


def get_transform(
    image_size: int = 1024, mean: List[float] = [123.675, 116.28, 103.53], std: List[float] = [58.395, 57.12, 57.375]
) -> MultipleInputsCompose:
    """Get transform pipeline.

    Args:
        image_size (int): Size of image. Defaults to 1024.
        mean (List[float]): Mean for normalization. Defaults to [123.675, 116.28, 103.53].
        std (List[float]): Standard deviation for normalization. Defaults to [58.395, 57.12, 57.375].

    Returns:
        MultipleInputsCompose: Transform pipeline.
    """
    return MultipleInputsCompose(
        [
            ResizeLongestSide(target_length=image_size),
            Pad(),
            transforms.Normalize(mean=mean, std=std),
        ]
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
    gt_mask = cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)
    return gt_mask


def generate_bbox(  # noqa: D417
    x1: int, y1: int, x2: int, y2: int, width: int, height: int, offset_bbox: int = 0
) -> List[int]:
    """Generate bounding box.

    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates. # type: ignore
        width (int): Width of image.
        height (int): Height of image.
        offset_bbox (int): Offset to apply to the bounding box, defaults to 0.

    Returns:
        List[int]: Generated bounding box.
    """

    def get_randomness(length: int) -> int:
        if offset_bbox == 0:
            return 0
        return np.random.normal(0, min(length * 0.1, offset_bbox))

    bbox = [
        max(0, x1 + get_randomness(width)),
        max(0, y1 + get_randomness(height)),
        min(width, x2 + get_randomness(width)),
        min(height, y2 + get_randomness(height)),
    ]
    return bbox


def generate_bbox_from_mask(gt_mask: np.ndarray, width: int, height: int) -> List[int]:
    """Generate bounding box from given mask.

    Args:
        gt_mask (np.ndarry): Mask to generate bounding box.
        width (int): Width of image.
        height (int): Height of image.

    Returns:
        List[int]: Generated bounding box from given mask.
    """
    y_indices, x_indices = np.where(gt_mask == 1)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return generate_bbox(x_min, y_min, x_max, y_max, width, height)


class OTXVisualPromptingDataset(Dataset):
    """Visual Prompting Dataset Adaptor.

    Args:
        dataset (DatasetEntity): Dataset entity.
        image_size (int): Target size to resize image.
        mean (List[float]): Mean for normalization.
        std (List[float]): Standard deviation for normalization.
        offset_bbox (int): Offset to apply to the bounding box, defaults to 0.
    """

    def __init__(
        self, dataset: DatasetEntity, image_size: int, mean: List[float], std: List[float], offset_bbox: int = 0
    ) -> None:

        self.dataset = dataset
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
    def get_prompts(dataset_item: DatasetItemEntity, dataset_labels: List[LabelEntity]) -> Dict[str, Any]:
        """Get propmts from dataset_item.

        Args:
            dataset_item (DatasetItemEntity): Dataset item entity.
            dataset_labels (List[LabelEntity]): Label information.

        Returns:
            Dict[str, Any]: Processed prompts with ground truths.
        """
        width, height = dataset_item.width, dataset_item.height
        bboxes: List[List[int]] = []
        points: List = []  # TBD
        gt_masks: List[np.ndarray] = []
        labels: List[ScoredLabel] = []
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

            # TODO (sungchul): generate random points from gt_mask

            # add labels
            labels.extend(annotation.get_labels(include_empty=False))

        bboxes = np.array(bboxes)
        return dict(
            original_size=(height, width),
            gt_masks=gt_masks,
            bboxes=bboxes,
            points=points,  # TODO (sungchul): update point information
            labels=labels,
        )

    def __getitem__(self, index: int) -> Dict[str, Union[int, List, Tensor]]:
        """Get dataset item.

        Args:
            index (int): Index of the dataset sample.

        Returns:
            Dict[str, Union[int, List, Tensor]]: Dataset item.
        """
        dataset_item = self.dataset[index]
        item: Dict[str, Union[int, Tensor]] = {"index": index, "images": dataset_item.numpy}

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
        item = self.transform(item)
        return item


class OTXVisualPromptingDataModule(LightningDataModule):
    """Visual Prompting DataModule.

    Args:
        config (Union[DictConfig, ListConfig]): Configuration.
        dataset (DatasetEntity): Dataset entity.
    """

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
            stage (Optional[str], optional): train/val/test stages, defaults to None.
        """
        if not stage == "predict":
            self.summary()

        image_size = self.config.image_size
        mean = self.config.normalize.mean
        std = self.config.normalize.std
        if stage == "fit" or stage is None:
            train_otx_dataset = self.dataset.get_subset(Subset.TRAINING)
            val_otx_dataset = self.dataset.get_subset(Subset.VALIDATION)

            self.train_dataset = OTXVisualPromptingDataset(
                train_otx_dataset, image_size, mean, std, offset_bbox=self.config.offset_bbox
            )
            self.val_dataset = OTXVisualPromptingDataset(val_otx_dataset, image_size, mean, std)

        if stage == "test":
            test_otx_dataset = self.dataset.get_subset(Subset.TESTING)
            self.test_dataset = OTXVisualPromptingDataset(test_otx_dataset, image_size, mean, std)

        if stage == "predict":
            predict_otx_dataset = self.dataset
            self.predict_dataset = OTXVisualPromptingDataset(predict_otx_dataset, image_size, mean, std)

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

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Train Dataloader.

        Returns:
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]: Train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,
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
