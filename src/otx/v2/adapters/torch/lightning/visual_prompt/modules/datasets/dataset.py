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

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

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

if TYPE_CHECKING:
    import albumentations as al
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
        pipeline: dict | list | None = None,
    ) -> None:
        """Initializes a Dataset object.

        Args:
            dataset (DatasetEntity): The dataset to use.
            image_size (int): The size of the images in the dataset.
            mean (list[float]): The mean values for normalization.
            std (list[float]): The standard deviation values for normalization.
            offset_bbox (int, optional): The offset for bounding boxes. Defaults to 0.
            pipeline (dict | list | None, optional): The pipeline to use for data transformation.
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


def get_image_filenames(path: str | Path) -> list[Path]:
    """Get image filenames.

    Args:
        path (str | Path): Path to image or image-folder.

    Returns:
        list[Path]: List of image filenames

    """
    image_filenames: list[Path] = []

    if isinstance(path, str):
        path = Path(path)

    if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
        image_filenames = [path]

    if path.is_dir():
        image_filenames = [p for p in path.glob("**/*") if p.suffix.lower() in IMG_EXTENSIONS]

    if not image_filenames:
        msg = f"Found 0 images in {path}"
        raise ValueError(msg)

    return image_filenames


def get_image_height_and_width(image_size: int | tuple[int, int]) -> tuple[int, int]:
    """Get image height and width from ``image_size`` variable.

    Args:
        image_size (int | tuple[int, int] | None, optional): Input image size.

    Raises:
        ValueError: Image size not None, int or tuple.

    Examples:
        >>> get_image_height_and_width(image_size=256)
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256))
        (256, 256)

        >>> get_image_height_and_width(image_size=(256, 256, 3))
        (256, 256)

        >>> get_image_height_and_width(image_size=256.)
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in get_image_height_and_width
        TypeError: ``image_size`` could be either int or tuple[int, int]

    Returns:
        tuple[int | None, int | None]: A tuple containing image height and width values.
    """
    if isinstance(image_size, int):
        height_and_width = (image_size, image_size)
    elif isinstance(image_size, tuple):
        height_and_width = int(image_size[0]), int(image_size[1])
    else:
        msg = "``image_size`` could be either int or tuple[int, int]"
        raise TypeError(msg)

    return height_and_width


def read_image(path: str | Path, image_size: int | tuple[int, int] | None = None) -> np.ndarray:
    """Read image from disk in RGB format.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_image("test_image.jpg")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_size:
        # This part is optional, where the user wants to quickly resize the image
        # with a one-liner code. This would particularly be useful especially when
        # prototyping new ideas.
        height, width = get_image_height_and_width(image_size)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)

    return image


class VisualPromptInferenceDataset(Dataset):
    """Inference Dataset to perform prediction.

    Args:
        path (str | Path): Path to an image or image-folder.
        transform (A.Compose | None, optional): Albumentations Compose object describing the transforms that are
            applied to the inputs.
        image_size (int | tuple[int, int] | None, optional): Target image size
            to resize the original image. Defaults to None.
    """

    def __init__(
        self,
        path: str | Path,
        transform: al.Compose | None = None,
        image_size: int = 1024,
    ) -> None:
        """Initializes a Dataset object.

        Args:
            path (str or Path): The path to the directory containing the images.
            transform (al.Compose or None): A composition of image transformations to apply to the images.
                Defaults to None.
            image_size (int): The size of the images to be loaded. Defaults to 1024.
        """
        super().__init__()

        self.image_filenames = get_image_filenames(path)

        if transform is None:
            self.transform = get_transform(image_size=image_size)
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> dict:
        """Get dataset item.

        Args:
            index (int): Index of the dataset sample.

        Returns:
            dict: Dataset item.
        """
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename)
        item: dict = {"index": index, "images": image}

        prompts = {
            "original_size": image.shape,
            "gt_masks": [],
            "bboxes": [],
            "points": [],
            "labels": [],
        }

        item.update({**prompts, "path": str(image_filename)})
        return self.transform(item)
