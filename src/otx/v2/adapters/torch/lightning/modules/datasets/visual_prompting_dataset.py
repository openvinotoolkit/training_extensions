"""Dataset & DataModule for OTX lightning modules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from datumaro.components.annotation import Mask, Polygon
from datumaro.components.dataset import Dataset as DatumDataset
from datumaro.plugins.transforms import PolygonsToMasks
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from otx.v2.adapters.torch.modules.utils.io import get_image_filenames, read_image
from otx.v2.adapters.torch.modules.utils.mask_to_bbox import generate_bbox_from_mask
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.subset import Subset
from otx.v2.api.utils.logger import get_logger

from .pipelines import MultipleInputsCompose, Pad, ResizeLongestSide, collate_fn

if TYPE_CHECKING:
    from pathlib import Path

    import albumentations as al
    from datumaro.components.dataset_base import DatasetItem as DatumDatasetItem
    from omegaconf import DictConfig, ListConfig

logger = get_logger()


def get_transform(
    image_size: int = 1024,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> MultipleInputsCompose:
    """Get transform pipeline.

    Args:
        image_size (int, optional): Size of image. Defaults to 1024.
        mean (list[float] | None, optional): Mean for normalization. Defaults to [123.675, 116.28, 103.53].
        std (list[float] | None, optional): Standard deviation for normalization. Defaults to [58.395, 57.12, 57.375].

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


class OTXVisualPromptingDataset(Dataset):
    """Visual Prompting Dataset Adaptor."""

    def __init__(
        self,
        dataset: DatumDataset,
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
        self.item_ids: list = [(item.id, item.subset) for item in self.dataset]

    def __len__(self) -> int:
        """Get size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return len(self.dataset)

    @staticmethod
    def get_prompts(dataset_item: DatumDatasetItem) -> dict:
        """Get propmts from dataset_item.

        Args:
            dataset_item (DatasetItemEntity): Dataset item entity.

        Returns:
            dict: Processed prompts with ground truths.
        """
        height, width = dataset_item.media.data.shape[:2]
        bboxes: list[list[int]] = []
        points: list = []  # TBD
        gt_masks: list[np.ndarray] = []
        labels: list[ScoredLabel] = []
        for annotation in dataset_item.annotations:
            if isinstance(annotation, Mask):
                # use mask as-is
                gt_mask = annotation.image
            elif isinstance(annotation, Polygon):
                # convert polygon to mask
                ann_mask = PolygonsToMasks.convert_polygon(annotation, width, height)
                gt_mask = ann_mask.image
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
            labels.extend([annotation.label])

        bboxes = np.array(bboxes)
        return {
            "original_size": (width, height),
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
        dataset_item = self.dataset.get(id=self.item_ids[index][0], subset=self.item_ids[index][1])

        prompts = self.get_prompts(dataset_item)
        if len(prompts["gt_masks"]) == 0:
            return {
                "index": index,
                "images": dataset_item.media.data.astype(np.uint8),
                "bboxes": [],
                "points": [],
                "gt_masks": [],
                "original_size": [],
                "labels": [],
            }

        item: dict = {"index": index, "images": dataset_item.media.data.astype(np.uint8)}
        prompts["bboxes"] = np.array(prompts["bboxes"])
        item.update({**prompts})
        return self.transform(item)


class OTXVisualPromptingDataModule(LightningDataModule):
    """Visual Prompting DataModule.

    Args:
        config (DictConfig | ListConfig): Configuration.
        dataset (DatasetEntity): Dataset entity.
    """

    def __init__(self, config: DictConfig | ListConfig, dataset: dict[Subset, DatumDataset]) -> None:
        """Initializes a Dataset object.

        Args:
            config (DictConfig | ListConfig): The configuration for the dataset.
            dataset (dict[Subset, DatumDataset]): The dataset to use.

        Attributes:
            train_otx_dataset (DatumDataset): The training dataset.
            val_otx_dataset (DatumDataset): The validation dataset.
            test_otx_dataset (DatumDataset): The testing dataset.
            predict_otx_dataset (DatumDataset): The prediction dataset.
        """
        super().__init__()
        self.config = config
        self.dataset = dataset

        self.train_otx_dataset: DatumDataset
        self.val_otx_dataset: DatumDataset
        self.test_otx_dataset: DatumDataset
        self.predict_otx_dataset: DatumDataset

    def setup(self, stage: str | None = None) -> None:
        """Setup Visual Prompting Data Module.

        Args:
            stage (str | None, optional): train/val/test stages, defaults to None.
        """
        if stage != "predict":
            self.summary()

        image_size = self.config.image_size
        mean = self.config.normalize.mean
        std = self.config.normalize.std
        if stage == "fit" or stage is None:
            train_otx_dataset = self.dataset.get(Subset.TRAINING)
            val_otx_dataset = self.dataset.get(Subset.VALIDATION)

            self.train_dataset = OTXVisualPromptingDataset(
                train_otx_dataset,
                image_size,
                mean,
                std,
                offset_bbox=self.config.offset_bbox,
            )
            self.val_dataset = OTXVisualPromptingDataset(val_otx_dataset, image_size, mean, std)

        if stage == "test":
            test_otx_dataset = self.dataset.get(Subset.TESTING)
            self.test_dataset = OTXVisualPromptingDataset(test_otx_dataset, image_size, mean, std)

        if stage == "predict":
            predict_otx_dataset = self.dataset.get(Subset.TESTING)
            self.predict_dataset = OTXVisualPromptingDataset(predict_otx_dataset, image_size, mean, std)

    def summary(self) -> None:
        """Print size of the dataset, number of images."""
        for subset in [Subset.TRAINING, Subset.VALIDATION, Subset.TESTING]:
            dataset: DatumDataset = self.dataset.get(subset, DatumDataset.from_iterable([]))
            num_items = len(dataset)
            logger.info(
                "'%s' subset size: Total '%d' images.",
                subset,
                num_items,
            )

    def train_dataloader(self) -> DataLoader | (list[DataLoader] | dict[str, DataLoader]):
        """Train Dataloader.

        Returns:
            DataLoader | (list[DataLoader] | dict[str, DataLoader]): Train dataloader.
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
            DataLoader | list[DataLoader]: Validation Dataloader.
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
            DataLoader | list[DataLoader]: Test Dataloader.
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
            DataLoader | list[DataLoader]: Predict Dataloader.
        """
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )


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
            transform (al.Compose | None, optional): A composition of image transformations to apply to the images.
                Defaults to None.
            image_size (int, optional): The size of the images to be loaded. Defaults to 1024.
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
