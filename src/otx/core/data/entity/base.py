# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX base data entities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import IntEnum, auto
from typing import Any, Dict, Generic, Iterator, TypeVar

import numpy as np
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import to_image

from otx.core.types.task import OTXTaskType


@dataclass
class ImageInfo:
    """Meta info for image.

    :param img_id: Image id
    :param img_shape: Image shape after preprocessing
    :param ori_shape: Image shape right after loading it
    :param pad_shape: Image shape before padding
    :param scale_factor: Scale factor if the image is rescaled during preprocessing
    """

    img_idx: int
    img_shape: tuple[int, int]
    ori_shape: tuple[int, int]
    pad_shape: tuple[int, int]
    scale_factor: tuple[float, float]
    attributes: dict


class ImageType(IntEnum):
    """Enum to indicate the image type in `ImageInfo` class."""

    NUMPY = auto()
    TV_IMAGE = auto()
    NUMPY_LIST = auto()
    TV_IMAGE_LIST = auto()


T_OTXDataEntity = TypeVar(
    "T_OTXDataEntity",
    bound="OTXDataEntity",
)


@dataclass
class OTXDataEntity(Mapping):
    """Base data entity for OTX.

    This entity is the output of each OTXDataset,
    which can be go through the input preprocessing tranforms.

    :param task: OTX task definition
    :param image: Image tensor or list of Image tensor which can have different type according to `image_type`
        1) `image_type=ImageType.NUMPY`: H x W x C numpy image tensor
        2) `image_type=ImageType.TV_IMAGE`: C x H x W torchvision image tensor
        3) `image_type=ImageType.NUMPY_LIST`: List of H x W x C numpy image tensors
        3) `image_type=ImageType.TV_IMAGE_LIST`: List of C x H x W torchvision image tensors
    :param imgs_info: Meta information for images
    """

    image: np.ndarray | tv_tensors.Image | list[np.ndarray] | list[tv_tensors.Image]
    img_info: ImageInfo

    @property
    def task(self) -> OTXTaskType:
        """OTX task type definition."""
        msg = "OTXTaskType is not defined."
        raise RuntimeError(msg)

    @property
    def image_type(self) -> ImageType:
        """Image type definition."""
        if isinstance(self.image, np.ndarray):
            return ImageType.NUMPY
        if isinstance(self.image, tv_tensors.Image):
            return ImageType.TV_IMAGE
        if isinstance(self.image, list):
            if isinstance(self.image[0], np.ndarray):
                return ImageType.NUMPY_LIST
            if isinstance(self.image[0], tv_tensors.Image):
                return ImageType.TV_IMAGE_LIST
        raise TypeError(self.image)

    def to_tv_image(self: T_OTXDataEntity) -> T_OTXDataEntity:
        """Convert `self.image` to TorchVision Image if it is a Numpy array (inplace operation)."""
        if isinstance(self.image, tv_tensors.Image):
            return self

        self.image = to_image(self.image)
        return self

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            yield field.name

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        return getattr(self, key)

    def __len__(self) -> int:
        """Get the number of fields in this data entity."""
        return len(fields(self))


@dataclass
class OTXPredEntity(OTXDataEntity):
    """Data entity to represent the model output prediction."""

    score: np.ndarray | Tensor


T_OTXBatchDataEntity = TypeVar(
    "T_OTXBatchDataEntity",
    bound="OTXBatchDataEntity",
)


@dataclass
class OTXBatchDataEntity(Generic[T_OTXDataEntity]):
    """Base Batch data entity for OTX.

    This entity is the output of PyTorch DataLoader,
    which is the direct input of OTXModel.

    :param images: List of B numpy image tensors (C x H x W)
    :param imgs_info: Meta information for images
    """

    batch_size: int
    images: list[tv_tensors.Image]
    imgs_info: list[ImageInfo]

    @property
    def task(self) -> OTXTaskType:
        """OTX task type definition."""
        msg = "OTXTaskType is not defined."
        raise RuntimeError(msg)

    @classmethod
    def collate_fn(cls, entities: list[T_OTXDataEntity]) -> OTXBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        if (batch_size := len(entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        task = entities[0].task

        if not all(task == entity.task for entity in entities):
            msg = "collate_fn() input should include a single OTX task"
            raise RuntimeError(msg)

        if not all(entity.image_type == ImageType.TV_IMAGE for entity in entities):
            msg = "All entities should be torchvision's Image tensor before collate_fn()"
            raise RuntimeError(msg)

        return OTXBatchDataEntity(
            batch_size=batch_size,
            images=[entity.image for entity in entities],
            imgs_info=[entity.img_info for entity in entities],
        )


T_OTXBatchPredEntity = TypeVar(
    "T_OTXBatchPredEntity",
    bound="OTXBatchPredEntity",
)


@dataclass
class OTXBatchPredEntity(OTXBatchDataEntity):
    """Data entity to represent model output predictions."""

    scores: list[np.ndarray] | list[Tensor]


T_OTXBatchLossEntity = TypeVar(
    "T_OTXBatchLossEntity",
    bound="OTXBatchLossEntity",
)


class OTXBatchLossEntity(Dict[str, Tensor]):
    """Data entity to represent model output losses."""
