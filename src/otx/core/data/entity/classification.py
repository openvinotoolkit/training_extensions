# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX classification data entities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torchvision import tv_tensors

from otx.core.data.entity.base import (
    ImageInfo,
    OTXBatchDataEntity,
    OTXBatchPredEntity,
    OTXDataEntity,
    OTXPredEntity,
)
from otx.core.data.entity.multi_transform import MultiTransformDataEntity
from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    import numpy as np
    from torch import LongTensor


@register_pytree_node
@dataclass
class MulticlassClsDataEntity(OTXDataEntity):
    """Data entity for multi-class classification task.

    :param labels: labels as integer indices
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_CLASS_CLS

    labels: LongTensor


class MultiTransformClsDataEntity(MulticlassClsDataEntity, MultiTransformDataEntity):
    """Data entity for multi-class classification task."""


@dataclass
class MulticlassClsPredEntity(OTXPredEntity, MulticlassClsDataEntity):
    """Data entity to represent the multi-class classification model output prediction."""


@dataclass
class MulticlassClsBatchDataEntity(OTXBatchDataEntity[MulticlassClsDataEntity]):
    """Data entity for multi-class classification task.

    :param labels: A list of bbox labels as integer indices
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_CLASS_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[MulticlassClsDataEntity],
        stack_images: bool = True,
    ) -> MulticlassClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return MulticlassClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> MulticlassClsBatchDataEntity:
        """Pin memory for member tensor variables."""
        return super().pin_memory().wrap(labels=[label.pin_memory() for label in self.labels])


@dataclass
class MultiTransformClsBatchDataEntity(MulticlassClsBatchDataEntity):
    """Data entity for multi-class classification task.

    :param labels: A list of bbox labels as integer indices
    """

    images: dict[str, np.ndarray | tv_tensors.Image | list[np.ndarray] | list[tv_tensors.Image]]
    imgs_info: dict[str, list[ImageInfo]]

    @classmethod
    def collate_fn(
        cls,
        entities: list[MultiTransformClsDataEntity],
        stack_images: bool = True,
    ) -> MultiTransformClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        if (batch_size := len(entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        task = entities[0].task

        if not all(task == entity.task for entity in entities):
            msg = "collate_fn() input should include a single OTX task"
            raise RuntimeError(msg)

        multi_transform_keys = list(entities[0].transformed_entities.keys())
        images = {key: [entity.transformed_entities[key].image for entity in entities] for key in multi_transform_keys}
        like = next(iter(images[multi_transform_keys[0]]))

        if stack_images and not all(like.shape == img.shape for img in images[multi_transform_keys[0]]):  # type: ignore[union-attr]
            msg = (
                "You set stack_images as True, but not all images in the batch has same shape. "
                "In this case, we cannot stack images. Some tasks, e.g., detection, "
                "can have different image shapes among samples in the batch. However, if it is not your intention, "
                "consider setting stack_images as False in the config."
            )
            warnings.warn(msg, stacklevel=1)
            stack_images = False

        images = {key: tv_tensors.wrap(torch.stack(imgs), like=like) if stack_images else imgs for key, imgs in images.items()}
        imgs_info = {key: [entity.img_info for entity in entities] for key in multi_transform_keys}

        return MultiTransformClsBatchDataEntity(
            batch_size=batch_size,
            images=images,
            imgs_info=imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> MultiTransformClsBatchDataEntity:
        """Pin memory for member tensor variables."""
        return self.wrap(
            images=(
                {key: tv_tensors.wrap(images.pin_memory(), like=images) for key, images in self.images.items()}
            ),
            labels=[label.pin_memory() for label in self.labels],
        )


@dataclass
class MulticlassClsBatchPredEntity(OTXBatchPredEntity, MulticlassClsBatchDataEntity):
    """Data entity to represent model output predictions for multi-class classification task."""


@register_pytree_node
@dataclass
class MultilabelClsDataEntity(OTXDataEntity):
    """Data entity for multi-label classification task.

    :param labels: Multi labels represented as an one-hot vector.
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_LABEL_CLS

    labels: LongTensor


@dataclass
class MultilabelClsPredEntity(OTXPredEntity, MultilabelClsDataEntity):
    """Data entity to represent the multi-label classification model output prediction."""


@dataclass
class MultilabelClsBatchDataEntity(OTXBatchDataEntity[MultilabelClsDataEntity]):
    """Data entity for multi-label classification task.

    :param labels: A list of labels as integer indices
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.MULTI_LABEL_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[MultilabelClsDataEntity],
        stack_images: bool = True,
    ) -> MultilabelClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return MultilabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )

    def pin_memory(self) -> MultilabelClsBatchDataEntity:
        """Pin memory for member tensor variables."""
        return super().pin_memory().wrap(labels=[label.pin_memory() for label in self.labels])


@dataclass
class MultilabelClsBatchPredEntity(OTXBatchPredEntity, MultilabelClsBatchDataEntity):
    """Data entity to represent model output predictions for multi-label classification task."""


@register_pytree_node
@dataclass
class HlabelClsDataEntity(OTXDataEntity):
    """Data entity for H-label classification task.

    :param labels: labels as integer indices
    :param label_group: the group of the label
    """

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.H_LABEL_CLS

    labels: LongTensor


@dataclass
class HlabelClsPredEntity(OTXPredEntity, HlabelClsDataEntity):
    """Data entity to represent the H-label classification model output prediction."""


@dataclass
class HlabelClsBatchDataEntity(OTXBatchDataEntity[HlabelClsDataEntity]):
    """Data entity for H-label classification task.

    :param labels: A list of labels as integer indices
    :param label_groups: A list of label group
    """

    labels: list[LongTensor]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.H_LABEL_CLS

    @classmethod
    def collate_fn(
        cls,
        entities: list[HlabelClsDataEntity],
        stack_images: bool = True,
    ) -> HlabelClsBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader."""
        batch_data = super().collate_fn(entities, stack_images=stack_images)
        return HlabelClsBatchDataEntity(
            batch_size=batch_data.batch_size,
            images=batch_data.images,
            imgs_info=batch_data.imgs_info,
            labels=[entity.labels for entity in entities],
        )


@dataclass
class HlabelClsBatchPredEntity(OTXBatchPredEntity, HlabelClsBatchDataEntity):
    """Data entity to represent model output predictions for H-label classification task."""
