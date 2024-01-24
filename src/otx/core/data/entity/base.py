# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX base data entities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterator, TypeVar

import torchvision.transforms.v2.functional as F  # noqa: N812
from torch import Tensor, stack
from torch.utils._pytree import tree_flatten
from torchvision import tv_tensors

from otx.core.data.entity.utils import register_pytree_node
from otx.core.types.image import ImageColorChannel, ImageType
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    import numpy as np


class ImageInfo(tv_tensors.TVTensor):
    """Meta info for image.

    Attributes:
        img_id: Image id
        img_shape: Image shape (heigth, width) after preprocessing
        ori_shape: Image shape (heigth, width) right after loading it
        padding: Number of pixels to pad all borders (left, top, right, bottom)
        scale_factor: Scale factor (height, width) if the image is resized during preprocessing.
            Default value is `(1.0, 1.0)` when there is no resizing. However, if the image is cropped,
            it will lose the scaling information and be `None`.
        normalized: If true, this image is normalized with `norm_mean` and `norm_std`
        norm_mean: Mean vector used to normalize this image
        norm_std: Standard deviation vector used to normalize this image
        image_color_channel: Color channel type of this image, RGB or BGR.
        ignored_labels: Label that should be ignored in this image. Default to None.
    """

    img_idx: int
    img_shape: tuple[int, int]
    ori_shape: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    scale_factor: tuple[float, float] | None = (1.0, 1.0)
    normalized: bool = False
    norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_color_channel: ImageColorChannel = ImageColorChannel.RGB
    ignored_labels: list[int] | None = None

    @classmethod
    def _wrap(
        cls,
        dummy_tensor: Tensor,
        *,
        img_idx: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        scale_factor: tuple[float, float] | None = (1.0, 1.0),
        normalized: bool = False,
        norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        ignored_labels: list | None = None,
    ) -> ImageInfo:
        image_info = dummy_tensor.as_subclass(cls)
        image_info.img_idx = img_idx
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.padding = padding
        image_info.scale_factor = scale_factor
        image_info.normalized = normalized
        image_info.norm_mean = norm_mean
        image_info.norm_std = norm_std
        image_info.image_color_channel = image_color_channel
        image_info.ignored_labels = ignored_labels
        return image_info

    def __new__(  # noqa: D102
        cls,
        img_idx: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        scale_factor: tuple[float, float] | None = (1.0, 1.0),
        normalized: bool = False,
        norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        ignored_labels: list | None = None,
    ) -> ImageInfo:
        return cls._wrap(
            dummy_tensor=Tensor(),
            img_idx=img_idx,
            img_shape=img_shape,
            ori_shape=ori_shape,
            padding=padding,
            scale_factor=scale_factor,
            normalized=normalized,
            norm_mean=norm_mean,
            norm_std=norm_std,
            image_color_channel=image_color_channel,
            ignored_labels=ignored_labels,
        )

    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: tuple[()] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> ImageType:
        """Wrap an output (`torch.Tensor`) obtained from PyTorch function.

        For example, this function will be called when

        >>> img_info = ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10))
        >>> `_wrap_output()` will be called after the PyTorch function `to()` is called
        >>> img_info = img_info.to(device=torch.cuda)
        """
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))

        if isinstance(output, Tensor) and not isinstance(output, ImageInfo):
            image_info = next(x for x in flat_params if isinstance(x, ImageInfo))
            output = ImageInfo._wrap(
                dummy_tensor=output,
                img_idx=image_info.img_idx,
                img_shape=image_info.img_shape,
                ori_shape=image_info.ori_shape,
                padding=image_info.padding,
                scale_factor=image_info.scale_factor,
                normalized=image_info.normalized,
                norm_mean=image_info.norm_mean,
                norm_std=image_info.norm_std,
                image_color_channel=image_info.image_color_channel,
                ignored_labels=image_info.ignored_labels,
            )
        elif isinstance(output, (tuple, list)):
            image_infos = [x for x in flat_params if isinstance(x, ImageInfo)]
            output = type(output)(
                ImageInfo._wrap(
                    dummy_tensor=dummy_tensor,
                    img_idx=image_info.img_idx,
                    img_shape=image_info.img_shape,
                    ori_shape=image_info.ori_shape,
                    padding=image_info.padding,
                    scale_factor=image_info.scale_factor,
                    normalized=image_info.normalized,
                    norm_mean=image_info.norm_mean,
                    norm_std=image_info.norm_std,
                    image_color_channel=image_info.image_color_channel,
                    ignored_labels=image_info.ignored_labels,
                )
                for dummy_tensor, image_info in zip(output, image_infos)
            )
        return output

    @property
    def pad_shape(self) -> tuple[int, int]:
        """Image shape after padding."""
        h_img, w_img = self.img_shape
        left, top, right, bottom = self.padding
        return (h_img + top + bottom, w_img + left + right)

    @pad_shape.setter
    def pad_shape(self, pad_shape: tuple[int, int]) -> None:
        """Set padding from the given pad shape.

        Args:
            pad_shape: Padded image shape (height, width)
                which should be larger than this image shape.
                In addition, the padded image should be padded
                only for the right or bottom borders.
        """
        h_img, w_img = self.img_shape
        h_pad, w_pad = pad_shape

        if h_pad < h_img or w_pad < w_img:
            raise ValueError(pad_shape)

        left = top = 0
        right = w_pad - w_img
        bottom = h_pad - h_img
        self.padding = (left, top, right, bottom)


@F.register_kernel(functional=F.resize, tv_tensor_cls=ImageInfo)
def _resize_image_info(image_info: ImageInfo, size: list[int], **kwargs) -> ImageInfo:  # noqa: ARG001
    """Register ImageInfo to TorchVision v2 resize kernel."""
    if len(size) == 2:
        image_info.img_shape = (size[0], size[1])
    elif len(size) == 1:
        image_info.img_shape = (size[0], size[0])
    else:
        raise ValueError(size)

    ori_h, ori_w = image_info.ori_shape
    new_h, new_w = image_info.img_shape
    image_info.scale_factor = (new_h / ori_h, new_w / ori_w)
    return image_info


@F.register_kernel(functional=F.crop, tv_tensor_cls=ImageInfo)
def _crop_image_info(
    image_info: ImageInfo,
    height: int,
    width: int,
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    image_info.img_shape = (height, width)
    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.resized_crop, tv_tensor_cls=ImageInfo)
def _resized_crop_image_info(
    image_info: ImageInfo,
    size: list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    if len(size) == 2:
        image_info.img_shape = (size[0], size[1])
    elif len(size) == 1:
        image_info.img_shape = (size[0], size[0])
    else:
        raise ValueError(size)

    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.center_crop, tv_tensor_cls=ImageInfo)
def _center_crop_image_info(
    image_info: ImageInfo,
    output_size: list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    img_shape = F._geometry._center_crop_parse_output_size(output_size)  # noqa: SLF001
    image_info.img_shape = (img_shape[0], img_shape[1])

    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.pad, tv_tensor_cls=ImageInfo)
def _pad_image_info(
    image_info: ImageInfo,
    padding: int | list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    left, right, top, bottom = F._geometry._parse_pad_padding(padding)  # noqa: SLF001
    image_info.padding = (left, top, right, bottom)
    return image_info


@F.register_kernel(functional=F.normalize, tv_tensor_cls=ImageInfo)
def _normalize_image_info(
    image_info: ImageInfo,
    mean: list[float],
    std: list[float],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    image_info.normalized = True
    image_info.norm_mean = (mean[0], mean[1], mean[2])
    image_info.norm_std = (std[0], std[1], std[2])
    return image_info


T_OTXDataEntity = TypeVar(
    "T_OTXDataEntity",
    bound="OTXDataEntity",
)


@register_pytree_node
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
        return ImageType.get_image_type(self.image)

    def to_tv_image(self: T_OTXDataEntity) -> T_OTXDataEntity:
        """Convert `self.image` to TorchVision Image if it is a Numpy array (inplace operation)."""
        if isinstance(self.image, tv_tensors.Image):
            return self

        self.image = F.to_image(self.image)
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

    :param images: List of B numpy RGB image tensors (C x H x W) or
        An image tensor stacked with B RGB image tensors (B x C x H x W)
    :param imgs_info: Meta information for images
    """

    batch_size: int
    images: list[tv_tensors.Image] | tv_tensors.Image
    imgs_info: list[ImageInfo]

    @property
    def task(self) -> OTXTaskType:
        """OTX task type definition."""
        msg = "OTXTaskType is not defined."
        raise RuntimeError(msg)

    @property
    def images_type(self) -> ImageType:
        """Images type definition."""
        return ImageType.get_image_type(self.images)

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

        return OTXBatchDataEntity(
            batch_size=batch_size,
            images=[entity.image for entity in entities],
            imgs_info=[entity.img_info for entity in entities],
        )

    @property
    def stacked_images(self) -> tv_tensors.Image:
        """A stacked image tensor (B x C x H x W).

        if its `images` field is a list of image tensors,
        convert it to a 4D image tensor and return it.
        Otherwise, return it as is.
        """
        if isinstance(self.images, tv_tensors.Image):
            return self.images

        like = next(iter(self.images))
        return tv_tensors.wrap(stack(self.images, dim=0), like=like)

    def pin_memory(self: T_OTXBatchDataEntity) -> T_OTXBatchDataEntity:
        """Pin memory for member tensor variables."""
        # TODO(vinnamki): Keep track this issue
        # https://github.com/pytorch/pytorch/issues/116403
        self.images = (
            [tv_tensors.wrap(image.pin_memory(), like=image) for image in self.images]
            if isinstance(self.images, list)
            else tv_tensors.wrap(self.images.pin_memory(), like=self.images)
        )
        return self


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
